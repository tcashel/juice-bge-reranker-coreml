# juice-bge-reranker-coreml

Maintainer-side conversion tool: turn [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) (PyTorch) into an Apple-Neural-Engine-resident Core ML `.mlpackage` and publish it to Hugging Face Hub.

> This is **not** a runtime. It does not run on a user's machine. The Juice macOS app downloads the published artifact at first launch via `swift-transformers`. The consumer cannot bundle Python in its app payload, which is why this conversion lives here as maintainer-side infra.

## Outputs per release

Two `.mlpackage` variants are published to **`tcashel/bge-reranker-base-coreml`** as separate git tags:

| Tag | Compute units | Role |
|---|---|---|
| `v{X}-ane` | `cpuAndNeuralEngine` | Headline build. Every op verified ANE-resident. |
| `v{X}-cpugpu` | `cpuAndGPU` | Known-good fallback if the ANE build fails to load. |

The Juice Swift app pins by tag and embeds it in any per-model cache key, so rotating the tag invalidates downstream caches. See `MODEL_CARD.md` for the full integration contract — that file is the consumer-side spec.

## Stack

- **Python env:** `pixi` (Python 3.13). `pixi.toml` is canonical; no `requirements.txt`.
- **Lint + format:** `ruff` (`pixi run lint`, `pixi run fmt`).
- **Type check:** `ty` (`pixi run typecheck`).
- **Tests:** `pytest` (`pixi run test`).
- **Conversion:** `coremltools 9.x`, `transformers 5.x`, `torch 2.x`.
- **ANE rewrites:** small slice of Apple's [`ml-ane-transformers`](https://github.com/apple/ml-ane-transformers) reference (`LayerNormANE`, `MultiHeadAttention` with `Linear→Conv2d` 1×1, `(B, C, 1, S)` layout) is **vendored** under `vendor/ane_transformers/` because the upstream pip package strict-pins `torch<=1.11`, incompatible with our stack. The vendored code is plain PyTorch and runs on torch 2.x unchanged. License headers preserved.

## End-to-end run

```sh
pixi install
pixi run convert                         # downloads HF weights, produces build/{ane,cpugpu}.mlpackage + tokenizer/
pixi run verify-ane build/bge-reranker-base-ane.mlpackage     # asserts every op is ANE-resident; non-zero exit on fallback
pixi run bench --variants ane:build/bge-reranker-base-ane.mlpackage cpugpu:build/bge-reranker-base-cpugpu.mlpackage --update-model-card MODEL_CARD.md
pixi run test                             # 3 always-on unit tests
JUICE_RUN_DOWNLOAD_TESTS=1 pixi run test  # adds the HF↔ANE numerical equivalence test (~1 GB download)
```

After bench numbers look good (ANE p95 at batch=20/seq=256 < 200 ms and per-pair p95 < 15 ms), publish:

```sh
export HUGGINGFACE_TOKEN=hf_xxx
pixi run python publish.py --variant both --tag v0.1 --dry-run    # always dry-run first
pixi run python publish.py --variant both --tag v0.1 --confirm    # actually pushes + tags
```

`publish.py` refuses without `--confirm`. The `tcashel/bge-reranker-base-coreml` Model repo on HF must already exist (Public, MIT).

## Repo layout

```
juice-bge-reranker-coreml/
├─ pixi.toml                     # env + pixi tasks
├─ pyproject.toml                # ty + ruff + pytest config
├─ MODEL_CARD.md                 # integration contract — published as the HF repo README
├─ convert.py                    # PyTorch → Core ML (both variants)
├─ verify_ane.py                 # asserts every op is ANE-resident
├─ bench.py                      # p50/p95 latency table; stamps numbers into MODEL_CARD.md
├─ publish.py                    # uploads to HF, creates v{X}-{variant} tags
├─ src/
│   ├─ ane_xlm_roberta.py        # ANE port of XLMRobertaForSequenceClassification
│   ├─ weight_transfer.py        # HF → ANE state_dict remap (with Linear→Conv2d reshape)
│   └─ provenance.py             # JSON sidecar capturing source SHA, package versions, machine, perf
├─ vendor/ane_transformers/      # vendored Apple reference primitives (Apple sample-code license preserved)
└─ tests/
    └─ test_numerical_equivalence.py
```

## Architecture note: XLM-RoBERTa, not BERT

The encoder geometry is BERT-like (12L / 768H / 12 heads / GELU / post-LN), so Apple's reference primitives lower cleanly to ANE. But the upstream `config.json` declares `model_type: xlm-roberta`, and the integration-contract details differ from a vanilla BERT cross-encoder: the tokenizer is SentencePiece-Unigram (not WordPiece), special tokens are `<s>`/`</s>`/`<pad>` (not `[CLS]`/`[SEP]`/`[PAD]`), and the paired-input template is `<s> query </s></s> doc </s>` (note the doubled separator). Don't pattern-match on the BERT-like geometry and reach for `[CLS]`/`[SEP]`. `MODEL_CARD.md` is the authoritative contract.

## What this repo does NOT do

- Fine-tune or evaluate retrieval quality. Format conversion only.
- Swift integration code (lives in the Juice app).
- Multiple reranker architectures. One model in, one model out.
- Run on user machines. This is maintainer-side infra.
