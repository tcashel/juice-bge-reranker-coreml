---
license: mit
language:
  - en
tags:
  - cross-encoder
  - reranker
  - core-ml
  - apple-silicon
  - ane
base_model: BAAI/bge-reranker-base
---

# bge-reranker-base — Core ML (.mlpackage) for Apple Silicon

Core ML port of [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) targeting the **Apple Neural Engine** on M-series Macs. Produced by the maintainer-side conversion tool at [github.com/tcashel/juice-bge-reranker-coreml](https://github.com/tcashel/juice-bge-reranker-coreml). Consumed by the Juice macOS app via [`swift-transformers`](https://github.com/huggingface/swift-transformers).

This card **is the integration contract**. The Swift consumer relies on every section below; do not change a tensor name, shape, or token ID without bumping the variant tag and the `model_id` cache key on the consumer.

## Identity

- **Source model:** `BAAI/bge-reranker-base` @ `<source_revision_sha>` (set by `convert.py`).
- **Conversion stack:** see `<variant>_provenance.json` published alongside the artifact (records exact torch / transformers / coremltools versions and host machine).
- **License:** MIT (inherited from the upstream model).

## Variants

| Tag | Compute units | Intended use |
|---|---|---|
| `v{X}-ane` | `cpuAndNeuralEngine` | Headline build. Every op verified ANE-resident by `verify_ane.py`. M-series Macs only. |
| `v{X}-cpugpu` | `cpuAndGPU` | Known-good fallback. Used by Swift if the `-ane` build fails to load (e.g. driver or macOS version mismatch). |

The Swift caller pins the tag in `Hub.snapshot(repo: "tcashel/bge-reranker-base-coreml", revision: "<tag>")` and embeds the same `<tag>` in the `model_id` cache key per Juice ADR 0006's `rerank_cache` table — rotating the tag invalidates the cache.

## Architecture

> **Correction vs ADR 0006:** ADR 0006 in the Juice repo describes this model as a "BERT cross-encoder." It is not. The upstream `config.json` declares `model_type: xlm-roberta`, `architectures: ["XLMRobertaForSequenceClassification"]`. The encoder *geometry* is BERT-like (12L/768H/12 heads, GELU, post-LN), but the tokenizer and special-token IDs are XLM-RoBERTa, not BERT. ADR 0006 should be patched in a follow-up Juice PR.

- 12 transformer encoder layers, hidden 768, 12 attention heads, intermediate FFN 3072.
- Single-segment model (`type_vocab_size = 1`).
- Classification head reads the `<s>` token (position 0): `dense(768→768) → tanh → out_proj(768→1)`. **No pooler.**
- Output: a single logit per pair. Apply `sigmoid` on the Swift side to get a relevance score in `[0, 1]`.

## Tokenizer

- **Class:** `XLMRobertaTokenizer` (SentencePiece-Unigram). Consumed in Swift via `swift-transformers`' `AutoTokenizer.from(modelFolder:)`, which dispatches to `UnigramTokenizer` for this `tokenizer_class`.
- **Files in this repo (under `tokenizer/`):** `tokenizer.json` (the fast-tokenizer file Swift consumes), `tokenizer_config.json`, `special_tokens_map.json`, `sentencepiece.bpe.model`. All four are required — `tokenizer.json` is the load path; the others are belt-and-braces.
- **Special tokens:**

  | Token | ID |
  |---|---|
  | `<s>` (BOS / CLS-equivalent) | 0 |
  | `<pad>` | 1 |
  | `</s>` (EOS / SEP-equivalent) | 2 |
  | `<unk>` | 3 |
  | `<mask>` | 250001 |

- **Padding:** right-side pad with `<pad>` (id 1).
- **Vocab size:** 250 002.
- **Max position embeddings:** 514 (= 512 max content tokens + `padding_idx + 1` offset).

## Paired-input template (must be constructed by the Swift consumer)

```
<s> {query} </s></s> {document} </s>
```

The doubled `</s></s>` separator is XLM-RoBERTa-specific (NOT the BERT `[SEP]` you might expect from ADR 0006's framing). `swift-transformers` does **not** expose `encode(text:textPair:)` for the Unigram path, so the Swift consumer must concatenate the template string itself before calling `encode(text:)`. Do not pre-tokenize and concatenate token IDs — let the tokenizer handle the special-token IDs.

## Truncation policy

If the tokenized template exceeds the target sequence length `S`, truncate the **document side from the right**. Never truncate the query — query terms drive both lexical and semantic match in the cross-encoder. Reserve 4 token slots for the special tokens (`<s>`, `</s>`, `</s>`, `</s>`):

```
max_doc_tokens = S - len(query_tokens) - 4
```

If `max_doc_tokens <= 0`, the query alone fills the budget — drop the document, the score is essentially noise, and the consumer should down-weight or skip this candidate at the orchestrator.

## Input tensors (Core ML)

| Name | Dtype | Shape (`-ane` variant) | Shape (`-cpugpu` variant) | Notes |
|---|---|---|---|---|
| `input_ids` | `Int32` | `(20, 1, 1, S)` | `(20, S)` | `S ∈ {128, 256, 512}` via `EnumeratedShapes`. Token IDs in `[0, 250001]`. |
| `attention_mask` | `Int32` | `(20, 1, 1, S)` | `(20, S)` | `1` for real tokens, `0` for `<pad>`. |

There is **no `token_type_ids` input** — `type_vocab_size = 1`, so token-type embedding is constant and folded internally.

Batch is fixed at 20 (matches the post-RRF top-20 candidate count from Juice's `docs/design/search.md`). Smaller actual batches must be padded with `<pad>` rows on the Swift side; the corresponding `attention_mask` rows should be all-zeros. The classification head still emits 20 logits — the consumer reads the first `actual_batch` of them and discards the rest.

## Output tensor

| Name | Dtype | Shape | Interpretation |
|---|---|---|---|
| `logit` | `Float32` | `(20, 1)` | Raw logit. Apply `sigmoid` to get relevance score in `[0, 1]`. |

## Position-ID computation (informational)

Position IDs inside the model are computed as:

```
position_ids[i] = (arange(S) + 2) * attention_mask + 1 * (1 - attention_mask)
```

i.e. real tokens get positions starting at 2 (= `pad_token_id + 1`), pad tokens get position 1 (= `pad_token_id`). This is bit-exact equivalent to HF's `create_position_ids_from_input_ids` when input is right-padded, and avoids `cumsum` (which doesn't lower cleanly to ANE). The Swift consumer **does not** pass position IDs as a model input.

## Performance

Measured by `bench.py` on the maintainer's machine (recorded under `<variant>_provenance.json → machine`). 50 warmup + 100 timed iterations per cell. `per-pair p95 = p95 / batch`.

<!-- BENCH:ane -->
_(filled in by `bench.py --update-model-card MODEL_CARD.md`)_
<!-- /BENCH:ane -->

<!-- BENCH:cpugpu -->
_(filled in by `bench.py --update-model-card MODEL_CARD.md`)_
<!-- /BENCH:cpugpu -->

**Pass criterion (ANE variant):** `p95(batch=20, seq=256) < 200 ms` AND `per-pair p95 < 15 ms`. Matches Juice ADR 0006's reranker budget.

## Failure modes the Swift consumer must handle

| Failure | Symptom | Recommended response |
|---|---|---|
| Download fails / hash mismatch | `Hub.snapshot` throws | Surface a one-line UI banner; reranker is bypassed; RRF order returned unchanged. |
| `MLModel` load fails (Intel Mac, missing ANE driver) | `MLModelLoadError` | Fall back to `-cpugpu` variant. If both fail, banner + RRF-only. |
| Per-query budget exceeded (>800 ms wall) | Cancel observed via `Task.cancel` | Return RRF order, log slow query. |
| Op fallback to CPU at runtime | Latency outliers in monitoring | Out of scope to detect from Swift; bench harness should catch this pre-publish via `verify_ane.py`. |

## Known limitations

- Apple Silicon only (`-ane` requires the Apple Neural Engine; Intel Macs must use `-cpugpu`).
- Fixed batch size 20. Smaller batches waste compute on pad rows; larger batches need a re-conversion.
- English-language reranking only (the upstream model is English; XLM-R's vocab supports more languages but the reranker has not been fine-tuned for them).
- FP16 internally on the ANE path — extreme inputs may show small numerical drift from the FP32 PyTorch reference. Tested within 1e-3 absolute tolerance on 16 fixed pairs; see `tests/test_numerical_equivalence.py`.

## How to reproduce

```sh
git clone https://github.com/tcashel/juice-bge-reranker-coreml
cd juice-bge-reranker-coreml
pixi install
pixi run convert         # produces build/bge-reranker-base-{ane,cpugpu}.mlpackage
pixi run verify-ane build/bge-reranker-base-ane.mlpackage
pixi run bench --variants ane:build/bge-reranker-base-ane.mlpackage cpugpu:build/bge-reranker-base-cpugpu.mlpackage --update-model-card MODEL_CARD.md
pixi run test
```

Publishing requires `HUGGINGFACE_TOKEN` in env and `--confirm`:

```sh
pixi run python publish.py --variant both --tag v0.1 --confirm
```
