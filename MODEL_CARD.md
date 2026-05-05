---
license: mit
language:
  - en
pipeline_tag: text-ranking
tags:
  - cross-encoder
  - reranker
  - core-ml
  - apple-silicon
  - ane
  - mteb
base_model: BAAI/bge-reranker-base
# MODEL_INDEX:auto-stamped — do not edit by hand
model-index:
  - name: bge-reranker-base-coreml
    results:
      - task:
          type: text-ranking
          name: Reranking
        dataset:
          type: mteb/scidocs-reranking
          name: SciDocs Reranking
          split: test
          revision: <not pinned>
        metrics:
          - type: ndcg_at_10
            name: nDCG@10
            value: 0.7415
          - type: map
            name: MAP
            value: 0.6743
# /MODEL_INDEX
---

# bge-reranker-base — Core ML (.mlpackage) for Apple Silicon

Core ML port of [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) targeting the **Apple Neural Engine** on M-series Macs. Produced by the maintainer-side conversion tool at [github.com/tcashel/juice-bge-reranker-coreml](https://github.com/tcashel/juice-bge-reranker-coreml). Consumed by the Juice macOS app via [`swift-transformers`](https://github.com/huggingface/swift-transformers).

This card **is the integration contract**. The Swift consumer relies on every section below; do not change a tensor name, shape, or token ID without bumping the variant tag and the `model_id` cache key on the consumer.

## Requirements

- **Apple Silicon Mac** (M1 / M2 / M3 / M4 / later). The headline `-ane` build requires the Apple Neural Engine.
- **macOS 15.0 (Sequoia) or later.** This is the artifact's `minimum_deployment_target`; older macOS versions cannot load the `.mlpackage`.
- **Swift consumer:** [`swift-transformers`](https://github.com/huggingface/swift-transformers) ≥ 1.3.0 for `HubApi` (snapshot download) and `AutoTokenizer` (XLM-R Unigram path). Direct `MLModel` load via `CoreML` also works.

## Usage

End-to-end working examples live in the [GitHub repo's `examples/`](https://github.com/tcashel/juice-bge-reranker-coreml/tree/main/examples) directory — both load the artifact, score one `(query, doc)` pair, and print the sigmoid-mapped relevance.

### Swift (`swift-transformers` + `CoreML`)

The canonical consumer pattern; mirrors what the Juice macOS app does. Full source at [`examples/swift/Sources/Predict/main.swift`](https://github.com/tcashel/juice-bge-reranker-coreml/blob/main/examples/swift/Sources/Predict/main.swift). Key steps:

```swift
import CoreML
import Hub
import Tokenizers

let repo = Hub.Repo(id: "tcashel/bge-reranker-base-coreml", type: .models)
let folder = try await HubApi.shared.snapshot(from: repo, revision: "v0.1-ane")
let tokenizer = try await AutoTokenizer.from(modelFolder: folder)

// XLM-R paired-input template (swift-transformers does not expose textPair for Unigram):
let bos: Int32 = 0, eos: Int32 = 2, pad: Int32 = 1
let q = tokenizer.encode(text: query, addSpecialTokens: false).map(Int32.init)
let d = tokenizer.encode(text: doc,   addSpecialTokens: false).map(Int32.init)
var ids: [Int32] = [bos] + q + [eos, eos] + d + [eos]
// ... pad to seq ∈ {128, 256, 512}, fill 20 batch rows with <pad>, then:

let config = MLModelConfiguration()
config.computeUnits = .cpuAndNeuralEngine
let model = try MLModel(contentsOf: folder.appendingPathComponent("model.mlpackage"), configuration: config)
let prediction = try await model.prediction(from: provider)
let logit = Double(truncating: prediction.featureValue(for: "logit")!.multiArrayValue![[0, 0]])
let score = 1.0 / (1.0 + exp(-logit))
```

Run:

```sh
cd examples/swift
swift run Predict --tag v0.1-ane --query "what is the capital of france?" --doc "Paris is the capital of France."
```

### Python (`coremltools` + `transformers` for tokenization)

For verifying the artifact end-to-end on macOS without a Swift toolchain. Full source at [`examples/predict.py`](https://github.com/tcashel/juice-bge-reranker-coreml/blob/main/examples/predict.py):

```python
import math, numpy as np
from coremltools.models import MLModel
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

folder = snapshot_download(repo_id="tcashel/bge-reranker-base-coreml", revision="v0.1-ane")
tokenizer = AutoTokenizer.from_pretrained(folder, use_fast=True)
model = MLModel(f"{folder}/model.mlpackage")

# Python's transformers builds the paired-input template internally:
enc = tokenizer(query, doc, padding="max_length", truncation=True, max_length=128, return_tensors="np")

# Pad up to the fixed batch=20 (read row 0 of the output, discard the rest):
ids  = np.full ((20, 1, 1, 128), 1, dtype=np.int32); ids [0, 0, 0, :] = enc["input_ids"][0]
mask = np.zeros((20, 1, 1, 128),    dtype=np.int32); mask[0, 0, 0, :] = enc["attention_mask"][0]

logit = float(model.predict({"input_ids": ids, "attention_mask": mask})["logit"][0, 0])
score = 1.0 / (1.0 + math.exp(-logit))
```

Run:

```sh
pixi run python examples/predict.py --source hub --tag v0.1-ane
```

## Identity

- **Source model:** `BAAI/bge-reranker-base` @ `<source_revision_sha>` (set by `convert.py`).
- **Conversion type:** PyTorch (FP32) → Core ML `.mlpackage` (FP16) with the [`apple/ml-ane-transformers`](https://github.com/apple/ml-ane-transformers) primitives (Conv2d 1×1 projections, BC1S layout, `LayerNormANE`) so the encoder lowers to the Apple Neural Engine. **This is a precision reduction (FP32 → FP16) and format conversion, not integer quantization** — there is no INT8/INT4 mapping. No fine-tuning, distillation, or weight pruning was applied; weights are bit-equivalent up to FP16 rounding.
- **Conversion stack:** see `<variant>_provenance.json` published alongside the artifact (records exact torch / transformers / coremltools versions and host machine).
- **License:** MIT (inherited from the upstream model).

## Variants

| Tag | Compute units | Intended use |
|---|---|---|
| `v{X}-ane` | `cpuAndNeuralEngine` | Headline build. The 12-layer encoder backbone (~924 ops: einsum, conv, softmax, layer_norm, gelu, transpose, residual add/mul) runs on the Apple Neural Engine. ~31 boundary ops (embedding gather over the 250k vocab, position-id arithmetic, mask construction, casts) dispatch to CPU; this is the Pareto frontier for XLM-RoBERTa-class models with very large vocabularies. Verified by `verify_ane.py`. M-series Macs only. |
| `v{X}-cpugpu` | `cpuAndGPU` | Known-good fallback — the same ANE port converted with `compute_units=CPU_AND_GPU`. Used by Swift if the `-ane` build fails to load (e.g. driver or macOS version mismatch). |

The Swift caller pins the tag in `Hub.snapshot(repo: "tcashel/bge-reranker-base-coreml", revision: "<tag>")` and embeds the same `<tag>` in the `model_id` cache key per Juice ADR 0006's `rerank_cache` table — rotating the tag invalidates the cache.

> **Repository layout.** This repo uses git **tags** (not subdirectories or sibling repos) to distinguish variants — `v{X}-ane` and `v{X}-cpugpu` point to different commits, each containing exactly one variant's files at the repo root (one `model.mlpackage`, one set of tokenizer files, one `provenance.json`). The `main` branch reflects whichever variant was published last, so consumers should always pin to a specific tag rather than reading from `main`. This layout optimizes for the Swift consumer: `HubApi.shared.snapshot(from:, revision: <tag>)` returns a flat ready-to-use directory.

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

Both variants share the same input shape contract — they're the same architecture (the ANE-friendly port) converted with different `compute_units`. The `(1, 1)` middle dims are constant on the cpuAndGPU path (no overhead) and required by ANE's BC1S layout on the ANE path.

| Name | Dtype | Shape | Notes |
|---|---|---|---|
| `input_ids` | `Int32` | `(20, 1, 1, S)` | `S ∈ {128, 256, 512}` via `EnumeratedShapes`. Token IDs in `[0, 250001]`. |
| `attention_mask` | `Int32` | `(20, 1, 1, S)` | `1` for real tokens, `0` for `<pad>`. |

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
### Variant: `ane`

| batch | seq | p50 (ms) | p95 (ms) | per-pair p95 (ms) |
|------:|----:|---------:|---------:|------------------:|
| 1 | 128 | 50.45 | 52.47 | 52.47 |
| 4 | 128 | 50.34 | 51.63 | 12.91 |
| 10 | 128 | 50.53 | 51.95 | 5.19 |
| 20 | 128 | 51.24 | 52.46 | 2.62 |
| 1 | 256 | 127.76 | 128.99 | 128.99 |
| 4 | 256 | 128.50 | 129.16 | 32.29 |
| 10 | 256 | 129.70 | 131.15 | 13.12 |
| 20 | 256 | 129.46 | 130.74 | 6.54 |
| 1 | 512 | 344.20 | 346.74 | 346.74 |
| 4 | 512 | 343.03 | 346.89 | 86.72 |
| 10 | 512 | 343.46 | 345.43 | 34.54 |
| 20 | 512 | 346.01 | 348.64 | 17.43 |
<!-- /BENCH:ane -->

<!-- BENCH:cpugpu -->
### Variant: `cpugpu`

| batch | seq | p50 (ms) | p95 (ms) | per-pair p95 (ms) |
|------:|----:|---------:|---------:|------------------:|
| 1 | 128 | 122.79 | 123.10 | 123.10 |
| 4 | 128 | 123.01 | 123.34 | 30.83 |
| 10 | 128 | 123.13 | 123.46 | 12.35 |
| 20 | 128 | 122.69 | 138.34 | 6.92 |
| 1 | 256 | 242.07 | 242.87 | 242.87 |
| 4 | 256 | 241.94 | 242.82 | 60.70 |
| 10 | 256 | 242.10 | 243.17 | 24.32 |
| 20 | 256 | 242.16 | 243.11 | 12.16 |
| 1 | 512 | 503.81 | 504.98 | 504.98 |
| 4 | 512 | 503.97 | 506.10 | 126.53 |
| 10 | 512 | 503.95 | 504.87 | 50.49 |
| 20 | 512 | 504.06 | 504.82 | 25.24 |
<!-- /BENCH:cpugpu -->

**Pass criterion (ANE variant):** `p95(batch=20, seq=256) < 200 ms` AND `per-pair p95 < 15 ms`. Matches Juice ADR 0006's reranker budget.

## Quality regression eval

Validates that the FP32 → FP16 + Core ML conversion preserved upstream behavior. Scored by `eval/quality_regression.py` against the [MTEB Reranking](https://huggingface.co/datasets?other=mteb&task_categories=task_categories%3Asentence-similarity) suite — the same benchmark family `BAAI/bge-reranker-base` is evaluated on. Pass criterion: `|Δ nDCG@10| < 0.005` per task vs the FP32 reference. Variant equivalence: scores apply to both `-ane` and `-cpugpu` (the FP16 weights inside each `.mlpackage` are bit-identical; only `compute_units` differs at load).

<!-- EVAL:reranking -->
### MTEB Reranking — FP32 reference vs Core ML FP16

_Variant equivalence: FP16 weights are bit-identical between `-ane` and `-cpugpu`; both inherit these numbers._

| Task | n queries | FP32 nDCG@10 | Core ML nDCG@10 | Δ nDCG@10 | FP32 MAP | Core ML MAP |
|---|---:|---:|---:|---:|---:|---:|
| scidocs-reranking | 3978 | 0.7410 | 0.7415 | +0.0005 | 0.6742 | 0.6743 |

**Pass criterion:** `|Δ nDCG@10| < 0.005` per task. FP32 baseline is `BAAI/bge-reranker-base` loaded with `attn_implementation="eager"`.

_Note on absolute scale:_ the nDCG@10 reported here (~0.74) reflects macro nDCG@10 over the test split's pre-ranked candidate pool (1 positive + ~29 negatives per query), which is structurally different from the full-corpus eval setup the BGE paper reports (~0.84). Δ vs the FP32 reference on the same setup is the meaningful regression signal; the absolute number is not directly comparable to the upstream paper.
<!-- /EVAL:reranking -->

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

## References

- **Source model:** [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base)
- **BGE family papers:**
  - [C-Pack: Packed Resources For General Chinese Embeddings (Xiao et al., 2023)](https://arxiv.org/abs/2309.07597)
  - [Making Large Language Models A Better Foundation For Dense Retrieval (Li et al., 2023)](https://arxiv.org/abs/2312.15503)
- **Apple Neural Engine + Core ML conversion:**
  - [`apple/ml-ane-transformers`](https://github.com/apple/ml-ane-transformers) — the reference primitives (LayerNormANE, Conv2d-projection MultiHeadAttention) we vendor for the ANE rewrite.
  - Apple Machine Learning Research — [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers).

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
