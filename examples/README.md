# Examples

Two minimal end-to-end examples for using the published `bge-reranker-base-coreml` artifact. Both score one `(query, doc)` pair and print the sigmoid-mapped relevance.

## Python (`predict.py`)

Works against either a local conversion (`build/`) or a tagged release on Hugging Face. Useful for smoke-testing a freshly-converted artifact before publishing.

```sh
# Pre-publish: use the local build/ dir.
pixi run python examples/predict.py

# Post-publish: pull from the Hub.
pixi run python examples/predict.py --source hub --tag v0.1-ane
pixi run python examples/predict.py --source hub --tag v0.1-cpugpu

# Custom pair.
pixi run python examples/predict.py --query "git rebase tutorial" --doc "git rebase replays commits onto another base."
```

Requires macOS — `coremltools.MLModel` only loads on Apple platforms.

## Swift (`swift/`)

Mirrors the integration pattern the Juice macOS app uses: download a tagged commit via `swift-transformers`' `HubApi.snapshot(...)`, load the `model.mlpackage` with the right `MLComputeUnits`, build the XLM-R paired-input template manually (`<s> query </s></s> doc </s>` — `swift-transformers` does not expose `encode(text:textPair:)` for the Unigram path), and run `MLModel.prediction`.

```sh
cd examples/swift
swift build
swift run Predict --tag v0.1-ane --query "..." --doc "..."

# Switch to the cpuAndGPU fallback (e.g. if the ANE build fails to load):
swift run Predict --tag v0.1-cpugpu --cpugpu --query "..." --doc "..."
```

Requires macOS 15+ (matches the `minimum_deployment_target` of the converted `.mlpackage`) and Xcode/Swift 5.10+.

The Swift example only supports downloading from the Hub. To exercise an unpublished local artifact, use `predict.py` with `--source local`.
