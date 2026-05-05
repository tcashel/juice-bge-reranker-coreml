// swift-tools-version: 5.10
//
// Minimal Swift Package showing how to load and call the bge-reranker-base
// Core ML artifact published at `tcashel/bge-reranker-base-coreml`. Mirrors the
// integration pattern the Juice macOS app uses.
//
// Run:
//   cd examples/swift
//   swift run Predict --tag v0.1-ane --query "..." --doc "..."

import PackageDescription

let package = Package(
    name: "Predict",
    platforms: [.macOS("15.0")],
    dependencies: [
        // swift-transformers gives us HubApi (snapshot download) and AutoTokenizer
        // (XLMRobertaTokenizer dispatches to UnigramTokenizer for this model).
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .executableTarget(
            name: "Predict",
            dependencies: [
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/Predict"
        ),
    ]
)
