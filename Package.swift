// swift-tools-version: 6.2
import PackageDescription

// mlx-voxcpm2-tts-swift — the MLXEngine `tts` package over VoxCPM2 (flow-matching, 48 kHz).
// A thin conformance layer: it wraps the standalone inference engine mlx-voxcpm-swift (product
// `VoxCPM`) the same way mlx-qwen-llm-swift wraps mlx-swift-lm and mlx-kokoro-tts-swift wraps
// mlx-audio-swift. The engine contract (MLXToolKit) is a local-path dep for in-workspace dev;
// the bespoke runtime (VoxCPM core) is pinned to a tagged release.
//
// Swift-port naming: `-swift` on the package/repo; module/product stays clean `MLXVoxCPM2TTS`.
let package = Package(
    name: "mlx-voxcpm2-tts-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "MLXVoxCPM2TTS", targets: ["MLXVoxCPM2TTS"]),
    ],
    dependencies: [
        // Bumped to 0.28.1 for Specialty.voiceClone (VoxCPM2's zero-shot cloning selection
        // axis); 0.27.0 brought the CAN cancellation-conformance gate
        // (MLXServeConformance.CancellationConformance).
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.28.1"),
        // v0.3.0 carries the per-patch cooperative cancellation bail in the autoregressive
        // loop (CAN gate); v0.2.0 added the cachedRefFeat/cachedPromptFeat E1 API.
        .package(url: "https://github.com/xocialize/mlx-voxcpm-swift.git", from: "0.3.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        // Native downloader for WeightSourcing auto-materialization.
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.9.0"),
    ],
    targets: [
        .target(
            name: "MLXVoxCPM2TTS",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "VoxCPM", package: "mlx-voxcpm-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            // VoxCPM's `ModelLoader.LoadResult` / `VoxCPMModel` (MLX + swift-transformers) aren't
            // Sendable-audited, so awaiting the nonisolated `ModelLoader.load` back into the
            // `@InferenceActor` trips strict region-isolation ("sending non-Sendable"). The engine
            // serializes all lifecycle on InferenceActor (no real concurrency), so v5 mode keeps
            // that a warning while `@InferenceActor` isolation still holds — same lever as Kokoro.
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXVoxCPM2TTSTests",
            dependencies: [
                "MLXVoxCPM2TTS",
                // Test-only: admissibility / manifest checks through the engine contract.
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                // The offline MAT-1..5 materialization gate.
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ]
        ),
    ]
)
