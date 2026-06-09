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
        .package(path: "../mlx-engine-swift"),
        .package(url: "https://github.com/xocialize/mlx-voxcpm-swift.git", from: "0.1.1"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        .target(
            name: "MLXVoxCPM2TTS",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "VoxCPM", package: "mlx-voxcpm-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
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
            ]
        ),
    ]
)
