import Foundation
import MLXToolKit

/// Init-time configuration for `VoxCPM2TTSPackage` (C9): which HF repo to load and the
/// flow-matching generation defaults. Per-request text/voice/reference ride the `TTSRequest`
/// envelope (`VoiceSelector.referenceAudio` + `referenceTranscript`), not here.
///
/// **QuantConfigured (1.14 efficiency contract).** VoxCPM2 ships a single bf16 runtime, so the
/// quant alone identifies the footprint — `QuantConfigured` lets the governor charge the matching
/// declared `QuantFootprint` split (no per-mode `FootprintConfigured` needed; there is no second
/// variant at the same quant to disambiguate).
public struct VoxCPM2Configuration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// The single runtime quant. The loader casts bf16 → float32 for Metal precision (see the
    /// core's PORTING.md), but the *configured* quant — what the manifest footprint is keyed on —
    /// is bf16.
    public var quant: Quant { .bf16 }

    /// HuggingFace repo in the MLX VoxCPM2 layout (config.json + sharded safetensors + tokenizer).
    public var repo: String
    /// Euler ODE steps per latent patch (VoxCPM default 10; 7 trades a little quality for speed).
    public var inferenceTimesteps: Int
    /// Classifier-free guidance scale (nominal 2.0, range ~1.5–2.5).
    public var cfgValue: Float
    /// Flow-matching noise temperature (nominal 1.0).
    public var temperature: Float
    /// Explicit snapshot directory (dev escape hatch — never touches the network).
    public var modelDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Set by the engine from its
    /// `ModelStore`. Excluded from `Codable` (environment-specific, not portable config).
    public var modelsRootDirectory: URL?

    public init(repo: String = "mlx-community/VoxCPM2-bf16",
                inferenceTimesteps: Int = 10,
                cfgValue: Float = 2.0,
                temperature: Float = 1.0,
                modelDirectory: URL? = nil,
                modelsRootDirectory: URL? = nil) {
        self.repo = repo
        self.inferenceTimesteps = inferenceTimesteps
        self.cfgValue = cfgValue
        self.temperature = temperature
        self.modelDirectory = modelDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case repo, inferenceTimesteps, cfgValue, temperature
    }
}

// MARK: - Weight sources (auto-materialization, engine MAT gate)

extension VoxCPM2Configuration: WeightSourcing {
    /// Presence probe: what `ModelLoader.load(from:)` reads first — config, the shard index
    /// (which names the safetensors shards), and the tokenizer pair.
    static let requiredFiles = [
        "config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
    ]
    /// Download globs — the loadable snapshot (shards via `model*.safetensors`), skipping the
    /// repo's README/sample wavs.
    static let snapshotGlobs = [
        "config.json", "model*.safetensors", "model.safetensors.index.json",
        "tokenizer*.json", "special_tokens_map.json",
    ]

    public var weightSources: [WeightSource] {
        [WeightSource(role: "main", repo: repo, matching: Self.snapshotGlobs)]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        // Explicit local directory first (dev escape hatch).
        if let dir = modelDirectory,
           fm.fileExists(atPath: dir.appending(path: "config.json").path) {
            return []
        }
        // Then the ModelStore layout (`<root>/<org>/<name>`).
        if let dir = ModelStore(root: storeRoot).directory(for: repo),
           Self.requiredFiles.allSatisfy({ fm.fileExists(atPath: dir.appending(path: $0).path) }) {
            return []
        }
        return weightSources
    }

    /// The configuration with a nil `modelDirectory` resolved to the store layout — what `load()`
    /// uses AFTER materialization. An explicit directory always wins.
    public func resolved(storeRoot: URL?) -> VoxCPM2Configuration {
        var cfg = self
        if cfg.modelDirectory == nil {
            cfg.modelDirectory = ModelStore(root: storeRoot).directory(for: repo)
        }
        return cfg
    }
}

// MARK: - Cold-start prewarm

extension VoxCPM2Configuration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        // Store-resolved snapshot directory; the prewarmer scans it recursively for weight files
        // and skips it when absent (first launch — nothing on disk yet).
        [resolved(storeRoot: modelsRootDirectory).modelDirectory].compactMap { $0 }
    }
}
