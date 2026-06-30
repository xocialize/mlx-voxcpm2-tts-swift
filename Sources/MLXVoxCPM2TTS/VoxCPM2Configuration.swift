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
    /// Where weights are materialized. Set by the engine from its `ModelStore`; `nil` → the default
    /// swift-transformers cache. Excluded from `Codable` (environment-specific, not portable config).
    public var modelsRootDirectory: URL?

    public init(repo: String = "mlx-community/VoxCPM2-bf16",
                inferenceTimesteps: Int = 10,
                cfgValue: Float = 2.0,
                temperature: Float = 1.0,
                modelsRootDirectory: URL? = nil) {
        self.repo = repo
        self.inferenceTimesteps = inferenceTimesteps
        self.cfgValue = cfgValue
        self.temperature = temperature
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case repo, inferenceTimesteps, cfgValue, temperature
    }
}
