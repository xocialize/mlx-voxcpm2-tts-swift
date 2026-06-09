import Foundation
import MLXToolKit

/// Init-time configuration for `VoxCPM2TTSPackage` (C9): which HF repo to load and the
/// flow-matching generation defaults. Per-request text/voice ride the `TTSRequest`, not here.
///
/// v1 is zero-shot only; reference-audio cloning and text-driven voice design are follow-ups,
/// so this carries no voice/reference fields yet.
public struct VoxCPM2Configuration: PackageConfiguration {
    /// HuggingFace repo in the MLX VoxCPM2 layout (config.json + sharded safetensors + tokenizer).
    public var repo: String
    /// Euler ODE steps per latent patch (VoxCPM default 10; 7 trades a little quality for speed).
    public var inferenceTimesteps: Int
    /// Classifier-free guidance scale (nominal 2.0, range ~1.5–2.5).
    public var cfgValue: Float
    /// Flow-matching noise temperature (nominal 1.0).
    public var temperature: Float

    public init(repo: String = "mlx-community/VoxCPM2-bf16",
                inferenceTimesteps: Int = 10,
                cfgValue: Float = 2.0,
                temperature: Float = 1.0) {
        self.repo = repo
        self.inferenceTimesteps = inferenceTimesteps
        self.cfgValue = cfgValue
        self.temperature = temperature
    }
}
