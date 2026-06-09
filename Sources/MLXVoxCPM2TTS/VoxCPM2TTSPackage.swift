import Foundation
import MLXToolKit
import MLX
import Hub
import Tokenizers
import VoxCPM

/// Errors specific to the VoxCPM2 package boundary (the engine propagates these to the caller
/// unchanged — `PackageError` only models the contract-level cases).
public enum VoxCPM2Error: Error, Equatable {
    /// The loaded checkpoint left model parameters unfilled — wrong/incompatible weights. Carries
    /// the count and a few example keys for diagnosis.
    case incompatibleWeights(missing: Int, examples: [String])
}

/// An MLXEngine `tts` package over **VoxCPM2** — a flow-matching TTS (TSLM → FSQ → RALM →
/// LocDiT → AudioVAE) producing 48 kHz speech. A thin conformance wrapper over the standalone
/// `VoxCPM` engine (mlx-voxcpm-swift); all model logic lives there.
///
/// Engine-owned lifecycle (C13): the engine constructs from a `VoxCPM2Configuration`, pages
/// weights in with `load()` (downloads the HF snapshot on first run, then validates weight-key
/// parity), drives `run(_:)`, and reclaims with `unload()`. Returns the canonical `Audio` (.wav).
///
/// **v1 is zero-shot only.** Reference-audio cloning and text-driven voice design are supported
/// by the underlying engine but not yet exposed here; any `VoiceSelector` falls back to the
/// model's default (zero-shot) voice.
@InferenceActor
public final class VoxCPM2TTSPackage: ModelPackage {
    public typealias Configuration = VoxCPM2Configuration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // VoxCPM2 weights are Apache-2.0; the Swift port (mlx-voxcpm-swift) is MIT.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "mlx-community/VoxCPM2-bf16", revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // ~2B params. The loader casts bf16 → float32 (Metal precision; see PORTING.md in
                // the core), so weights alone are ~10 GB resident; budget headroom for the KV cache
                // and the per-patch DiT/VAE activations puts the working set near ~11 GB.
                footprints: [QuantFootprint(quant: .bf16, residentBytes: 11_000_000_000)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                // Heavy autoregressive + diffusion lift — a capability floor as a sanity marker;
                // the MemoryGovernor still gates on the ~11 GB footprint.
                chipFloor: .pro
            ),
            specialties: [],
            surfaces: [
                TTSContract.descriptor(
                    name: "voxcpm2-tts",
                    summary: "VoxCPM2 flow-matching text-to-speech (48 kHz .wav), zero-shot.",
                    modes: [.neutral, .expressive]
                )
            ]
        )
    }

    private let configuration: Configuration
    private var loaded: ModelLoader.LoadResult?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard loaded == nil else { return }
        // Download (or reuse the cached) HF snapshot, then load weights + tokenizer.
        // (Redirecting the download into the engine's chosen models folder is a follow-up, as with
        // Kokoro — it depends on threading a custom Hub download base through here.)
        let directory = try await HubApi.shared.snapshot(from: configuration.repo)
        let result = try await ModelLoader.load(from: directory)

        // Weight-parity gate (the heart of "use the HF weights"): the core loader filters weights
        // to keys the model already has and silently drops the rest, so a mismatched/incompatible
        // checkpoint loads *partially* and would emit garbage audio with no other symptom. Refuse.
        guard result.missingKeys.isEmpty else {
            throw VoxCPM2Error.incompatibleWeights(
                missing: result.missingKeys.count,
                examples: Array(result.missingKeys.prefix(3))
            )
        }
        loaded = result
    }

    public func unload() async {
        loaded = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let loaded else { throw PackageError.notLoaded }
        guard request.capability == .tts, let tts = request as? TTSRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        // Zero-shot: tokenize text exactly as the reference harness does (the model appends its
        // own audio-start token internally). Voice selection is ignored in v1.
        let tokens = loaded.tokenizer.encode(text: tts.text)
        let inputIds = MLXArray(tokens.map { Int32($0) })

        let result = loaded.model.generate(
            inputIds: inputIds,
            inferenceTimesteps: configuration.inferenceTimesteps,
            cfgValue: configuration.cfgValue,
            temperature: configuration.temperature
        )

        let samples = result.audio.asType(.float32).asArray(Float.self) // 1-D mono, 48 kHz
        let sampleRate = result.sampleRate
        let wav = Self.encodeWAV16(samples: samples, sampleRate: sampleRate)
        return TTSResponse(audio: Audio(format: .wav, data: wav, sampleRate: sampleRate, channels: 1))
    }

    /// Encodes mono float samples as a 16-bit PCM WAV (broadly playable) in memory.
    nonisolated static func encodeWAV16(samples: [Float], sampleRate: Int) -> Data {
        let channels = 1
        let bitsPerSample = 16
        let blockAlign = channels * bitsPerSample / 8
        let byteRate = sampleRate * blockAlign
        let dataSize = samples.count * blockAlign

        var data = Data(capacity: 44 + dataSize)
        func ascii(_ s: String) { data.append(contentsOf: Array(s.utf8)) }
        func u32(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }

        ascii("RIFF"); u32(UInt32(36 + dataSize)); ascii("WAVE")
        ascii("fmt "); u32(16); u16(1) // PCM
        u16(UInt16(channels)); u32(UInt32(sampleRate)); u32(UInt32(byteRate))
        u16(UInt16(blockAlign)); u16(UInt16(bitsPerSample))
        ascii("data"); u32(UInt32(dataSize))

        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            var le = Int16(clamped * 32767).littleEndian
            withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
        }
        return data
    }
}

extension VoxCPM2TTSPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(VoxCPM2TTSPackage.self)
    }
}
