import Foundation
import AVFoundation
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
    /// The `.referenceAudio` clip couldn't be decoded into samples.
    case unreadableReferenceAudio(String)
}

/// An MLXEngine `tts` package over **VoxCPM2** — a flow-matching TTS (TSLM → FSQ → RALM →
/// LocDiT → AudioVAE) producing 48 kHz speech. A thin conformance wrapper over the standalone
/// `VoxCPM` engine (mlx-voxcpm-swift); all model logic lives there.
///
/// Engine-owned lifecycle (C13): the engine constructs from a `VoxCPM2Configuration`, pages
/// weights in with `load()` (downloads the HF snapshot on first run, then validates weight-key
/// parity), drives `run(_:)`, and reclaims with `unload()`. Returns the canonical `Audio` (.wav).
///
/// ## Voice selection
///
/// - `.auto` (and `.named` — VoxCPM2 has no named voices) → zero-shot synthesis.
/// - `.referenceAudio(clip)` → **reference-only cloning**: the clip's timbre/style conditions
///   generation via the VoxCPM2 ref prefix (`[103, ref…, 104, text…, 101]`); no transcript needed.
/// - `.referenceAudio(clip)` + `referenceTranscript` → **continuation cloning** (the official
///   "Ultimate Cloning"): tokens = `tokenizer(transcript + text)` with the clip as prompt audio;
///   the model warm-starts from real audio features and the result is the highest-fidelity clone.
///   `referenceTranscript` is the canonical `TTSRequest` field (contract 1.1.0).
///
/// Tokenization note: the MiniCPM tokenizer config sets `add_bos_token`, but VoxCPM2 was
/// trained without a BOS before the text — encoding with special tokens shifts every position
/// and causes onset artifacts (the bug fixed in our mlx-audio contribution, commit c7c79b8).
/// All paths here encode with `addSpecialTokens: false`.
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
                // the core), so the weights are the dominant resident term. Split per contract 1.14
                // (memory-harness.md): persistent weights in `residentBytes`, the transient (the
                // max of the autoregressive TSLM generation scratch and the diffusion LocDiT/AudioVAE
                // decode peak within one generate() call) in `peakActivationBytes` — the engine
                // reserves a single transient across residents.
                //
                // MEASURED ("measure the transient, don't derive it") via the package's own gated
                // bench (`VoxCPM2MemoryReport`, VX2_MEM=1) at a documented synth envelope — zero-shot,
                // ~154 chars → ~9 s of 48 kHz audio, 10 ODE steps: resident floor ~8.7 GB · worst
                // peak ~11.9 GB ⇒ transient ~3.2 GB. The old flat 11 GB *under-declared* the true
                // peak (≈11.9 GB) — the split is a correctness fix as much as an efficiency one.
                // Declared: weights 9.3 GB (measured floor) + transient 4.0 GB (~3.2 GB + ~20%
                // headroom) → a ~13.3 GB reserve, vs the old flat 11 GB that the peak overran.
                footprints: [QuantFootprint(
                    quant: .bf16,
                    residentBytes: 9_300_000_000,
                    peakActivationBytes: 4_000_000_000)],
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
                    summary: "VoxCPM2 flow-matching text-to-speech (48 kHz .wav): zero-shot, "
                        + "reference-audio timbre cloning, and continuation cloning with transcript.",
                    modes: [.neutral, .expressive]
                )
            ]
        )
    }

    private let configuration: Configuration
    private var loaded: ModelLoader.LoadResult?
    // Prompt-feat reuse (E1): encoding a reference clip runs the AudioVAE encoder + patchify
    // (`encodePromptAudio`) — the dominant per-call cloning cost. Long-form synthesis (e.g.
    // TTSOrchestratorKit anchor-to-first-segment) sends the *same* reference for every chunk, so
    // we memoize the encoded feature keyed by the reference bytes and pass it to the core as a
    // cached feat, skipping decode + resample + encode on reuse. Safe to hold: InferenceActor
    // serializes run() and the feature is read-only once eval'd.
    private var cachedPromptFeat: (key: Int, feat: MLXArray)?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard loaded == nil else { return }
        // Download (or reuse the cached) HF snapshot, then load weights + tokenizer. When the engine
        // has set a model-store root, point the Hub download base there (the caller holds
        // security-scoped access) so weights land in the chosen models folder, not the default cache.
        let hub = configuration.modelsRootDirectory.map { HubApi(downloadBase: $0) } ?? .shared
        let directory = try await hub.snapshot(from: configuration.repo)
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
        cachedPromptFeat = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let loaded else { throw PackageError.notLoaded }
        guard request.capability == .tts, let tts = request as? TTSRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        // No BOS, ever — see the tokenization note in the type doc.
        func tokenize(_ text: String) -> MLXArray {
            MLXArray(loaded.tokenizer.encode(text: text, addSpecialTokens: false).map { Int32($0) })
        }

        let result: VoxCPMGenerationResult
        switch tts.voice.selection {
        case .auto, .named:
            // Zero-shot. VoxCPM2 has no named-voice catalog; `.named` falls back to zero-shot.
            result = loaded.model.generate(
                inputIds: tokenize(tts.text),
                inferenceTimesteps: configuration.inferenceTimesteps,
                cfgValue: configuration.cfgValue,
                temperature: configuration.temperature
            )

        case .referenceAudio(let reference):
            // Encode the reference once and reuse it across repeats (E1). The encoded feat is a
            // pure function of the clip bytes, so the same key serves both cloning modes.
            let key = Self.promptKey(data: reference.data)
            let promptFeat: MLXArray
            if let cached = cachedPromptFeat, cached.key == key {
                promptFeat = cached.feat
            } else {
                // Decode the canonical clip and resample to the AudioVAE encoder rate (16 kHz).
                let encodeRate = loaded.model.audioVae.sampleRate
                let (decoded, sourceRate) = try Self.decodeToMono(reference)
                let refSamples = sourceRate == encodeRate
                    ? decoded
                    : SincResampler.resample(audio: decoded, from: sourceRate, to: encodeRate)
                let feat = loaded.model.encodePromptAudio(MLXArray(refSamples))
                eval(feat)  // materialize so reuse skips the encode graph, not just re-runs it
                cachedPromptFeat = (key, feat)
                promptFeat = feat
            }
            try Task.checkCancellation()

            if let transcript = tts.referenceTranscript, !transcript.isEmpty {
                // Continuation ("Ultimate Cloning"): tokens are tokenizer(transcript + text) —
                // direct concatenation, matching the reference implementations — with the clip's
                // encoded features warm-starting the autoregressive loop. The returned audio is
                // only the newly-generated continuation (the core never decodes the prompt).
                result = loaded.model.generate(
                    inputIds: tokenize(transcript + tts.text),
                    inferenceTimesteps: configuration.inferenceTimesteps,
                    cfgValue: configuration.cfgValue,
                    temperature: configuration.temperature,
                    cachedPromptFeat: promptFeat
                )
            } else {
                // Reference-only: timbre/style cloning via the VoxCPM2 ref prefix; the text is
                // unrelated to the clip, so no transcript is required.
                result = loaded.model.generate(
                    inputIds: tokenize(tts.text),
                    inferenceTimesteps: configuration.inferenceTimesteps,
                    cfgValue: configuration.cfgValue,
                    temperature: configuration.temperature,
                    cachedRefFeat: promptFeat
                )
            }
        }

        try Task.checkCancellation()
        let samples = result.audio.asType(.float32).asArray(Float.self) // 1-D mono, 48 kHz
        let sampleRate = result.sampleRate
        let wav = Self.encodeWAV16(samples: samples, sampleRate: sampleRate)
        return TTSResponse(audio: Audio(format: .wav, data: wav, sampleRate: sampleRate, channels: 1))
    }

    // MARK: - Prompt-feat cache key (E1)

    /// In-memory cache key for an encoded reference: the clip bytes. (Hasher is per-process
    /// seeded — fine, reuse happens within a single long-form run.)
    nonisolated static func promptKey(data: Data) -> Int {
        var hasher = Hasher()
        hasher.combine(data)
        return hasher.finalize()
    }

    // MARK: - Canonical Audio decode (serialized round-trip form, C3)

    /// Decodes a canonical `Audio` (.wav) artifact to mono float samples + sample rate.
    /// Multi-channel input is mixed down by averaging. Uses AVFoundation so any valid WAV
    /// layout (extra chunks, float PCM) decodes, not just the canonical 44-byte header.
    nonisolated static func decodeToMono(_ audio: Audio) throws -> (samples: [Float], sampleRate: Int) {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxcpm2-ref-\(UUID().uuidString).wav")
        try audio.data.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: tmp)
        } catch {
            throw VoxCPM2Error.unreadableReferenceAudio(error.localizedDescription)
        }
        let format = file.processingFormat
        let frames = AVAudioFrameCount(file.length)
        guard frames > 0, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else {
            throw VoxCPM2Error.unreadableReferenceAudio("empty audio")
        }
        try file.read(into: buffer)

        let channels = Int(format.channelCount)
        let count = Int(buffer.frameLength)
        guard channels > 0, count > 0, let channelData = buffer.floatChannelData else {
            throw VoxCPM2Error.unreadableReferenceAudio("no decodable samples")
        }
        var mono = [Float](repeating: 0, count: count)
        for channel in 0..<channels {
            let p = channelData[channel]
            for i in 0..<count { mono[i] += p[i] }
        }
        if channels > 1 {
            let inv = 1 / Float(channels)
            for i in 0..<count { mono[i] *= inv }
        }
        return (mono, Int(format.sampleRate))
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
