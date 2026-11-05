import Foundation
import AVFoundation
import MLXToolKit
import MLX
import MLXRandom
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
    /// Weight sources are missing and there is no store root (or resolved directory) to
    /// materialize into.
    case missingWeights(String)
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
/// ## metaData keys (package-specific, C5)
///
/// - `seed` (int): seeds MLX's global random state (clamped to 32 bits, the fleet's guard
///   against runaway 64-bit seeds) right before generation, after any reference encode — so a
///   seeded request reproduces byte for byte, whether its reference came from the cache or not.
/// - `voiceCacheKey` (string) and `evictVoiceCacheKeys` (string or array of strings): below.
///
/// **Reference retention (MES Base §13 rules 4 and 6, AB-A-0251 — the mlx-qwen3-tts-swift 0.7.0
/// API, AB-A-0233).** Without `metaData["voiceCacheKey"]`, nothing derived from the reference
/// outlives the call. With a key (Gaea passes the MES asset id), the encoded prompt feature — a
/// function of the clip bytes alone, serving both cloning modes — is kept under that key, at
/// most `voiceCacheCapacity` keys, least recently used dropped first. A key that comes back
/// with other bytes is rebuilt; identical bytes under another key, or under none, never hit an
/// entry. `metaData["evictVoiceCacheKeys"]` drops keys before anything else runs (before the
/// weights check); with empty `text` the request is an eviction alone and returns empty audio
/// (a 44-byte WAV, zero samples, 48 kHz). `unload()` drops every key.
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
            specialties: [
                // Zero-shot voice cloning is VoxCPM2's core selection axis (shared with
                // IndexTTS2 / Qwen3-TTS / Gepard); no native emotion/duration control plane.
                SpecialtyWeight(.voiceClone, strength: 1.0),
            ],
            surfaces: [
                TTSContract.descriptor(
                    name: "voxcpm2-tts",
                    summary: "VoxCPM2 flow-matching text-to-speech (48 kHz .wav): zero-shot, "
                        + "reference-audio timbre cloning, and continuation cloning with transcript. "
                        + "E12 typed plane: `emotion` as .textDescription (a style description, rendered "
                        + "as the model's inline `(description)text` control) or .categorical over the "
                        + "shared 8-name vocabulary, rendered to a description. No native duration control.",
                    modes: [.neutral, .expressive],
                    // E12 declaration (contract 1.38.0, AB-A-0049 part 3): VoxCPM2 honors inline
                    // `(description)text` style control (listen-tested, ENHANCEMENTS E11), so both
                    // declared modes become that prefix. No native duration control — consumers
                    // time-stretch (E8 v2) for cue fit.
                    controls: TTSControls(emotionModes: [.categorical, .textDescription],
                                          supportsTargetDuration: false)
                )
            ]
        )
    }

    private let configuration: Configuration
    private var loaded: ModelLoader.LoadResult?
    // Prompt-feat reuse (E1), held only under a caller's `voiceCacheKey` (AB-A-0251): encoding a
    // reference clip runs the AudioVAE encoder + patchify (`encodePromptAudio`) — the dominant
    // per-call cloning cost, and long-form synthesis sends the same reference for every chunk.
    // Safe to hold: InferenceActor serializes run() and the feature is read-only once eval'd.
    var voiceCache = VoiceFeatureCache<CachedPromptFeat>(capacity: voiceCacheCapacity)
    /// Reference encodes since construction — observability for the cache tests.
    private(set) var promptFeatBuilds = 0

    /// How many `voiceCacheKey`s are kept at once.
    public nonisolated static let voiceCacheCapacity = 4

    /// The output rate (AudioVAE decode), for the eviction-only response that runs unloaded.
    nonisolated static let outputSampleRate = 48_000

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard loaded == nil else { return }

        // Auto-materialize missing sources into the engine store (dir-less configs only;
        // explicit directories never touch the network). Progress is forwarded per-file via
        // WeightDownloadProgress so the engine's PreparationMonitor surfaces `.downloading`.
        let storeRoot = configuration.modelsRootDirectory
        let missing = configuration.missingWeightSources(storeRoot: storeRoot)
        if !missing.isEmpty {
            guard let storeRoot else {
                throw VoxCPM2Error.missingWeights(
                    "no models root set and sources missing: \(missing.map(\.role).joined(separator: ", "))")
            }
            try await WeightMaterializer.materialize(missing, into: storeRoot)
        }
        try Task.checkCancellation()

        guard let directory = configuration.resolved(storeRoot: storeRoot).modelDirectory else {
            throw VoxCPM2Error.missingWeights("unresolved model directory (no store root)")
        }
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
        voiceCache.removeAll()
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS (not just drop refs)
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation
        // (engine ≥ 0.27.0). Mid-run cadence: the core's autoregressive patch loop bails once
        // per generated patch (`Task.isCancelled` break, mlx-voxcpm-swift ≥ 0.3.0 — the
        // TSLM/RALM/LocDiT interleave cadence) and the post-synthesis checkpoint below rethrows
        // the CancellationError unchanged (generate() is non-throwing by design).
        try Task.checkCancellation()
        guard request.capability == .tts, let tts = request as? TTSRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        // Eviction first, and without weights: a key dropped on asset deletion must go even
        // when the package isn't loaded (it then holds nothing, but the call still succeeds).
        if let keys = tts.metaData.stringList("evictVoiceCacheKeys") {
            for key in keys { voiceCache.remove(key) }
            if tts.text.isEmpty {
                return TTSResponse(audio: Audio(
                    format: .wav, data: Self.encodeWAV16(samples: [], sampleRate: Self.outputSampleRate),
                    sampleRate: Self.outputSampleRate, channels: 1))
            }
        }
        guard let loaded else { throw PackageError.notLoaded }

        // E12 typed plane (contract 1.38.0): a typed emotion becomes the model's inline
        // `(description)text` control on the text being generated — on every voice path, since
        // the model card says voice design, style control and cloning compose (E11).
        let text = try Self.controlledText(tts.text, emotion: tts.emotion)

        // No BOS, ever — see the tokenization note in the type doc.
        func tokenize(_ text: String) -> MLXArray {
            MLXArray(loaded.tokenizer.encode(text: text, addSpecialTokens: false).map { Int32($0) })
        }

        // The reference's prompt feature first (from the caller's key, or encoded for this call),
        // then the seed — so the RNG stream generation sees doesn't depend on the cache.
        var promptFeat: MLXArray?
        if case .referenceAudio(let reference) = tts.voice.selection {
            let cacheKey = tts.metaData.stringValue("voiceCacheKey").flatMap { $0.isEmpty ? nil : $0 }
            promptFeat = try encodedPrompt(reference, key: cacheKey, model: loaded.model)
            try Task.checkCancellation()
        }
        if let seed = tts.metaData.intValue("seed") {
            MLXRandom.seed(UInt64(bitPattern: Int64(seed)) & 0xFFFF_FFFF)
        }

        let result: VoxCPMGenerationResult
        switch tts.voice.selection {
        case .auto, .named:
            // Zero-shot. VoxCPM2 has no named-voice catalog; `.named` falls back to zero-shot.
            result = loaded.model.generate(
                inputIds: tokenize(text),
                inferenceTimesteps: configuration.inferenceTimesteps,
                cfgValue: configuration.cfgValue,
                temperature: configuration.temperature
            )

        case .referenceAudio:
            guard let promptFeat else { preconditionFailure("encoded above") }
            if let transcript = tts.referenceTranscript, !transcript.isEmpty {
                // Continuation ("Ultimate Cloning"): tokens are tokenizer(transcript + text) —
                // direct concatenation, matching the reference implementations — with the clip's
                // encoded features warm-starting the autoregressive loop. The returned audio is
                // only the newly-generated continuation (the core never decodes the prompt).
                result = loaded.model.generate(
                    inputIds: tokenize(transcript + text),
                    inferenceTimesteps: configuration.inferenceTimesteps,
                    cfgValue: configuration.cfgValue,
                    temperature: configuration.temperature,
                    cachedPromptFeat: promptFeat
                )
            } else {
                // Reference-only: timbre/style cloning via the VoxCPM2 ref prefix; the text is
                // unrelated to the clip, so no transcript is required.
                result = loaded.model.generate(
                    inputIds: tokenize(text),
                    inferenceTimesteps: configuration.inferenceTimesteps,
                    cfgValue: configuration.cfgValue,
                    temperature: configuration.temperature,
                    cachedRefFeat: promptFeat
                )
            }
        }

        // Post-synthesis checkpoint: the throwing half of the non-throwing-core pattern — if
        // the core's per-patch bail broke out early, this surfaces the CancellationError
        // unchanged (and discards the partial audio) before the CPU sample pull + WAV encode.
        try Task.checkCancellation()
        let samples = result.audio.asType(.float32).asArray(Float.self) // 1-D mono, 48 kHz
        let sampleRate = result.sampleRate
        let wav = Self.encodeWAV16(samples: samples, sampleRate: sampleRate)
        return TTSResponse(audio: Audio(format: .wav, data: wav, sampleRate: sampleRate, channels: 1))
    }

    // MARK: - Reference prompt feature (E1, AB-A-0251)

    /// The reference's encoded prompt feature: from the cache when `key` holds it for these
    /// bytes, else encoded now — and kept only when there is a key. The feature is a pure
    /// function of the clip bytes, so one entry serves both cloning modes.
    private func encodedPrompt(_ reference: Audio, key: String?, model: VoxCPMModel) throws -> MLXArray {
        let fingerprint = Self.promptKey(data: reference.data)
        if let key, let cached = voiceCache.value(for: key), cached.fingerprint == fingerprint {
            return cached.feat
        }
        // Decode the canonical clip and resample to the AudioVAE encoder rate (16 kHz).
        let encodeRate = model.audioVae.sampleRate
        let (decoded, sourceRate) = try Self.decodeToMono(reference)
        let refSamples = sourceRate == encodeRate
            ? decoded
            : SincResampler.resample(audio: decoded, from: sourceRate, to: encodeRate)
        let feat = model.encodePromptAudio(MLXArray(refSamples))
        eval(feat)  // materialize so reuse skips the encode graph, not just re-runs it
        promptFeatBuilds += 1
        if let key { voiceCache.insert(CachedPromptFeat(fingerprint: fingerprint, feat: feat), for: key) }
        return feat
    }

    /// Fingerprint of the reference bytes, so a key reused with other bytes re-encodes instead
    /// of serving the old voice. Never a cache key on its own. In-memory only (Hasher is
    /// per-process seeded).
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

extension MetaData {
    /// Convenience: read a string-valued metaData key.
    func stringValue(_ key: String) -> String? {
        if case .string(let value)? = self[key] { return value }
        return nil
    }

    /// Convenience: read a string or an array of strings (e.g. keys to evict).
    func stringList(_ key: String) -> [String]? {
        switch self[key] {
        case .string(let value)?: return [value]
        case .array(let values)?:
            return values.compactMap { if case .string(let v) = $0 { return v } else { return nil } }
        default: return nil
        }
    }

    /// Convenience: read an int-valued metaData key (e.g. the sampling seed).
    func intValue(_ key: String) -> Int? {
        if case .int(let value)? = self[key] { return value }
        return nil
    }
}

extension VoxCPM2TTSPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(VoxCPM2TTSPackage.self)
    }
}

// MARK: - E12 typed plane → inline `(description)text`

extension VoxCPM2TTSPackage {
    /// The typed `TTSRequest.emotion` rendered as VoxCPM2's inline control: `(description)text`,
    /// no separator — the form listen-tested in E11. `.textDescription` is used verbatim;
    /// `.categorical` renders one canonical name to the style description below. `.vector` /
    /// `.referenceAudio` are not declared, so the engine refuses them pre-admission and a direct
    /// caller is refused here, legibly. No typed emotion = the text untouched (no metaData compat
    /// path existed for this package; callers wrote the tags into the text themselves).
    nonisolated static func controlledText(_ text: String, emotion: TTSEmotion?) throws -> String {
        guard let emotion else { return text }
        switch emotion {
        case .textDescription(let description):
            return "(\(description))\(text)"
        case .categorical(let label):
            guard let resolved = E12Emotion.resolve(label) else {
                throw PackageError.unsupportedRequestFeature(
                    "emotion '\(label)' — known: \(E12Emotion.knownLabels)")
            }
            return "(\(resolved.styleDescription))\(text)"
        case .vector:
            throw PackageError.unsupportedRequestFeature(
                "emotion.vector — VoxCPM2 steers by an inline description; send .categorical or .textDescription")
        case .referenceAudio:
            throw PackageError.unsupportedRequestFeature(
                "emotion.referenceAudio — VoxCPM2 steers by an inline description; send .categorical or .textDescription")
        @unknown default:
            throw PackageError.unsupportedRequestFeature("emotion — mode not known to this package")
        }
    }
}

extension E12Emotion {
    /// The `(description)` each canonical category renders to — a delivery description in the
    /// style the model card demonstrates, not a bare label.
    public var styleDescription: String {
        switch self {
        case .happy: "cheerful and upbeat"
        case .angry: "angry and forceful"
        case .sad: "sad and subdued"
        case .afraid: "fearful and nervous"
        case .disgusted: "disgusted, with distaste"
        case .melancholic: "slow and melancholic"
        case .surprised: "surprised and astonished"
        case .calm: "calm and even"
        }
    }
}
