import XCTest
import Foundation
import MLXServeCore
import MLXToolKit
@testable import MLXVoxCPM2TTS

/// Live voice-cache retention and seed tests (AB-A-0251). Gated: VX2_CLONE=1, run with
/// `swift test --build-system swiftbuild` (the metallib boundary), against the VoxCPM2 bf16
/// checkpoint already in the models store (VX2_MODELS, default /Volumes/DEV_VOL1/Models/Models).
/// Nothing is downloaded: a test that would need weights it can't find fails instead.
///
/// The references are the model's own seeded zero-shot voices, so the test needs no fixture
/// audio. Each test unloads before it returns: MLX state left live at process exit crashes
/// xctest's teardown.
final class VoiceCloneLiveTests: XCTestCase {
    static let referenceText = "The quick brown fox jumps over the lazy dog."
    static let targetText = "Every morning she walked to the old stone bridge."

    static var modelsRoot: URL {
        URL(fileURLWithPath: ProcessInfo.processInfo.environment["VX2_MODELS"]
            ?? "/Volumes/DEV_VOL1/Models/Models")
    }

    static var configuration: VoxCPM2Configuration {
        VoxCPM2Configuration(modelsRootDirectory: modelsRoot)
    }

    override func setUp() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VX2_CLONE"] == "1", "VX2_CLONE=1")
        let missing = Self.configuration.missingWeightSources(storeRoot: Self.modelsRoot)
        XCTAssertTrue(missing.isEmpty, "weights not in \(Self.modelsRoot.path): \(missing.map(\.role))")
        if !missing.isEmpty { throw XCTSkip("no weights") }
    }

    @InferenceActor
    static func loadedPackageAndReference() async throws -> (VoxCPM2TTSPackage, Audio) {
        let package = VoxCPM2TTSPackage(configuration: configuration)
        try await package.load()
        let response = try await package.run(TTSRequest(
            text: referenceText, metaData: ["seed": .int(7)])) as! TTSResponse
        return (package, response.audio)
    }

    static func clone(_ reference: Audio, transcript: String? = nil,
                      key: String? = nil, seed: Int? = 42, text: String = targetText) -> TTSRequest {
        var meta = MetaData()
        if let key { meta["voiceCacheKey"] = .string(key) }
        if let seed { meta["seed"] = .int(seed) }
        return TTSRequest(text: text, voice: VoiceSelector(.referenceAudio(reference)),
                          referenceTranscript: transcript, metaData: meta)
    }

    static func evict(_ keys: [String]) -> TTSRequest {
        TTSRequest(text: "", metaData: ["evictVoiceCacheKeys": .array(keys.map { .string($0) })])
    }

    static func audioOf(_ response: any CapabilityResponse) throws -> Audio {
        try XCTUnwrap(response as? TTSResponse).audio
    }

    // MARK: - Rule 4: no key, nothing outlives the call

    @InferenceActor
    func testCloneWithoutKeyLeavesNothingCached() async throws {
        let (package, reference) = try await Self.loadedPackageAndReference()

        let refOnly = try Self.audioOf(await package.run(Self.clone(reference)))
        XCTAssertTrue(package.voiceCache.isEmpty, "an unkeyed reference-only clone left \(package.voiceCache.keys)")
        let continuation = try Self.audioOf(await package.run(
            Self.clone(reference, transcript: Self.referenceText)))
        XCTAssertTrue(package.voiceCache.isEmpty, "an unkeyed continuation clone left \(package.voiceCache.keys)")
        XCTAssertEqual(package.promptFeatBuilds, 2, "an unkeyed reference was reused")
        for audio in [refOnly, continuation] {
            XCTAssertGreaterThan(audio.data.count, 44 + 2 * 48_000, "under a second of audio")
        }
        await package.unload()
    }

    // MARK: - Rule 6: reuse only under its key, gone after eviction

    @InferenceActor
    func testKeyedCloneIsReusedOnlyForItsKeyAndGoneAfterEviction() async throws {
        let (package, reference) = try await Self.loadedPackageAndReference()
        let short = "Hello there."

        _ = try await package.run(Self.clone(reference, key: "asset-A", text: short))
        XCTAssertEqual(package.voiceCache.keys, ["asset-A"])
        XCTAssertEqual(package.promptFeatBuilds, 1)

        // Same key: reused, by either cloning mode.
        _ = try await package.run(Self.clone(reference, key: "asset-A", text: "Another line."))
        _ = try await package.run(Self.clone(reference, transcript: Self.referenceText, key: "asset-A", text: short))
        XCTAssertEqual(package.promptFeatBuilds, 1, "the keyed prompt feature was rebuilt")

        // The same bytes under another key, or under none, do not hit asset-A.
        _ = try await package.run(Self.clone(reference, key: "asset-B", text: short))
        XCTAssertEqual(package.promptFeatBuilds, 2)
        _ = try await package.run(Self.clone(reference, text: short))
        XCTAssertEqual(package.promptFeatBuilds, 3)
        XCTAssertEqual(Set(package.voiceCache.keys), ["asset-A", "asset-B"])

        // Eviction: an empty-text request returns empty audio and drops the key.
        let evicted = try Self.audioOf(await package.run(Self.evict(["asset-A"])))
        XCTAssertEqual(evicted.data.count, 44)
        XCTAssertEqual(package.voiceCache.keys, ["asset-B"])
        _ = try await package.run(Self.clone(reference, key: "asset-A", text: short))
        XCTAssertEqual(package.promptFeatBuilds, 4, "an evicted key was served from the cache")

        // A key reused with other bytes is rebuilt, never served the old voice.
        let other = try Self.audioOf(await package.run(TTSRequest(
            text: "A second voice for the same key.", metaData: ["seed": .int(99)])))
        _ = try await package.run(Self.clone(other, key: "asset-A", text: short))
        XCTAssertEqual(package.promptFeatBuilds, 5)

        // At most voiceCacheCapacity keys; unload drops them all.
        for i in 0..<VoxCPM2TTSPackage.voiceCacheCapacity {
            _ = try await package.run(Self.clone(reference, key: "fill-\(i)", text: short))
        }
        XCTAssertEqual(package.voiceCache.keys.count, VoxCPM2TTSPackage.voiceCacheCapacity)
        XCTAssertFalse(package.voiceCache.keys.contains("asset-B"))
        await package.unload()
        XCTAssertTrue(package.voiceCache.isEmpty)
    }

    // MARK: - The seed reproduces output byte for byte

    @InferenceActor
    func testSameSeedReproducesOutputByteForByte() async throws {
        let (package, reference) = try await Self.loadedPackageAndReference()

        // Zero-shot.
        let z1 = try Self.audioOf(await package.run(TTSRequest(text: Self.targetText, metaData: ["seed": .int(1234)])))
        let z2 = try Self.audioOf(await package.run(TTSRequest(text: Self.targetText, metaData: ["seed": .int(1234)])))
        XCTAssertEqual(z1.data, z2.data, "zero-shot not reproducible from its seed")
        let zOther = try Self.audioOf(await package.run(TTSRequest(text: Self.targetText, metaData: ["seed": .int(4321)])))
        XCTAssertNotEqual(z1.data, zOther.data, "the seed does not reach the zero-shot sampler")

        // Reference-only clone, unkeyed and keyed: the cache must not change the output.
        let r1 = try Self.audioOf(await package.run(Self.clone(reference, seed: 1234)))
        let r2 = try Self.audioOf(await package.run(Self.clone(reference, seed: 1234)))
        let r3 = try Self.audioOf(await package.run(Self.clone(reference, key: "k", seed: 1234)))
        let r4 = try Self.audioOf(await package.run(Self.clone(reference, key: "k", seed: 1234)))
        XCTAssertEqual(r1.data, r2.data, "reference-only clone not reproducible from its seed")
        XCTAssertEqual(r1.data, r3.data, "a keyed clone differs from the unkeyed one")
        XCTAssertEqual(r1.data, r4.data, "a cached clone differs from a fresh one")
        let rOther = try Self.audioOf(await package.run(Self.clone(reference, seed: 4321)))
        XCTAssertNotEqual(r1.data, rOther.data, "the seed does not reach the clone sampler")

        // Continuation clone, from the cache and fresh.
        let c1 = try Self.audioOf(await package.run(Self.clone(reference, transcript: Self.referenceText, seed: 1234)))
        let c2 = try Self.audioOf(await package.run(Self.clone(reference, transcript: Self.referenceText, key: "k", seed: 1234)))
        XCTAssertEqual(c1.data, c2.data, "continuation clone not reproducible from its seed")

        print("[VX2-CLONE] zero-shot \(z1.data.count) B, reference-only \(r1.data.count) B, "
            + "continuation \(c1.data.count) B — byte-identical per seed, keyed or not")
        await package.unload()
    }

    // MARK: - Through MLXServeEngine

    func testEvictionPassesThroughTheEngine() async throws {
        let engine = MLXServeEngine()
        await engine.useModelStore(ModelStore(root: Self.modelsRoot))
        let id = try await engine.register(VoxCPM2TTSPackage.registration, configuration: VoxCPM2Configuration(),
                                           id: PackageID("voxcpm2-clone-test"))
        let needsDownload = await engine.needsDownload(.tts, package: id)
        XCTAssertFalse(needsDownload, "the engine would download into \(Self.modelsRoot.path)")
        guard !needsDownload else { return }
        let reference = try Self.audioOf(await engine.run(
            TTSRequest(text: Self.referenceText, metaData: ["seed": .int(7)]), package: id))
        let cloned = try Self.audioOf(await engine.run(
            Self.clone(reference, key: "asset-E", seed: 5, text: "Hello there."), package: id))
        XCTAssertGreaterThan(cloned.data.count, 44 + 48_000)
        let evicted = try Self.audioOf(await engine.run(Self.evict(["asset-E"]), package: id))
        XCTAssertEqual(evicted.data.count, 44)
        await engine.evict(package: id)
    }
}
