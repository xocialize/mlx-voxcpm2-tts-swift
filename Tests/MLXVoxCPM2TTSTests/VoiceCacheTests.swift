import Foundation
import Testing
import MLXToolKit
@testable import MLXVoxCPM2TTS

// Offline (no MLX, no weights): the keyed voice cache's LRU, the metaData shapes, and the
// eviction request, which runs without weights (AB-A-0251, the mlx-qwen3-tts-swift 0.7.0 API).
// Reuse under a key, eviction of real entries and the seed are proven live in
// VoiceCloneLiveTests (VX2_CLONE=1).

@Suite struct VoiceFeatureCacheTests {
    @Test func keepsAtMostCapacityDroppingTheLeastRecent() {
        var cache = VoiceFeatureCache<Int>(capacity: 2)
        cache.insert(1, for: "a")
        cache.insert(2, for: "b")
        #expect(cache.value(for: "a") == 1)   // a is now the most recent
        cache.insert(3, for: "c")             // drops b
        #expect(cache.keys == ["a", "c"])
        #expect(cache.value(for: "b") == nil)
    }

    @Test func removeDropsOnlyThatKey() {
        var cache = VoiceFeatureCache<Int>(capacity: 4)
        cache.insert(1, for: "a")
        cache.insert(2, for: "b")
        cache.remove("a")
        cache.remove("never-held")
        #expect(cache.keys == ["b"])
        #expect(cache.value(for: "a") == nil)
        cache.removeAll()
        #expect(cache.isEmpty)
    }

    @Test func reinsertReplacesAndRefreshes() {
        var cache = VoiceFeatureCache<Int>(capacity: 2)
        cache.insert(1, for: "a")
        cache.insert(2, for: "b")
        cache.insert(10, for: "a")
        #expect(cache.keys == ["b", "a"])
        #expect(cache.value(for: "a") == 10)
    }

    @Test func packageCapacityIsAFewKeys() {
        #expect(VoxCPM2TTSPackage.voiceCacheCapacity == 4)
    }
}

@Suite struct VoiceCacheMetaDataTests {
    @Test func evictionKeysAcceptAStringOrAnArray() {
        let one: MetaData = ["evictVoiceCacheKeys": .string("asset-1")]
        #expect(one.stringList("evictVoiceCacheKeys") == ["asset-1"])
        let many: MetaData = ["evictVoiceCacheKeys": .array([.string("a"), .string("b"), .int(3)])]
        #expect(many.stringList("evictVoiceCacheKeys") == ["a", "b"])
        #expect(MetaData().stringList("evictVoiceCacheKeys") == nil)
    }

    @Test func seedReadsAnInt() {
        let meta: MetaData = ["seed": .int(1234), "voiceCacheKey": .string("asset-1")]
        #expect(meta.intValue("seed") == 1234)
        #expect(meta.stringValue("voiceCacheKey") == "asset-1")
        #expect(MetaData().intValue("seed") == nil)
    }

    @Test func fingerprintFollowsTheBytes() {
        let a = VoxCPM2TTSPackage.promptKey(data: Data([1, 2, 3]))
        #expect(a == VoxCPM2TTSPackage.promptKey(data: Data([1, 2, 3])))
        #expect(a != VoxCPM2TTSPackage.promptKey(data: Data([1, 2, 4])))
    }
}

@Suite struct VoiceCacheEvictionTests {
    static func evict(_ keys: [String], text: String = "") -> TTSRequest {
        TTSRequest(text: text, metaData: ["evictVoiceCacheKeys": .array(keys.map { .string($0) })])
    }

    @Test @InferenceActor func evictionAloneNeedsNoWeightsAndReturnsEmptyAudio() async throws {
        let package = VoxCPM2TTSPackage(configuration: VoxCPM2Configuration())
        let raw = try await package.run(Self.evict(["asset-1", "asset-2"]))
        let response = try #require(raw as? TTSResponse)
        #expect(response.audio.format == .wav)
        #expect(response.audio.sampleRate == 48_000)
        #expect(response.audio.data.count == 44)   // header only, zero samples
        #expect(String(data: response.audio.data.prefix(4), encoding: .ascii) == "RIFF")
        let dataSize = response.audio.data.subdata(in: 40..<44).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
        #expect(UInt32(littleEndian: dataSize) == 0)
    }

    @Test @InferenceActor func evictionWithTextStillSynthesizesSoNeedsWeights() async throws {
        let package = VoxCPM2TTSPackage(configuration: VoxCPM2Configuration())
        await #expect(throws: PackageError.self) {
            _ = try await package.run(Self.evict(["a"], text: "Hello."))
        }
    }

    @Test @InferenceActor func ordinaryRequestWithoutWeightsIsStillNotLoaded() async throws {
        let package = VoxCPM2TTSPackage(configuration: VoxCPM2Configuration())
        await #expect(throws: PackageError.self) {
            _ = try await package.run(TTSRequest(text: ""))
        }
    }
}
