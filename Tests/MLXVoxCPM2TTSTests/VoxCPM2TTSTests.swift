import Testing
import Foundation
import MLXToolKit
@testable import MLXVoxCPM2TTS

/// Offline conformance checks — no weights, no Metal. Live synthesis + weight-parity is proven
/// in the test app (the model is ~11 GB and needs the GPU).
struct VoxCPM2TTSTests {

    @Test func manifestIsTTSAndPermissive() {
        let m = VoxCPM2TTSPackage.manifest
        #expect(m.capabilities == [.tts])
        #expect(m.license.weightLicense == .apache2)
        #expect(m.license.portCodeLicense == .mit)
        #expect(m.provenance.sourceRepo == "mlx-community/VoxCPM2-bf16")
    }

    @Test func manifestRequirementsReflectAHeavyModel() {
        let r = VoxCPM2TTSPackage.manifest.requirements
        #expect(r.requiredBackends.contains(.metalGPU))
        #expect(r.chipFloor == .pro)
        #expect(r.os.minMacOS == SemanticVersion(major: 26, minor: 0, patch: 0))
        #expect(r.footprints.first?.quant == .bf16)
        #expect(r.footprints.first?.residentBytes == 11_000_000_000)
    }

    @Test func registrationConstructs() throws {
        let reg = VoxCPM2TTSPackage.registration
        #expect(reg.manifest.capabilities == [.tts])
        let pkg = try reg.makePackage(VoxCPM2Configuration())
        #expect(pkg is VoxCPM2TTSPackage)
    }

    @Test func configurationDefaults() {
        let c = VoxCPM2Configuration()
        #expect(c.repo == "mlx-community/VoxCPM2-bf16")
        #expect(c.inferenceTimesteps == 10)
        #expect(c.cfgValue == 2.0)
        #expect(c.temperature == 1.0)
    }

    @Test func wavHeaderIsValid() {
        let sampleRate = 48000
        let samples = [Float](repeating: 0, count: 100)
        let wav = VoxCPM2TTSPackage.encodeWAV16(samples: samples, sampleRate: sampleRate)
        #expect(wav.count == 44 + samples.count * 2)           // 16-bit mono
        #expect(wav.prefix(4) == Data("RIFF".utf8))
        #expect(wav[8..<12] == Data("WAVE".utf8))
        #expect(wav[36..<40] == Data("data".utf8))
    }
}
