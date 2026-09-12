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
        // Split footprint (1.14 efficiency contract): measured weights floor + measured transient.
        #expect(r.footprints.first?.residentBytes == 9_300_000_000)
        #expect(r.footprints.first?.peakActivationBytes == 4_000_000_000)
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

    @Test func referenceAudioDecodesMonoRoundTrip() throws {
        // encode → canonical Audio → decodeToMono recovers samples + rate (the .referenceAudio
        // ingest path; AVFoundation decode, so any valid WAV layout works).
        let samples: [Float] = (0..<2400).map { sin(Float($0) * 0.05) * 0.7 }
        let wav = VoxCPM2TTSPackage.encodeWAV16(samples: samples, sampleRate: 24_000)
        let audio = Audio(format: .wav, data: wav, sampleRate: 24_000, channels: 1)

        let (decoded, rate) = try VoxCPM2TTSPackage.decodeToMono(audio)
        #expect(rate == 24_000)
        #expect(decoded.count == samples.count)
        for (a, b) in zip(decoded, samples) {
            #expect(abs(a - b) < 0.001) // 16-bit quantization tolerance
        }
    }

    @Test func unreadableReferenceAudioThrowsTypedError() {
        let junk = Audio(format: .wav, data: Data([0x00, 0x01, 0x02]))
        #expect(throws: VoxCPM2Error.self) {
            _ = try VoxCPM2TTSPackage.decodeToMono(junk)
        }
    }
}

// MARK: - E12 typed plane (contract 1.38.0, AB-A-0049 part 3)

@Suite struct E12ControlsTests {
    @Test func declaresInlineDescriptionSteeringOnly() {
        let surface = VoxCPM2TTSPackage.manifest.surfaces[0]
        #expect(surface.ttsControls?.emotionModes == [.categorical, .textDescription])
        #expect(surface.ttsControls?.supportsTargetDuration == false)
        #expect(surface.parameters.contains { $0.name == "emotion" })
        #expect(!surface.parameters.contains { $0.name == "targetDuration" })
        #expect(surface.controlsMatchCapability)
    }

    @Test func typedEmotionBecomesTheInlineDescriptionPrefix() throws {
        #expect(try VoxCPM2TTSPackage.controlledText("Hello.", emotion: .textDescription("in a whisper"))
                == "(in a whisper)Hello.")
        #expect(try VoxCPM2TTSPackage.controlledText("Hello.", emotion: .categorical("happy"))
                == "(cheerful and upbeat)Hello.")
        #expect(try VoxCPM2TTSPackage.controlledText("Hello.", emotion: .categorical("Neutral"))
                == "(calm and even)Hello.")
        #expect(try VoxCPM2TTSPackage.controlledText("Hello.", emotion: nil) == "Hello.")
        #expect(E12Emotion.allCases.map(\.rawValue)
                == ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"])
    }

    @Test func undeclaredModesAndUnknownLabelsAreRefusedLegibly() {
        #expect(throws: PackageError.self) {
            try VoxCPM2TTSPackage.controlledText("x", emotion: .categorical("ecstatic"))
        }
        #expect(throws: PackageError.self) {
            try VoxCPM2TTSPackage.controlledText("x", emotion: .vector([0.2]))
        }
        #expect(throws: PackageError.self) {
            try VoxCPM2TTSPackage.controlledText("x", emotion: .referenceAudio(Audio(data: Data())))
        }
    }
}
