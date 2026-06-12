import XCTest
import Foundation
import MLXToolKit
@testable import MLXVoxCPM2TTS

/// Live validation of the APP-VALIDATION VoxCPM2 checklist (gated: VX2_LIVE=1, needs the
/// Cmlx metallib bundle in .build/debug + weights at DEV_ARCHIVE/models). One load, all paths:
/// zero-shot (BOS-fix onset), reference-only cloning, continuation cloning, stereo/44.1k
/// resample, unload/reload. WAVs land on ~/Desktop for the ear + ECAPA timbre checks.
final class VoxCPM2LiveValidation: XCTestCase {
    static let transcript =
        "The morning light spilled across the quiet harbor as the boats began to stir."
    static let continuation = " And the gulls wheeled overhead in the brightening sky."

    static let transcriptMarker = ()

    func refAudio(_ path: String) throws -> Audio {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return Audio(format: .wav, data: data)
    }

    func stats(_ resp: TTSResponse, _ label: String) throws -> [Float] {
        let (samples, rate) = try VoxCPM2TTSPackage.decodeToMono(resp.audio)
        let peak = samples.map { abs($0) }.max() ?? 0
        let rms = (samples.reduce(Float(0)) { $0 + $1 * $1 } / Float(max(samples.count, 1)))
            .squareRoot()
        // Onset probe: first 50 ms vs the 200–400 ms span (a BOS-shift click shows as an
        // energy spike or dead-air anomaly right at the head).
        let w = rate / 20
        let head = Array(samples.prefix(w))
        let body = Array(samples.dropFirst(rate / 5).prefix(rate / 5))
        let headRms = (head.reduce(Float(0)) { $0 + $1 * $1 } / Float(max(head.count, 1))).squareRoot()
        let bodyRms = (body.reduce(Float(0)) { $0 + $1 * $1 } / Float(max(body.count, 1))).squareRoot()
        print("[VX2] \(label): \(String(format: "%.2f", Double(samples.count) / Double(rate)))s "
            + "@ \(rate) Hz · peak \(String(format: "%.3f", peak)) rms \(String(format: "%.3f", rms)) "
            + "· onset 50ms-rms \(String(format: "%.4f", headRms)) vs body \(String(format: "%.4f", bodyRms))")
        XCTAssertTrue(peak > 0.02 && peak <= 1.0, "\(label): degenerate level")
        XCTAssertTrue(rms > 0.005 && rms < 0.4, "\(label): rms out of range")
        XCTAssertGreaterThan(samples.count, rate, "\(label): under 1s — implausible")
        try resp.audio.data.write(
            to: URL(fileURLWithPath: NSHomeDirectory() + "/Desktop/VX2-\(label).wav"))
        return samples
    }

    func testLiveChecklist() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VX2_LIVE"] == "1", "VX2_LIVE=1")

        let pkg = VoxCPM2TTSPackage(configuration: VoxCPM2Configuration(
            modelsRootDirectory: URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/models")))

        var t0 = Date()
        try await pkg.load()
        print("[VX2] load: \(String(format: "%.1f", -t0.timeIntervalSinceNow))s")

        // 1. Zero-shot (BOS fix — onset should be clean).
        t0 = Date()
        let zs = try await pkg.run(TTSRequest(text: Self.transcript)) as! TTSResponse
        print("[VX2] zero-shot gen: \(String(format: "%.1f", -t0.timeIntervalSinceNow))s")
        _ = try stats(zs, "zeroshot")

        // 2. Reference-only timbre cloning (no transcript), samantha.
        let samantha = try refAudio("/tmp/ref_voice_samantha.wav")
        t0 = Date()
        let ref = try await pkg.run(TTSRequest(
            text: "Completely unrelated words about a winter mountain expedition.",
            voice: VoiceSelector(.referenceAudio(samantha)))) as! TTSResponse
        print("[VX2] ref-only gen: \(String(format: "%.1f", -t0.timeIntervalSinceNow))s")
        _ = try stats(ref, "refclone-samantha")

        // 3. Continuation / Ultimate cloning (transcript + continuation text).
        t0 = Date()
        let cont = try await pkg.run(TTSRequest(
            text: Self.continuation,
            voice: VoiceSelector(.referenceAudio(samantha)),
            referenceTranscript: Self.transcript)) as! TTSResponse
        print("[VX2] continuation gen: \(String(format: "%.1f", -t0.timeIntervalSinceNow))s")
        let contSamples = try stats(cont, "continuation-samantha")
        // Output must be ONLY the continuation (~3-5s), not reference+continuation (~8s+).
        let contSeconds = Double(contSamples.count) / Double(cont.audio.sampleRate ?? 48000)
        XCTAssertLessThan(contSeconds, 7.0, "continuation contains re-synthesized reference?")

        // 4. Stereo 44.1 kHz reference (mixdown + resample path).
        let stereo = try refAudio("/tmp/ref_samantha_stereo441.wav")
        let st = try await pkg.run(TTSRequest(
            text: "Testing the stereo and resample ingest path.",
            voice: VoiceSelector(.referenceAudio(stereo)))) as! TTSResponse
        _ = try stats(st, "stereo441-ref")

        // 5. Unload → reload → run.
        await pkg.unload()
        try await pkg.load()
        let again = try await pkg.run(TTSRequest(text: "Reload check.")) as! TTSResponse
        _ = try stats(again, "after-reload")
        print("[VX2] checklist complete")
    }
}
