import XCTest
import Foundation
import MLX
import MLXToolKit
@testable import MLXVoxCPM2TTS

/// Split-footprint memory bench for the 1.14 efficiency contract (gated: VX2_MEM=1, needs the
/// Cmlx metallib bundle in .build/debug + weights at DEV_ARCHIVE/models).
///
/// Measures the two halves the manifest must declare (see memory-harness.md):
///   1. Resident floor  — load + one warmup synth + clearCache → activeMemory ≈ weights resident.
///   2. Transient peak  — reset Memory.peakMemory, run a real synth with forced eval, read peakMemory;
///                        peakActivationBytes = worstPeak − resident floor.
/// VoxCPM2 has two transient sources — the autoregressive TSLM and the diffusion (LocDiT) decoder —
/// so the peak is the max across them within one generate() call; we measure the whole call.
/// Guards the silent-output failure class (a silent stem reads −∞ dBFS) before trusting any number.
final class VoxCPM2MemoryReport: XCTestCase {
    static let sentence =
        "The morning light spilled across the quiet harbor as the boats began to stir, "
        + "and the gulls wheeled overhead in the brightening sky above the waking town."

    func testMeasure_voxcpm2_bf16() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["VX2_MEM"] == "1", "VX2_MEM=1")

        let pkg = VoxCPM2TTSPackage(configuration: VoxCPM2Configuration(
            modelsRootDirectory: URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/models")))

        // --- load + warmup (compiles size-specific kernels, materializes weights) ---
        var t0 = Date()
        try await pkg.load()
        let loadSecs = -t0.timeIntervalSinceNow
        _ = try await pkg.run(TTSRequest(text: "Warm up.")) as! TTSResponse

        // --- resident floor: drop activations back to cache, then read active ---
        MLX.Memory.clearCache()
        let residentFloor = MLX.Memory.activeMemory

        // --- transient peak: rebase peak to current active (weights), run a real synth ---
        MLX.Memory.clearCache()
        MLX.Memory.peakMemory = 0
        t0 = Date()
        let resp = try await pkg.run(TTSRequest(text: Self.sentence)) as! TTSResponse
        let genSecs = -t0.timeIntervalSinceNow
        let worstPeak = MLX.Memory.peakMemory

        // --- silent-output guard: a valid non-silent 48 kHz WAV, not a dead stem ---
        let (samples, rate) = try VoxCPM2TTSPackage.decodeToMono(resp.audio)
        let peakAmp = samples.map { abs($0) }.max() ?? 0
        let rms = (samples.reduce(Float(0)) { $0 + $1 * $1 } / Float(max(samples.count, 1))).squareRoot()
        XCTAssertTrue(peakAmp > 0.02 && peakAmp <= 1.0, "degenerate/silent level (peak \(peakAmp))")
        XCTAssertTrue(rms > 0.005 && rms < 0.4, "rms out of range (\(rms))")
        XCTAssertGreaterThan(samples.count, rate, "under 1s — implausible")

        let transient = max(0, worstPeak - residentFloor)
        let durSecs = Double(samples.count) / Double(rate)
        let gib = 1024.0 * 1024.0 * 1024.0
        print(String(
            format: """
            [VX2-MEM] envelope: zero-shot, %d chars → %.1fs @ %d Hz, %d ODE steps
            [VX2-MEM] load: %.1fs · gen: %.1fs
            [VX2-MEM] resident floor: %.2f GB (%ld bytes)
            [VX2-MEM] worst peak:     %.2f GB (%ld bytes)
            [VX2-MEM] transient (peak − floor): %.2f GB (%ld bytes)
            [VX2-MEM] audio: peak %.3f rms %.3f
            """,
            Self.sentence.count, durSecs, rate, 10,
            loadSecs, genSecs,
            Double(residentFloor) / gib, residentFloor,
            Double(worstPeak) / gib, worstPeak,
            Double(transient) / gib, transient,
            peakAmp, rms))
    }
}
