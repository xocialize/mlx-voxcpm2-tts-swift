// CancellationTests.swift — VoxCPM2 through the engine's CAN gate (offline, no MLX kernels).
// CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires before notLoaded
// validation or weights); CAN-3 is the document of record for the checkpoint cadence: the core's
// autoregressive patch loop (VoxCPMModel._generateOnce, mlx-voxcpm-swift ≥ 0.3.0) bails once per
// generated latent patch via `Task.isCancelled` — the TSLM → FSQ → RALM → LocDiT interleave —
// and the wrapper's post-synthesis `try Task.checkCancellation()` rethrows the CancellationError
// unchanged (the core's generate() is non-throwing by design; no signature churn).

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXVoxCPM2TTS

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = VoxCPM2TTSPackage(configuration: VoxCPM2Configuration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: TTSRequest(text: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // tts is a long-run capability (and the 4 GB declared transient independently implies
        // long runs) — the sub-second exemption is not available.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: VoxCPM2TTSPackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: VoxCPM2TTSPackage.manifest,
            posture: .cadence([
                // The autoregressive loop checks Task.isCancelled once per generated latent
                // patch (each patch = one LocDiT flow-matching sample of `inferenceTimesteps`
                // ODE steps) — mlx-voxcpm-swift VoxCPMModel._generateOnce, after the patch
                // append. The wrapper checkpoints again post-synthesis (VoxCPM2TTSPackage.run)
                // before the final AudioVAE decode result is pulled + WAV-encoded. Unit: a
                // latent patch is the fixed-duration AR audio unit — `frame`, matching the
                // qwen3-tts codec-frame precedent.
                .init(phase: .generate, unit: .frame),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
