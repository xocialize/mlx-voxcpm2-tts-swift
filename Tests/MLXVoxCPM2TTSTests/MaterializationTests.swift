// MaterializationTests.swift — VoxCPM2 through the engine's MAT gate (offline, no network):
// the WeightSourcing declaration, fresh-machine honesty, explicit-path satisfaction, and the
// store-layout probe/resolution. VoxCPM2 has a single bf16 runtime, so one declaration covers
// the package (no quant tiers to iterate).

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXVoxCPM2TTS

final class MaterializationTests: XCTestCase {

    /// Temp dir holding probe files that make an explicit-dir config read as satisfied.
    private func satisfiedDir() throws -> (dir: URL, cleanup: () -> Void) {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "voxcpm2-mat-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for f in VoxCPM2Configuration.requiredFiles {
            FileManager.default.createFile(atPath: dir.appending(path: f).path, contents: Data([0]))
        }
        return (dir, { try? FileManager.default.removeItem(at: dir) })
    }

    // MARK: - Engine MAT gate

    func testMATGate() throws {
        let (dir, cleanup) = try satisfiedDir()
        defer { cleanup() }
        let report = MaterializationConformance.check(
            freshConfiguration: VoxCPM2Configuration(),
            satisfiedConfiguration: VoxCPM2Configuration(modelDirectory: dir))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - Source declaration shape

    func testDeclaresSingleMainSource() {
        let sources = VoxCPM2Configuration().weightSources
        XCTAssertEqual(sources.map(\.role), ["main"])
        XCTAssertEqual(sources[0].repo, "mlx-community/VoxCPM2-bf16")
        XCTAssertEqual(sources[0].matching, VoxCPM2Configuration.snapshotGlobs)
    }

    // MARK: - Store-layout probe + resolution

    func testStoreLayoutSatisfiesAndResolves() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "voxcpm2-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = VoxCPM2Configuration()
        // Empty store: the source is missing.
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 1)
        // Populate the expected layout.
        let dir = root.appending(path: cfg.repo)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for f in VoxCPM2Configuration.requiredFiles {
            FileManager.default.createFile(atPath: dir.appending(path: f).path, contents: Data([0]))
        }
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: root).isEmpty)
        // Resolution lands on the store layout; an explicit dir always wins.
        XCTAssertEqual(cfg.resolved(storeRoot: root).modelDirectory?.path, dir.path)
        let explicit = VoxCPM2Configuration(modelDirectory: URL(fileURLWithPath: "/x"))
            .resolved(storeRoot: root)
        XCTAssertEqual(explicit.modelDirectory?.path, "/x")
    }

    func testPrewarmPathsUseResolvedStoreLayout() {
        let root = URL(fileURLWithPath: "/tmp/some-store")
        let cfg = VoxCPM2Configuration(modelsRootDirectory: root)
        XCTAssertEqual(cfg.prewarmPaths.map(\.path),
                       [root.appending(path: "mlx-community/VoxCPM2-bf16").path])
    }

    func testCodableRoundTrip() throws {
        let cfg = VoxCPM2Configuration(inferenceTimesteps: 7,
                                       modelDirectory: URL(fileURLWithPath: "/x"))
        let decoded = try JSONDecoder().decode(VoxCPM2Configuration.self,
                                               from: JSONEncoder().encode(cfg))
        XCTAssertEqual(decoded.repo, cfg.repo)
        XCTAssertEqual(decoded.inferenceTimesteps, 7)
        XCTAssertNil(decoded.modelDirectory)   // environment-specific, never encoded
    }
}
