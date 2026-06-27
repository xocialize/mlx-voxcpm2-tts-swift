# mlx-voxcpm2-tts-swift

An [MLXEngine](https://github.com/xocialize/mlx-engine-swift) model package exposing the **`tts`**
capability over **VoxCPM2** — a flow-matching text-to-speech model (TSLM → FSQ → RALM → LocDiT →
AudioVAE) producing 48 kHz speech on Apple silicon.

It's a thin conformance wrapper: all model logic lives in the standalone engine
[`mlx-voxcpm-swift`](https://github.com/xocialize/mlx-voxcpm-swift) (product `VoxCPM`), which this
package pins to a tagged release — mirroring how `mlx-qwen-llm-swift` wraps mlx-swift-lm and
`mlx-kokoro-tts-swift` wraps kokoro-mlx-swift. This package adds the `ModelPackage` conformance: the
`PackageManifest`, the HF download, a **weight-parity gate**, and the canonical `TTSRequest` →
`Audio` (.wav) mapping. The `MLXServeEngine` coordinator handles licensing, device eligibility, and
memory budgeting.

## Weights & footprint

Loads [`mlx-community/VoxCPM2-bf16`](https://huggingface.co/mlx-community/VoxCPM2-bf16) (Apache-2.0,
~4.96 GB). The engine casts weights to float32 (Metal precision), so expect **~11 GB resident** — the
manifest declares a `.pro` chip floor and Metal-GPU requirement, and the `MemoryGovernor` gates
admission on the footprint.

The loader validates **weight-key parity** on load: a checkpoint that doesn't match the ported
architecture leaves model parameters unfilled, and rather than emit garbage audio the package throws
`VoxCPM2Error.incompatibleWeights`.

## Usage

```swift
import MLXServeCore
import MLXVoxCPM2TTS

let engine = MLXServeEngine()
try await engine.register(VoxCPM2TTSPackage.registration, configuration: VoxCPM2Configuration())
try await engine.prepare(.tts)
let response = try await engine.run(TTSRequest(text: "Hello from VoxCPM2 on MLXEngine."))
```

## Status

**Zero-shot and reference-audio cloning are supported.** `VoiceSelector.auto` (and `.named`, which
VoxCPM2 has no catalog for) synthesize zero-shot. `.referenceAudio(clip)` clones the clip's timbre;
adding `referenceTranscript` switches to continuation ("Ultimate Cloning") for the highest-fidelity
result. Repeated references are encoded once and reused (prompt-feat cache), so long-form cloning
doesn't re-pay the AudioVAE encode per chunk. Text-driven voice design is not yet exposed.

## Development

Co-developed in the MLXEngine workspace: depends on the engine contract (`MLXToolKit`) via a
tagged-URL net dependency (`.package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.3.0")`)
and on the VoxCPM core via a tagged GitHub release. The Xcode workspace overrides the tagged
core with the in-workspace copy for local development.

## License

MIT — this wrapper. VoxCPM2 weights are Apache-2.0 (their publisher); the core port is MIT. Review
the model card before redistribution.
