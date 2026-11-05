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
result. Text-driven voice design is not yet exposed.

**Voice cache and seed (v0.6.0, AB-A-0251; the mlx-qwen3-tts-swift 0.7.0 API).** MES Base §13
rules 4 and 6: without `metaData["voiceCacheKey"]` (a string; Gaea passes the MES asset id)
nothing derived from the reference outlives the call. With a key, the encoded prompt feature
(AudioVAE encode + patchify — a function of the clip bytes alone, serving both cloning modes) is
kept under that key, at most `VoxCPM2TTSPackage.voiceCacheCapacity` (4) keys, least recently used
dropped first, so long-form cloning doesn't re-pay the encode per chunk. Identical bytes under
another key, or under none, never hit an entry; a key that comes back with other bytes is
re-encoded. Before 0.6.0 the last reference was reused implicitly, keyed by its bytes; long-form
callers now pass a key. To drop keys, run
`TTSRequest(text: "", metaData: ["evictVoiceCacheKeys": .array([.string(assetId)])])` through
`MLXServeEngine.run`: eviction runs first, before the weights check, and with empty text returns a
44-byte WAV with no samples. `unload()` drops every key.

`metaData["seed"]` (int, clamped to 32 bits) seeds MLX's random state right before generation,
after any reference encode, so a seeded request reproduces byte for byte — zero-shot or cloned,
keyed or not. Live tests: `VX2_CLONE=1 swift test --build-system swiftbuild --filter
VoiceCloneLiveTests` (weights from `VX2_MODELS`, default `/Volumes/DEV_VOL1/Models/Models`; nothing
is downloaded).

## Development

Co-developed in the MLXEngine workspace: depends on the engine contract (`MLXToolKit`) via a
tagged-URL net dependency (`.package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.74.1")`, which brings the
fleet's mlx-swift fork with the leak fix, AB-D-0208)
and on the VoxCPM core via a tagged GitHub release. The Xcode workspace overrides the tagged
core with the in-workspace copy for local development.

## License

MIT — this wrapper. VoxCPM2 weights are Apache-2.0 (their publisher); the core port is MIT. Review
the model card before redistribution.
