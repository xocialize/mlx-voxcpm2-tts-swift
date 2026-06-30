# Efficiency Adoption Brief — `mlx-voxcpm2-tts-swift` (VoxCPM2, `tts`)

> **For a session-specific agent.** Adopt the engine 1.14 efficiency contract (engine 0.15.0). Load the
> `mlx-swift-integration` skill; read references/package-efficiency.md (four levers + "Measurement
> findings") + references/memory-harness.md. Audited 2026-06-30.

## Package at a glance
- Wrapper `MLXVoxCPM2TTS` (`VoxCPM2TTSPackage`). **Flow-matching TTS:** TSLM (autoregressive) → FSQ →
  RALM → **diffusion** decoder → 48 kHz WAV. Zero-shot + reference cloning. Capability `tts`.
- **Footprint today:** flat `QuantFootprint(.bf16, 11 GB)` — a "heavy autoregressive + diffusion lift"
  capability floor (the package comment already calls it a sanity marker). No split. Single bf16 config.
- Engine pinned `from: "0.3.0"`.

## The shape
Two transient sources: the **autoregressive TSLM** (generation scratch, sequence-scaled — the prefill
lesson) and the **diffusion decoder** (a denoise activation peak). So it's a multi-component pipeline with
a real **per-stage** angle (TSLM/RALM vs the diffusion decoder run at different phases) AND a measured
transient. Both apply.

## Audit vs. the four levers
| Lever | State | Finding | Priority |
|---|---|---|---|
| Engine dep | 🟡 | from 0.3.0 → 0.15.0 | **P0** |
| 1. Split footprint | ❌ | flat 11 GB; autoregressive + diffusion transient unaccounted | **P1** |
| 2. Per-stage evict | 🟡 verify | TSLM/RALM vs diffusion decoder — stage if a component is idle for a phase | P2 (verify) |
| 3. mmap/lazy | 🟡 verify | confirm lazy/mmap load | note |
| 4. BudgetAware | ➖ | single bf16 runtime | defer |

## Plan
- **P0:** `swift package update` → 0.15.0; build + fix any drift.
- **P1:** declare the split — `residentBytes` = weights floor, `peakActivationBytes` = the **measured**
  transient (the max of the TSLM-gen scratch and the diffusion-decode peak) at a documented synth
  envelope. Conform the config to `QuantConfigured` (single bf16 → quant alone suffices; FootprintConfigured
  only if a real second variant exists). **Measure, don't derive** (autoregressive + diffusion — verify the
  peak against a real synth).
- **P2:** verify the component lifecycle — if the TSLM/RALM stage completes before the diffusion decode (or
  vice-versa) and the idle component is multi-GB, stage load→use→evict (Wan-T5 pattern; watch the Swift 6
  `#isolation` gotcha if the staged path is async). If they interleave, P2 is N/A — note it.
- **P3** mmap (note). **P4** defer.
- **Measure** via the package's own smoke/CLI target (`xcodebuild`): load → weights floor → a
  representative synth → peak. Document the synth envelope.

## Definition of done
- [ ] engine 0.15.0; `QuantConfigured`; split declared (`residentBytes` + measured `peakActivationBytes` @ envelope).
- [ ] P2 decided (staged or N/A-with-reason); mmap noted; BudgetAware deferred.
- [ ] Smoke green (valid non-silent 48 kHz WAV — guard the silent-output failure class); split recorded.
- [ ] Registry: mlx-voxcpm2-tts-swift row Eff ⬜→✅, Eng→0.15.0.

## Report back
Flat→split, the measured transient (TSLM vs diffusion, which dominates) + synth envelope, P2 decision,
drift, effort, SHAs. STAY IN SCOPE — four-lever adoption + brief + registry row only; stop-and-report if bigger.
