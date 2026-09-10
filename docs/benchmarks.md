# Benchmark scenes (Phase 0 checklist, `radical/opt`)

All on **RTX 3070**, driver 550.x. Record wall-clock + `Profile render` counters +
`clang -fsyntax-only` + diagnostic harness PASS per change. Do NOT compare against
the stale 16m45s baseline without noting the sun state (hamhut json currently has
the sun below the horizon, so the sun gate is active and sun rays = 0).

## A. hamhut — traversal baseline

- Scene: `~/.chunky/scenes/hamhut/hamhut.json`, 3840x2160, target 2000 spp.
- Fast A/B variant: 1920x1080 x 200 spp (same seed) for iteration; promote to 4K
  only for phase gates.
- What it stresses: octree march (1.43 steps/ray), BVH node tests (1.48/ray),
  cloud DDA (0.84 steps/ray). Few emitters, essentially no visible water.
- Gates active: sun gate (sun down), water-gate (skips water octree march +
  water medium lookup via `atmosphere.hasWater`), emitter-intensity gate.

## B. water-heavy — Simplex vs still

- Duplicate of A (or a coastal scene) with visible water + water plane on.
- Two runs per change: water-shader selector = Simplex vs Still.
- What it stresses: `WaterModel_intersect` (12x triangle tests), 4x Simplex noise
  per water-air hit (`noise.h:148-166`), underwater `exp(-d/visibility)`.
- Watch: profile counters `WAVE_NOISE`, `WATER_PLANE`, octree steps delta.

## C. emitter-heavy — ONE strategy

- Night/interior scene with many emitters, emitter intensity ~22x, strategy ONE.
- What it stresses: emitter-grid triple indirection
  (`meta->cells->indexes->emitters`), `sampleEmitterFace` early-outs, the
  `while(traveled<distance)` occluder loop (full `closestIntersect` per step),
  ~5.6G emitter shadow rays in the reference profile.
- Watch: `EMITTER_*`, `OCCLUDER_FAST`, `SUN_*` counters; beauty must be
  bit-identical (seeded) unless a parity break was explicitly signed off.

## Run procedure per change

1. `clang -x cl -cl-std=CL1.2 -fsyntax-only` on the assembled TU (see
   `OPENCODE_CONTEXT.md` section 3 for the exact command).
2. `JAVA_HOME=... ./gradlew clean jar`, copy to `~/.chunky/plugins/`, fix
   `chunky.json` plugin name.
3. Fast variant first (1080p x 200); if delta > 2%, promote to full scene.
4. Save: wall-clock, spp, resolution, sun altitude, water shader, emitter
   strategy, full profile-counter dump, `ncu` summary line (once wizard is used).

## Beauty-MSE protocol (M5)

Deterministic re-renders make PNG comparison a real gate: the host RNG is
`new Random(0)` per render, so same code + same device + same start state =
same noise. Any pixel diff after a slice means changed math, not luck.

Per-slice procedure (windowsill view, 1080p x 200):

1. Profile OFF, OIDN off, no `-DchunkyClWorkGroupSize`, reset scene (spp 0).
2. Render to target, save snapshot PNG as
   `~/.chunky/scenes/hamhut/snapshots/hamhut-200-<slice>.png`
   (keep the pre-Phase-2 file as `hamhut-200-baseline.png` — never overwrite it).
3. Compare:
   `python3 -c "from PIL import Image; import numpy as np, sys; a=np.asarray(Image.open(sys.argv[1]).convert('RGB'),dtype=np.int16); b=np.asarray(Image.open(sys.argv[2]).convert('RGB'),dtype=np.int16); d=np.abs(a-b); print('MSE=%.4f max=%d diffpx=%d/%d' % ((d*d).mean(), d.max(), (d.sum(axis=2)>0).sum(), d.shape[0]*d.shape[1]))" hamhut-200-baseline.png hamhut-200-<slice>.png`
4. Thresholds: **MSE = 0** required for host/JIT/bitmask slices. `native_*`
   only: tiny epsilon allowed, then eyeball the two PNGs before accepting.

Caveats: the Profile checkbox changes RNG streams (batch 1 vs 32), so it must
match the baseline run's setting; cross-device comparison uses MSE threshold,
never exactness.

## M5 floor gate (measured 2026-09-09)

Two identically-configured reset 0 -> 200 Profile-OFF renders (same code, same
seeds, 8 min apart) differ in **~270 of 2,073,600 px (0.013%), max 16/255,
R-heavy, scattered, no connected regions**. RNG streams are provably identical
(host `Random(0)` used only for per-dispatch seeds), so this is rare-path
nondeterminism below the noise floor (VRAM-state/driver dust in long-tail
firefly paths), not a math difference.
**Gate: a slice passes M5 iff its PNG shows diffs at or below this floor
(<= ~300 px, max <= 16, scattered). Dense or structured diffs = investigate.**
Integer counters (M2) remain the exactness proof; pixels prove "no structural
change".

## RR-threshold sweep protocol (Phase 2 slice 0, zero code)

Russian Roulette (kill dim paths after 3 bounces, survivors boosted — unbiased,
deterministic) currently defaults to **50%** (`ChunkyClTab`). Sweep
0 / 20 / 50 / 80 at 1080p x 200, Profile OFF, OIDN off: record wall-clock +
eyeball noise at 1:1 crops (lantern + foliage + glass). Higher = faster +
noisier; pick the knee for the windowsill view, then confirm the water and
emitter scenes don't disagree. No code changes — tab slider only.

## Locked baselines (RTX 3070, driver 550.x)

- 2026-09-09, `radical/opt` pre-Phase-1 (per-frame profile accumulation + gates):
  hamhut 1080p windowsill view (emissive cherry leaves/lanterns/petals, glass
  behind camera) x 200 spp, profile ON: **22 s** (18.9M rays/s). Ratios:
/ray: 45.52 octree steps, 30.41 BVH tests, 2.60 hits, 2.51 diffuse, 0.08
  refraction, 2.28 emitter lookups, 1.14 emitter rays, 0 sun rays, 3.25 cloud
  steps, ~0 water work; descent depth 6.91 levels/step. `ncu`/`smi`: sm 98-99%,
  mem 5-8% (compute-bound, NOT host-starved).
- Phase 1 slice 1 (spp-batching, K=32, profiler off): A/B protocol is
  profile-OFF wall-clock for speed (batching active) + profile-ON counter dump
  for correctness (batching forced to 1, ratios must reproduce the 22 s run
  exactly; noise pattern will differ — per-batch reseeding, statistically
  identical convergence).
- 2026-09-09 slice-1 result: profile-OFF x 200 = **20 s** (vs 22 s profile-ON
  baseline; ~3-4% of the gap is profiler atomics, true batching gain ~5-6%).
  Profile-ON x 200 = 23 s with ratios reproduced exactly (45.52/30.41/2.51/
  2.60/6.91, invariants hold). 200 dispatches -> 7 launches (6x32+8).
- Phase 1 slice 2 (async per-dispatch writes + `-DchunkyClWorkGroupSize=N`
  knob, default = driver choice): A/B is profile-OFF x 200 wall-clock only.
  Expect ~0-1% (at 7 launches the drains are already negligible); the knob
  enables a future 64/128/256 sweep in single runs if Phase 2 needs it.
- Stage 1 JIT (specialized program per scene feature-set, cached; first new
  set pays one recompile, watch the log for `compiling OpenCL kernel [...]`):
  A/B is M1 profile-OFF x 200 wall (median of 3 after warmup) + M2 profile-ON
  ratios (must reproduce exactly — JIT is behavior-identical by construction)
  + M5 floor gate (<= ~300 px, max <= 16, scattered). Hamhut combo is
  `-DFOG_MODE=<n> -DHAS_CLOUDS=1 -DHAS_EMITTERS=1` (no water/plane/sun —
  verify against the `compiling` log line).
- Stage 1b UX + persistence: the OpenCL tab shows `Compiling OpenCL
  kernel…` during a blocking build (0 SPP then is expected, not frozen).
  Linked binaries persist under `<chunky.home>/kernel-cache/` (portable —
  never hardcoded) keyed by (defines, device, driver, plugin build); later
  sessions log `reused cached kernel binary` and skip the ~1 min compile. Any
  cache failure falls back to source compile. Verify: this session compiles
  once per define-set, next session reuses.
- Stage 1c (one program/session + lap timer): preview uses the same
  specialized program (no separate default build at startup anymore), the tab
  shows `Render Time: X s (compile Y s)` plus a persistent `Kernel: ...` line
  (`compiled in Ns` vs `cache, ready in Ns`). Stale `kernel-*.bin` files from
  older jars are ignored automatically (key mismatch), safe to delete.
- hamhut-day (sun-up validator, altitude ~95): combo
  `-DFOG_MODE=1 -DHAS_WATER=1 -DHAS_CLOUDS=1 -DHAS_EMITTERS=1 -DHAS_SUN=1`,
  1080p x 200 = **28 s** median (27/28/28). Ratios/ray: 56.11 octree steps,
  43.65 BVH tests, 2.60 hits, 2.50 diffuse, 2.27 emitter lookups, 1.14 emitter
  rays, **1.33 sun rays**, 0.77 occluder fast-path, 5.92 cloud steps, ~0 water;
  descent 6.77 levels/step (380.00/ray). Day M5 gate = **exact-0**
  (baseline pair bit-identical); day validation runs stay Profile ON (pinned).
  Cold compile 87 s (fattest combo), cached after.
- Stage 2 native_* (slices A+B, both scenes gate): A = fog/water
  `native_exp` (8-bit-output sites) + `Sun_emittance` per-launch hoist +
  `cos(0.03)` literal. B = diffuse hemisphere natives + specular/sun-direction
  fast-normalize. HELD (never): refract sqrt (TIR), emitter length/1-d2/powr,
  geometry normalize/area, sky acos/atan2. A/B per slice: night
  M1/M2/M5-floor (OFF-pinned) + day M1/M2/M5-exact (ON-pinned).
- Stage 2 regression + bisect (2026-09-09): day 28 s -> 32 s (+14%) with
  IDENTICAL counters — night flat at 19 s, exonerating diffuse/specular
  natives. Suspect pool = sun-only changes (hoist live-range vs sun natives).
  Reverting the `sunEmit` hoist alone restored 28 s with pixel-identical output
  (test2-vs-baseline stats digit-identical to test1) and M2 exact twice:
  a `float3` parked live across the megakernel loop cost more occupancy than
  one `pow` per sun bounce saved. Lesson: on register-bound kernels, live
  ranges are the budget, not ALU. M5 gate redesigned: exact/floor pixel gates
  apply to structural slices only — precision slices validate via M2 exact +
  |Dmean| < 1% + rms sanity + eyeball (bounce chaos amplifies ~1e-7
  perturbations into globally different noise with identical statistics).
- Workgroup triad (day, OFF): 64 -> 26 s (-7%), 128 -> 27 s, 256 = 28 s =
  driver default. Mild occupancy sensitivity: somewhat register-bound, not
  starved. Wavefront-lite stays parked; work-reduction outranks occupancy
  games. Knob published in README, not hardcoded (device-specific).
- Bitmask ascent: KILLED before implementation (2026-09-09). Bytecode audit of
  `PackedOctree.set/mergeNode` proves fully-empty internal nodes cannot exist
  (8-equal children, including all-air, always collapse into a leaf) — so an
  "ascend to largest empty ancestor" check can never fire; big voids already
  exit coarse boxes, and 6.91-deep descents happen exactly where ancestors are
  occupied. A sidecar would have cost VRAM + args + registers for zero skips.
  The remaining march tax is the per-step root re-walk (~7 loads × 45 steps);
  fixing that needs persistent restart/rope state — a project, not a slice.
  Recorded so nobody re-proposes it.
- Shadow lightweight sampling (2026-09-09): walls FLAT both scenes (19/28 s),
  M2 exact twice, day PNG digit-identical to pre-cut (180935/87568/77922).
  The eliminated SMR/emittance reads hide behind traversal-divergence latency;
  shadow cost is pure march. Kept as zero-cost insurance for
  translucent/entity-heavy scenes (hamhut is fast-path-saturated). Lesson:
  on latency-bound kernels, removing ALU/reads that overlap stalls measures
  zero — profile the critical path, not the instruction count.
- JIT kill-switch: `-DchunkyClJit=off` forces the single everything-on program
  (A/B "was JIT worth it", paranoia runs). Driver-cache note: NVIDIA's
  `ComputeCache` (default 256 MB cap) evicts under pressure — measured 21 s
  backend with a full cache vs 0 s after `CUDA_CACHE_MAXSIZE=1073741824`
  (RTX 3070 / 550.x). See README.
- Cloud gate (2026-09-10): night 19 -> 18 s, day 28 -> 27 s, day-ON 32 ->
  31 s; M2 exact to ghost-floor; day PNG statistically clean (uniform
  sub-8-LSB scatter, zero sky concentration). Small and real, as predicted —
  emitter-shadow segments skipping call overhead.
- M5 methodology correction: same-jar pairs may demand exact-0 (day) or floor
  (night ghost); CROSS-JAR pairs are inherently statistical (SASS/FMA ulp
  noise + rare edge chaos) — validate those via M2 exact + |Dmean| + rms +
  structure, never pixel equality.
- Host batch 1: stash guides only while the buffer converges (bufferSppReal
  mirrors GUIDE_SPP_CAP) or when no stash exists; output-only GPU buffers
  zero via device fill (defeats NaN-garbage poisoning of the spp=0 first
  average); biome buffers gain an emitter-grid-style identity guard.
  modCount non-commit documented as load-bearing (super's non-store is the
  subclass dirty signal). Walls unchanged by design (load/reset path);
  M2 exact.
- Host batch 2: preview persists kernel + pixel buffer (+ panoramic camera
  only — pinhole snapshots position at construction, persisting it froze the
  view; caught on self-review) with rebuild-on-change and finally-released
  events; sky HDR rebake debounced to 1/500 ms (finals always follow a
  pause). Deferred with reason: overlapped merge readback (~0.5% steady
  state; saveEvent path is Chunky snapshot semantics).
