# Per-frame profile accumulation

The 32-bit GPU profile counters wrapped silently when accumulated only at merge points (a 4K 500-spp window holds ~4G rays, near 2^32), manufacturing impossible invariants. We read and accumulate the counters after every frame — per-frame deltas are tiny so the modular-delta math is always exact.

## Considered Options

- 64-bit GPU counters via `atomic_inc(ulong)`: impossible — clang's CL1.2 header only declares the 64-bit `atom_*` forms under `cl_khr_int64_base_atomics`, which cannot be pragma-enabled after the header is included.
- Merge-point accumulation every 1024 spp: the old design; dropped for the wrap reason above.

## Consequences

Kernel-written counter buffers must be `CL_MEM_READ_WRITE` (a `CL_MEM_READ_ONLY` buffer wedged the render thread), and every `Scene`/`Octree`/`Bvh` constructor must default the profile fields to false/NULL — uninitialised fields caused a preview-kernel GPU fault. A per-work-group `__local` Scene setup was tried and reverted after a reproducible ~6.5% regression: never trust static cost estimates on GPU compilers, always A/B.
