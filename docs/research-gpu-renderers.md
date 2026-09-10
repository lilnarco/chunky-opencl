# GPU renderer research brief (Phase 0, `radical/opt`)

Sources are primary: project docs/papers, not blogs. Vocabulary follows `CONTEXT.md`
(octree march, water octree, emitter grid, guides, sun gate, water-gate, layered fog,
profile counters, dynamic offset).

## 1. LuxCoreRender (OpenCL path tracer)

- Engine `Path` / `Tiled Path` (CPU/OpenCL): unidirectional, whole-film progressive.
  OpenCL build auto-tunes tile count to device; last tile splits across devices.
  Source: https://wiki.luxcorerender.org/Render_Configuration
- Samplers: Sobol (default, adaptive), Metropolis (bright-area/caustics, high RAM,
  not recommended on GPU), Random. Variance clamping supported.
- Caches: environment-light cache helps indoor/HDRI-through-window, hurts open scenes.
- v2.4 notes that matter to us: single kernel compilation after install; new material
  eval is equal on complex scenes, slower on simple ones; **sampling-pattern modes:
  progressive (1 spp/px/pass, 16x16 tiles, best preview), cache-friendly (32 spp/px/pass,
  best final), out-of-core (512 spp/px/pass, min host-device swap)**.
  Source: https://luxcorerender.org/new-features-in-v2-4
- **Gap vs ChunkyCL:** we dispatch **1 spp per `clEnqueueNDRangeKernel`**
  (`OpenClPathTracingRenderer.java:102-109`, `PathTraceKernel.java:95-100`) with
  2x blocking 4B writes + `clWaitForEvents` per spp. LuxCore's 32-spp cache-friendly
  pass is the direct precedent for Phase 1 spp-batching (fewer launches, better L2
  reuse, same math).

## 2. Cycles split kernel / `enqueue_inactive` (pre-HIP OpenCL)

- Cycles split the megakernel into stage kernels (direct lighting, `shadow_blocked`,
  etc.) plus a **`kernel_enqueue_inactive` queue**: inactive rays become candidates
  to share indirect-loop work, keeping the GPU busy through divergence.
  Source: bf-blender-cvs `86b8427c852` (device_split_kernel.cpp, kernel_split.cl).
- OpenCL backend was later removed (limited kernel, driver bugs, stalled standard);
  HIP/Metal replaced it. Lesson: split-kernel pays only if the queue management is
  cheap (prefix-sum compaction, not per-thread atomics).
  Source: https://developer.blender.org/docs/release_notes/3.0/cycles

## 3. Wavefront vs megakernel

- Laine et al. 2013, "Megakernels Considered Harmful" (HPG): split path tracer into
  specialized kernels so **lean raycast kernels use few registers** while fat material
  evaluators don't pollute them. Path state in SOA layout in VRAM; queue writes
  coalesced via warp-aggregated atomics. Overheads (state traffic, launches) are
  outweighed by coherence.
  Source: https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf
- 2026 replication (Vulkan, RTX 3060 Ti, glTF scene): **wavefront +16% FPS over
  megakernel** (73.6 vs 64.7 FPS, 13.58 vs 15.47 ms). Nsight: SM 34.1 vs 37.1%,
  RTCore 11.7 vs 16.9%, **VRAM 41.4 vs 19.3%, L2 22.5 vs 15.8%** — win from cache
  locality, nothing saturated (latency/sync bound).
  Source: https://arxiv.org/abs/2605.27323
- **Gap vs ChunkyCL:** our `render()` in `integrator/path_tracer.h:72-461` is a
  textbook megakernel: octree march + water octree + BVHs + emitter-grid `while`
  shadow marches (full `closestIntersect` per step) + cloud DDA + 4x Simplex noise +
  3x `read_imagef` all live in one register footprint. Our workload is low-divergence
  (0.53 bounces/ray, many misses), so full wavefront ROI is uncertain — hence
  **wavefront-lite first** (shadow-ray kernel + queue, Phase 4 prototype).

## 4. HIPRT / wide BVH / persistent threads

- HIPRT (HIP ray-tracing framework, used by Cycles/PBRT-v4/ProRender): stack-based
  traversal with **persistent threads + dynamic fetch**, apex-point tight bounds,
  compressed BVH8/triangle packs on RDNA4.
  Source: https://gpuopen.com/download/HIPRT-paper.pdf
- **Gap vs ChunkyCL:** our world is a packed-voxel octree marched by DDA
  (`octree.h`, `intersect/octree_intersect.h`), not a triangle BVH — HW RT cores
  don't accelerate it. BVHs cover entities/actors only, so BVH8/HW-RT payoff is
  small. Persistent-thread scheduling is the transferable idea, not the BVH format.

## 5. OpenCL 1.2 constraints (we stay here until Phase 4 justifies a bump)

- No subgroups/SVM/`cl_khr_ray_tracing` (2.0+/extensions). `CL_FLOAT` images are
  `NEAREST` only — hence the adaptive HDR sky bake to 8192x4096.
- Practical levers under 1.2: explicit `localSize`, `__constant` for broadcast
  (sun/camera/scene-settings), `native_*` transcendentals where parity-tolerant,
  JIT `-D` specialization per scene (we already compile at runtime via
  `KernelLoader`), `__local` only as explicitly managed cache (naive
  `__local Scene` regressed 6.5% — don't repeat without `ncu` occupancy proof).

## 6. What we will steal (ordered)

1. **LuxCore 32-spp pass** -> Phase 1 spp-batching (host).
2. **Cycles inactive-queue + Laine queues** -> Phase 4 wavefront-lite shadow kernel.
3. **Laine register split** -> Phase 2 JIT defines + `noinline` shade helpers.
4. **HIPRT persistent-fetch** -> only if `ncu` shows launch/sync bound.
5. Explicitly **not** pursuing: ReSTIR (temporal/parity mismatch), voxel DAGs (octree
   already compact), HW RT for the world (octree DDA isn't BVH), far-emitter cull
   (rejected per ADR-0005).
