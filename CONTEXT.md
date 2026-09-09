# ChunkyCL

GPU (OpenCL 1.2) backend for the Chunky voxel renderer, pursuing CPU parity at GPU speed.

## Language

**Octree march**:
The DDA walk through the packed voxel octree that finds each ray's block hit.
_Avoid_: voxel traversal, grid march

**Water octree**:
A second octree holding only water blocks, marched separately from the solid world.
_Avoid_: liquid octree, water grid

**Emitter grid**:
The acceleration grid over emissive surfaces used for direct-light sampling.
_Avoid_: light grid, emission grid

**Guides**:
The auxiliary albedo and normal passes accumulated for the OIDN denoiser.
_Avoid_: AOVs, feature passes, auxiliary buffers

**Sun gate**:
Skipping all sun shadow rays when the sun disk is below the horizon or its intensity is ~0.
_Avoid_: sun cull, daylight check

**Water-gate**:
Skipping the water octree march and water medium lookup in scenes without water.
_Avoid_: water cull, aqua skip

**Layered fog**:
Height-dependent fog extinction integrated over the path's y-span (CPU parity).
_Avoid_: altitude fog, gradient fog

**Profile counters**:
The kernel-side atomic counters of ray operations, accumulated host-side per frame.
_Avoid_: stats buffer, perf counters

**Dynamic offset**:
The host-computed march offset from the maximum coordinate magnitude, keeping rays moving at large ranges.
_Avoid_: virtual depth, ray epsilon

**Water-shader selector**:
The render-time override choosing still or Simplex water normals (or the scene default).
_Avoid_: wave toggle, water mode
