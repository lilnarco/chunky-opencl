# ChunkyCL — OpenCL Plugin

An unofficial, heavily modified fork of
[chunky-opencl](https://github.com/chunky-dev/chunky-opencl) by ThatRedox
(the OpenCL backend for [Chunky](https://github.com/chunky-dev/chunky)),
forked through [njes9701/chunky-opencl](https://github.com/njes9701/chunky-opencl)
and extended here ([lilnarco/chunky-opencl](https://github.com/lilnarco/chunky-opencl)).

> This is a vibe-coded fork of a vibe-coded fork (built with the
> [opencode](https://opencode.ai) agent this time). It works well for its authors,
> but 100% stability is not guaranteed and crashes may occur. Use at your own risk.

## What this fork adds

### Emitters

- Invisible `LightBlock` emitters work like the CPU renderer: they are in the emitter
  grid (6 unit-cube faces), rays intersect them to pick up their light, and they
  occlude shadow rays — all gated on the emitters-enabled setting.
- Entity-based emitters (campfires, candles, ...) contribute light: the emitter
  shadow-ray arrival test is now CPU-style (distance-based + material emittance)
  instead of requiring a block id that triangle hits never set.
- Emittance is packed as a full float, so Materials-tab values (0–100) render at full
  strength instead of wrapping through an 8-bit quantization.
- Emitter-grid sampling runs on all diffuse surfaces (CPU parity) and the `NONE`
  sampling strategy genuinely disables grid sampling.
- The emitter-grid GPU buffers are cached and only rebuilt when the world changes.
- The stained-glass tint boost for emitter light passing through glass is retained.

### Glass & translucent blocks

- Fully transparent texels on non-refractive translucent blocks (cherry leaves, tinted
  glass, ...) produce holes, matching the CPU's alpha check; semi-transparent texels
  use alpha-based translucency.
- Metalness/specular bounces are scaled by the texel alpha, so translucent materials
  stay translucent even when they are metallic/emissive.
- The preview renders the same holes instead of opaque white texels.

### Water

- The water octree and flowing-water geometry (corner-height triangles) are exported
  and rendered on the GPU.

### Atmosphere

- Uniform fog: distance-based haze blending toward the fog color (CPU parity — no sky
  ghosting through geometry), with a horizon sky-fog blend.
- Clouds: the CPU renderer's cloud DDA is ported — the periodic 256×256 cloud grid,
  enter/exit semantics, and cloud occlusion of light rays.
- Sky maps are baked into a float (HDR) GPU texture at an adaptive resolution up to
  8192×4096 (2:1 equirect, capped by the device's image-size limit and memory budget),
  so bright HDRI values survive the bake and 8k HDRIs render close to CPU quality.

### Sun parity

- The "draw sun" toggle only controls the sun disk sprite; sun illumination is
  independent of it (CPU behavior).
- Sun-luminosity mode uses the correct luminosity pdf for direct light.
- The sky's brightness is independent of the sun intensity slider (CPU behavior).

### OIDN denoiser (built-in)

- No external denoiser plugin needed: the OpenCL tab has a "Denoise final image with
  OIDN" checkbox, a "Denoise now" button, and the `oidnDenoise` binary path (with a
  Browse… file picker; the path is persisted between sessions).
- The renderer accumulates beauty plus internal albedo/normal passes — the guides
  converge within 32 samples and serve as OIDN inputs (they also look through
  semi-transparent texels).
- When enabled, the denoiser runs automatically when a render completes (or is
  stopped), displays the result and saves `<scene name>-<spp>_denoised.png` into the
  scene's `snapshots/` directory.
- Requires the Intel Open Image Denoise CLI (`oidnDenoise`); the passes are
  round-tripped through temporary PFM files.

### Preview

- Translucent blocks render with holes (no white texels).
- Depth of field works: the preview RNG is seeded per pixel, so the lens jitter varies
  spatially instead of being an identical shift.

### Robustness

- OpenCL platform enumeration retries after a system suspend, and a failed GPU probe
  degrades gracefully instead of failing the plugin load (Chunky keeps running with
  the CPU renderer).

### OpenCL tab

- Russian Roulette threshold (0–100%)
- Virtual octree depth (7–16)
- OpenCL device selector
- OIDN denoiser section

## Current limitations

- Layered fog is not implemented (uniform only).
- Water optics (underwater visibility, absorption, reflections) are not implemented.
- Sky maps must be `.hdr` or `.pfm` — Chunky has no EXR support.
- The OIDN denoiser requires the external `oidnDenoise` binary.
- 100% CPU parity is not guaranteed; the sky is a baked approximation of the CPU's
  per-ray sampling.
- Fix notes: [Stained glass & emitters](STAINED_GLASS_AND_EMITTER_FIXES.md),
  [Water](WATER_RENDER_FIXES.md), [Parallel projection Y-clip](PARALLEL_YCLIP_BUGFIX.md).

## Water Preview

![Water Preview 1](photo/ship.png)
![Water Preview 2](photo/ChunghwaMC%20server.png)

This is a plugin for [Chunky](https://github.com/chunky-dev/chunky) which harnesses the power of the GPU with OpenCL 1.2+ to accelerate rendering.

#### This is currently a work in progress and does not support many features. The core renderer itself is still under development so render results may change drastically between versions.

## Installation

### Note: This requires the `2.5.0` snapshots.
Download the plugin jar (from this fork's Releases page, or build it from source — see
Development), then in the Chunky Launcher expand `Advanced Settings` and click on
`Manage plugins`. In the `Plugin Manager` window click on `Add` and select the `.jar`
file. Click on `Save` and start Chunky as usual.

> The launcher's plugin list (`chunky.json`, key `"plugins"`) must contain exactly the
> jar's filename, or the plugin will fail to load with a `NoSuchFileException`.

![image](https://user-images.githubusercontent.com/42661490/116319916-28ef2580-a76c-11eb-9f93-86d444a349fd.png)

Select `ChunkyCL` as your renderer for the scene in the `Advanced` tab.

![image](https://user-images.githubusercontent.com/42661490/122492084-fc040580-cf99-11eb-9b08-b166dc25db41.png)

## Performance

Rough performance with a RTX 2070 is around 400 times that of the traditional CPU renderer as of 2022-01-27.

Some settings have been added to improve render performance.
* Indoor scenes should disable sunlight under `Lighting`
* Draw depth may be adjusted under `Advanced`
* Draw entities may be unchecked under `Advanced`
* OpenCL Device selector under `Advanced`

## Compatibility

* Not compatible with the old Denoiser Plugin (it wraps the CPU renderer). Use the
  built-in OIDN denoiser section in the OpenCL tab instead.
* Enable only this plugin in the launcher to avoid duplicate renderer registrations.

---

## Development
This project is setup to work with IntelliJ and CLion. The base directory is intended to be opened in IntelliJ and the `src/main/opencl` directory in CLion.

For hot reloading, add `-DchunkyClHotReload="<src/main/opencl directory>"` as a JVM option.

### Building from source

- JDK 17 (`JAVA_HOME` set to a JDK 17 install).
- Place `chunky-core-2.5.0-SNAPSHOT.446.ga4b37eb.jar` in the repository root (download
  from https://repo.lemaik.de/se/llbit/chunky-core/2.5.0-SNAPSHOT.446.ga4b37eb/).
- Run `./gradlew jar` — the plugin lands in `build/libs/chunky-opencl.jar`.

The OpenCL kernel is compiled with `-cl-std=CL1.2 -Werror` at runtime; the render
kernel is assembled from the headers under `src/main/opencl/kernel/include`.

## Copyright & License
ChunkyCL is Copyright (c) 2021 - 2024, [ThatRedox](https://github.com/ThatRedox) and contributors.

Permission to modify and redistribute is granted under the terms of the GPLv3 license. See the file `LICENSE` for the full license.

ChunkyCL uses the following 3rd party libraries:
* [Chunky](https://github.com/chunky-dev/chunky/)
* [JOCL](http://www.jocl.org/)
* [OpenCL header from the LLVM Project](https://llvm.org)
