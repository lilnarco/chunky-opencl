# ChunkyCL — OpenCL Plugin

An unofficial, heavily modified fork of
[chunky-opencl](https://github.com/chunky-dev/chunky-opencl) by ThatRedox
(the OpenCL backend for [Chunky](https://github.com/chunky-dev/chunky)),
forked through [njes9701/chunky-opencl](https://github.com/njes9701/chunky-opencl)
and extended here ([lilnarco/chunky-opencl](https://github.com/lilnarco/chunky-opencl)).

> Built with AI coding agents. It works well for its authors, but 100%
> stability is not guaranteed and crashes may occur. Use at your own risk.

## What this fork adds

### Emitters

- Invisible `LightBlock` emitters work like the CPU renderer: they are in the emitter
  grid (6 unit-cube faces), rays intersect them to pick up their light, and they
  occlude shadow rays — all gated on the emitters-enabled setting.
- Entity-based emitters (campfires, candles, ...) contribute light: the emitter
  shadow-ray arrival test is CPU-style (distance-based + material emittance).
- Emittance is packed as a full float, so Materials-tab values (0–100) render at full
  strength instead of wrapping through an 8-bit quantization.
- Emitter-grid sampling runs on all diffuse surfaces (CPU parity) and the `NONE`
  sampling strategy genuinely disables grid sampling.
- The emitter-grid GPU buffers are cached and only rebuilt when the world changes.
- The stained-glass tint boost for emitter light passing through glass is retained.
- Emitter sampling is skipped entirely when the emitter intensity is ~0.

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
- Underwater visibility attenuation: density falls off with distance travelled
  through water (`exp(-d/visibility)`), black at zero visibility.
- Infinite water plane below the loaded chunks (matches the default-off setting),
  using the real water material.
- Simplex water waves (faithful port of the CPU noise): normal perturbation at
  water-air boundaries. Selectable per render via the water-shader selector
  (scene default / still / simplex).
- Water surface alpha follows the CPU (`waterOpacity`, default 0.42).
- Sky misses from underwater are black (CPU parity — no sky underlay beneath
  a water world), in render and preview.
- The water octree march is skipped entirely in waterless scenes (`hasWater` gate).

### Atmosphere

- Uniform fog: distance-based haze blending toward the fog color (CPU parity — no sky
  ghosting through geometry), with a horizon sky-fog blend.
- Layered fog (CPU parity): height-dependent extinction integrated over the path's
  y-span, including sky fog out to the fog limit.
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
- Sun sampling is skipped entirely when the sun disk is below the horizon or the
  sun intensity is ~0 — a down sun contributes no light, so this removes only
  wasted shadow rays.

### OIDN denoiser (built-in)

- No external denoiser plugin needed: the OpenCL tab has a "Denoise final image with
  OIDN" checkbox, a "Denoise now" button, and the `oidnDenoise` binary path (with a
  Browse… file picker; the path is persisted between sessions).
- The renderer accumulates beauty plus internal albedo/normal guides — the guides
  converge within 32 samples and serve as OIDN inputs (they also look through
  semi-transparent texels).
- Guide buffers are allocated full-res only when OIDN is enabled at render start;
  otherwise the guide passes are gated off (saves ~600 MB VRAM at 5K with OIDN off).
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

- Render time display
- Russian Roulette threshold slider (0–100%, default 50%)
- Water shader selector (Scene default / Still / Simplex)
- Profile render checkbox (kernel operation counters — experimental, output format
  not yet documented)
- OIDN denoiser section (enable, binary path, Denoise now)
- OpenCL device selector button
- Kernel status line (`Kernel: compiled in Ns` vs `Kernel: cache, ready in Ns`)
  plus a render-vs-compile lap timer

### Work-group size tuning

The driver picks the OpenCL work-group size unless overridden with
`-DchunkyClWorkGroupSize=N` (JVM option, read once per session — restart
Chunky between values). Smaller groups can hide traversal latency better on
register-bound kernels. Measured 1080p×200 on RTX 3070 / 550.x (sun scene):
64 → 26 s, 128 → 27 s, 256 / driver default → 28 s. The optimum is
device- and scene-specific, so it stays a knob, not a default — tune once per
GPU.

### Kernel compile cache

The render kernel is specialized to each scene's feature set and cached twice:
linked binaries persist under `<chunky.home>/kernel-cache/`, and NVIDIA keeps
its own compile cache (`~/.nv/ComputeCache`, default cap 256 MB shared with
all CUDA apps). A first-seen feature set pays one compile (up to ~1 min on a
big scene); later sessions reuse it.

If session startup still shows long `Kernel:` times, the driver cache is likely
full and evicting entries — raise it before launching:

```
export CUDA_CACHE_MAXSIZE=1073741824
```

Measured on RTX 3070 / 550.x: 21 s → 0 s backend time. Safe to delete
`kernel-cache/` anytime (rebuilds as needed); entries self-invalidate after
plugin or driver updates. To disable specialization entirely (single
everything-on program, e.g. for A/B runs): `-DchunkyClJit=off`.

## Current limitations

- Subsurface scattering is not implemented.
- Water plane chunk clipping is not implemented.
- Sky maps must be `.hdr` or `.pfm` — Chunky has no EXR support.
- The OIDN denoiser requires the external `oidnDenoise` binary.
- 100% CPU parity is not guaranteed; the sky is a baked approximation of the CPU's
  per-ray sampling.

## Gallery

![Water Preview 1](photo/ship.png)
![Water Preview 2](photo/ChunghwaMC%20server.png)

Wanted (not yet captured — contributions welcome):

- Layered vs uniform fog on a tall scene
- Simplex vs still water on a water-heavy scene
- OIDN off vs on at low spp, side by side

## Installation

### Note: This requires the `2.5.0` snapshots.

Download the plugin jar (from this fork's Releases page, or build it from source — see
Development), then in the Chunky Launcher expand `Advanced Settings` and click on
`Manage plugins`. In the `Plugin Manager` window click on `Add` and select the `.jar`
file. Click on `Save` and start Chunky as usual.

> The launcher's plugin list (`chunky.json`, key `"plugins"`) must contain exactly the
> jar's filename, or the plugin will fail to load with a `NoSuchFileException`.

Select `ChunkyCL` as your renderer for the scene in the `Advanced` tab.

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
- Run `JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 ./gradlew clean jar` (adjust the
  JDK path to your machine) — the plugin lands in `build/libs/chunky-opencl.jar`.

The OpenCL kernel is compiled with `-cl-std=CL1.2 -Werror` at runtime; the render
kernel is assembled from the headers under `src/main/opencl/kernel/include`.

Domain terms for contributors live in [CONTEXT.md](CONTEXT.md); architectural
decisions in [docs/adr/](docs/adr/).

## Copyright & License

ChunkyCL is Copyright (c) 2021 - 2024, [ThatRedox](https://github.com/ThatRedox) and contributors.

Permission to modify and redistribute is granted under the terms of the GPLv3 license. See the file `LICENSE` for the full license.

ChunkyCL uses the following 3rd party libraries:
* [Chunky](https://github.com/chunky-dev/chunky/)
* [JOCL](http://www.jocl.org/)
* [OpenCL header from the LLVM Project](https://llvm.org)
