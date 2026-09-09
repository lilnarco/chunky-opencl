# Memory-lean guides

The albedo/normal guide buffers are allocated full-res only when the OIDN denoiser is enabled at render start; otherwise 1-float dummies are bound and the kernel's guide march and writes are gated off. This saves ~600 MB VRAM at 5K with OIDN off, at the cost of requiring a render restart to toggle the denoiser mid-render.
