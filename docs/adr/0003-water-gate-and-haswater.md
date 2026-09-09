# Water-gate and hasWater

The water octree march and water medium lookup run only when the host detects water in the palette (`hasWater`). Waterless scenes previously paid ~2 wasted marches per ray; the runtime gate captures that waste without kernel recompilation.

## Considered Options

- JIT feature-specialisation via `-D` defines (dead water/cloud/emitter/fog paths compiled out per scene): deferred — the water-gate captures the only unconditional executed-work waste, and the rest only reduces register/icache pressure, which is speculative until occupancy data says the megakernel is register-bound.
