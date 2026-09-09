# Sun and emitter sampling gates

All sun sampling is skipped when the sun disk is below the horizon or its intensity is ~0, and emitter sampling is skipped at zero emitter intensity — a down sun contributes zero light, so only wasted shadow rays are removed (~16% faster on the dev benchmark). Culling the water octree from shadow rays was rejected (it would break light attenuation through water), and a far-emitter distance cull was explicitly rejected by the maintainer — do not propose it again.
