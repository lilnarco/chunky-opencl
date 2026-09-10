package dev.thatredox.chunkynative.opencl.renderer.kernel;

import dev.thatredox.chunkynative.opencl.renderer.ClSceneLoader;
import se.llbit.chunky.renderer.scene.Scene;

/**
 * Stage 1 JIT specialization: the canonical OpenCL {@code -D} set for a scene.
 * Each flag mirrors its runtime gate exactly (see {@code GpuSceneResources} for
 * the buffer slots and the kernel for the gates), so a specialized binary is
 * behavior-identical to the default one — the compiler just gets to dead-strip
 * the unreachable blocks, shrinking register pressure on the SM-bound
 * megakernel. The returned string is canonical (fixed order) so it doubles as
 * the program-cache key in {@code ContextManager.Renderer#kernelFor}.
 *
 * <p>Runtime-only toggles (profiler, OIDN guides, work-group size) stay kernel
 * arguments: specializing on them would recompile on UI clicks.
 */
public final class KernelDefines {
    private KernelDefines() {
    }

    public static String forScene(Scene scene, ClSceneLoader sceneLoader) {
        // Kill-switch (A/B runs, paranoia): single everything-on program, no
        // specialization. -DchunkyClJit=off
        if (System.getProperty("chunkyClJit", "on").equalsIgnoreCase("off")) {
            se.llbit.log.Log.info("ChunkyCL: JIT specialization disabled, using full program.");
            return "-DFOG_MODE=" + scene.fog.getFogMode().ordinal()
                    + " -DHAS_WATER=1 -DHAS_WATERPLANE=1 -DHAS_CLOUDS=1 -DHAS_EMITTERS=1 -DHAS_SUN=1";
        }
        StringBuilder sb = new StringBuilder();
        // Always present: NONE = 0, UNIFORM = 1, LAYERED = 2 (FogMode order).
        sb.append("-DFOG_MODE=").append(scene.fog.getFogMode().ordinal());
        if (sceneLoader.hasWater()) {
            sb.append(" -DHAS_WATER=1");
        }
        if (scene.isWaterPlaneEnabled()) {
            sb.append(" -DHAS_WATERPLANE=1");
        }
        if (scene.sky().cloudsEnabled()) {
            sb.append(" -DHAS_CLOUDS=1");
        }
        if (scene.getEmittersEnabled()) {
            sb.append(" -DHAS_EMITTERS=1");
        }
        if (hasSun(scene)) {
            sb.append(" -DHAS_SUN=1");
        }
        return sb.toString();
    }

    /**
     * Mirrors the kernel's sun-block entry condition
     * ({@code doSunSampling && intensity > EPS && sw.y >= 0}): only when every
     * clause holds can the block run, so omitting {@code HAS_SUN} otherwise is
     * exact. EPS matches the kernel's {@code constants.h} (5e-6).
     */
    private static boolean hasSun(Scene scene) {
        if (!scene.getSunSamplingStrategy().doSunSampling()) {
            return false;
        }
        if (!(scene.sun().getIntensity() > 0.000005)) {
            return false;
        }
        return scene.sun().getAltitude() >= 0.0;
    }
}
