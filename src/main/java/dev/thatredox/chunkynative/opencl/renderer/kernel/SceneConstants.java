package dev.thatredox.chunkynative.opencl.renderer.kernel;

import dev.thatredox.chunkynative.opencl.renderer.OidnDenoiser;
import dev.thatredox.chunkynative.opencl.ui.ChunkyClTab;
import se.llbit.chunky.renderer.scene.Scene;

public class SceneConstants {
    private final int emittersEnabled;
    private final float emitterIntensity;
    private final int emitterSamplingStrategy;
    private final int preventNormalEmitterWithSampling;
    private final int profileRender;
    private final int guidesEnabled;

    private SceneConstants(int emittersEnabled, float emitterIntensity, int emitterSamplingStrategy, int preventNormalEmitterWithSampling, int profileRender, int guidesEnabled) {
        this.emittersEnabled = emittersEnabled;
        this.emitterIntensity = emitterIntensity;
        this.emitterSamplingStrategy = emitterSamplingStrategy;
        this.preventNormalEmitterWithSampling = preventNormalEmitterWithSampling;
        this.profileRender = profileRender;
        this.guidesEnabled = guidesEnabled;
    }

    public static SceneConstants fromScene(Scene scene) {
        return new SceneConstants(
                scene.getEmittersEnabled() ? 1 : 0,
                (float) scene.getEmitterIntensity(),
                scene.getEmitterSamplingStrategy().ordinal(),
                scene.isPreventNormalEmitterWithSampling() ? 1 : 0,
                ChunkyClTab.profileRender ? 1 : 0,
                OidnDenoiser.enabled ? 1 : 0
        );
    }

    public int getEmittersEnabled() {
        return emittersEnabled;
    }

    public float getEmitterIntensity() {
        return emitterIntensity;
    }

    public int getEmitterSamplingStrategy() {
        return emitterSamplingStrategy;
    }

    public int getPreventNormalEmitterWithSampling() {
        return preventNormalEmitterWithSampling;
    }

    public int getProfileRender() {
        return profileRender;
    }

    public int getGuidesEnabled() {
        return guidesEnabled;
    }
}
