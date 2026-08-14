package dev.thatredox.chunkynative.opencl.renderer;

import static org.jocl.CL.*;

import dev.thatredox.chunkynative.opencl.context.ClContext;
import dev.thatredox.chunkynative.opencl.context.ContextManager;
import dev.thatredox.chunkynative.opencl.ui.ChunkyClTab;
import dev.thatredox.chunkynative.opencl.util.ClIntBuffer;
import dev.thatredox.chunkynative.opencl.util.ClMemory;
import dev.thatredox.chunkynative.util.Reflection;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import se.llbit.chunky.renderer.WaterShadingStrategy;
import se.llbit.chunky.renderer.scene.Scene;

import java.lang.reflect.Field;

public class GpuSceneResources implements AutoCloseable {
    private final ClContext context;
    private final ClMemory buffer;
    private final ClMemory albedoBuffer;
    private final ClMemory normalBuffer;
    private final ClMemory randomSeed;
    private final ClMemory bufferSpp;
    private final ClIntBuffer canvasConfig;
    private final ClIntBuffer rayDepth;
    private final ClMemory sceneSettings;
    private final ClMemory atmosphereSettings;
    private final ClIntBuffer cloudData;
    private final ClMemory profileCounters;

    public GpuSceneResources(ClContext context, Scene scene, float[] passBuffer) {
        this.context = context;

        this.buffer = new ClMemory(clCreateBuffer(context.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                (long) Sizeof.cl_float * passBuffer.length, Pointer.to(passBuffer), null));
        // Auxiliary render passes (albedo/normal), used by the pass modes and OIDN denoising.
        this.albedoBuffer = new ClMemory(clCreateBuffer(context.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                (long) Sizeof.cl_float * passBuffer.length, Pointer.to(passBuffer), null));
        this.normalBuffer = new ClMemory(clCreateBuffer(context.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                (long) Sizeof.cl_float * passBuffer.length, Pointer.to(passBuffer), null));
        this.randomSeed = new ClMemory(clCreateBuffer(context.context, CL_MEM_READ_ONLY, Sizeof.cl_int, null, null));
        this.bufferSpp = new ClMemory(clCreateBuffer(context.context, CL_MEM_READ_ONLY, Sizeof.cl_int, null, null));

        this.canvasConfig = new ClIntBuffer(new int[] {
                scene.canvasConfig.getWidth(), scene.canvasConfig.getHeight(),
                scene.canvasConfig.getCropWidth(), scene.canvasConfig.getCropHeight(),
                scene.canvasConfig.getCropX(), scene.canvasConfig.getCropY()
        }, context);
        this.rayDepth = new ClIntBuffer(scene.getRayDepth(), context);

        // Maximum coordinate magnitude for the octree march's dynamic offset: the larger
        // of the octree extent and the camera's distance from the origin, with a safety
        // margin so float precision stays safe at large coordinates.
        double octreeExtent = Math.pow(2, scene.getWorldOctree().getDepth());
        double camX = scene.camera().getPosition().x - scene.getOrigin().x;
        double camY = scene.camera().getPosition().y - scene.getOrigin().y;
        double camZ = scene.camera().getPosition().z - scene.getOrigin().z;
        float maxCoord = (float) (Math.max(octreeExtent, Math.sqrt(camX * camX + camY * camY + camZ * camZ)) * 1.5);

        this.sceneSettings = new ClMemory(
                clCreateBuffer(context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        (long) Sizeof.cl_float * 7,
                        Pointer.to(new float[] {
                                ((Double) Reflection.getFieldValue(scene, "transmissivityCap", Double.class)).floatValue(),
                                ((Boolean) Reflection.getFieldValue(scene, "fancierTranslucency", Boolean.class)) ? 1.0f : 0.0f,
                                scene.getSunSamplingStrategy().doSunSampling() ? 1.0f : 0.0f,
                                scene.getSunSamplingStrategy().isSunLuminosity() ? 1.0f : 0.0f,
                                scene.getSunSamplingStrategy().isStrictDirectLight() ? 1.0f : 0.0f,
                                ChunkyClTab.russianRouletteThreshold,
                                maxCoord
                        }), null));

        // Fog + cloud + water settings. Layout (floats):
        // 0: fog mode (0 = NONE, 1 = UNIFORM, 2 = LAYERED)
        // 1-3: fog color
        // 4: uniform density, 5: sky fog density, 6: fast fog
        // 7: clouds enabled, 8: cloud size, 9-11: cloud offset
        // 12-14: octree origin
        // 15: water visibility, 16: water plane enabled, 17: water plane height (world)
        // 18: water shader (0 = still, 1 = simplex), 19: animation time
        // 20: water material palette id (float bits)
        // 21: water opacity
        WaterShadingStrategy waterShader = scene.getWaterShadingStrategy();
        int waterShaderId = waterShader == WaterShadingStrategy.STILL ? 0 : 1;
        this.atmosphereSettings = new ClMemory(
                clCreateBuffer(context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        (long) Sizeof.cl_float * 22,
                        Pointer.to(new float[] {
                                scene.fog.getFogMode().ordinal(),
                                (float) scene.fog.getFogColor().x,
                                (float) scene.fog.getFogColor().y,
                                (float) scene.fog.getFogColor().z,
                                (float) scene.fog.getUniformDensity(),
                                (float) scene.fog.getSkyFogDensity(),
                                scene.fog.fastFog() ? 1.0f : 0.0f,
                                scene.sky().cloudsEnabled() ? 1.0f : 0.0f,
                                (float) scene.sky().cloudSize(),
                                (float) scene.sky().cloudXOffset(),
                                (float) scene.sky().cloudYOffset(),
                                (float) scene.sky().cloudZOffset(),
                                scene.getOrigin().x,
                                scene.getOrigin().y,
                                scene.getOrigin().z,
                                (float) scene.getWaterVisibility(),
                                scene.isWaterPlaneEnabled() ? 1.0f : 0.0f,
                                (float) scene.getEffectiveWaterPlaneHeight(),
                                waterShaderId,
                                (float) scene.getAnimationTime(),
                                Float.intBitsToFloat(ContextManager.get().sceneLoader.getWaterMaterialId()),
                                (float) scene.getWaterOpacity()
                        }), null));

        // Kernel operation counters: the kernel atomically writes these, so the buffer
        // must be READ_WRITE (ClIntBuffer creates READ_ONLY buffers).
        this.profileCounters = new ClMemory(
                clCreateBuffer(context.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        (long) Sizeof.cl_int * 17, Pointer.to(new int[17]), null));

        this.cloudData = new ClIntBuffer(exportCloudData(), context);
    }

    private static int[] exportCloudData() {
        int[] data = new int[2048];
        try {
            Field cloudField = Class.forName("se.llbit.chunky.world.Clouds").getDeclaredField("clouds");
            cloudField.setAccessible(true);
            long[][] clouds = (long[][]) cloudField.get(null);
            for (int tiley = 0; tiley < 32; tiley++) {
                for (int tilex = 0; tilex < 32; tilex++) {
                    long value = clouds[tilex][tiley];
                    int idx = (tiley * 32 + tilex) * 2;
                    data[idx] = (int) value;
                    data[idx + 1] = (int) (value >>> 32);
                }
            }
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("Failed to export cloud data", e);
        }
        return data;
    }

    public ClContext getContext() {
        return context;
    }

    public ClMemory getBuffer() {
        return buffer;
    }

    public ClMemory getAlbedoBuffer() {
        return albedoBuffer;
    }

    public ClMemory getNormalBuffer() {
        return normalBuffer;
    }

    public ClMemory getRandomSeed() {
        return randomSeed;
    }

    public ClMemory getBufferSpp() {
        return bufferSpp;
    }

    public ClIntBuffer getCanvasConfig() {
        return canvasConfig;
    }

    public ClIntBuffer getRayDepth() {
        return rayDepth;
    }

    public ClMemory getSceneSettings() {
        return sceneSettings;
    }

    public ClMemory getAtmosphereSettings() {
        return atmosphereSettings;
    }

    public ClIntBuffer getCloudData() {
        return cloudData;
    }

    public ClMemory getProfileCounters() {
        return profileCounters;
    }

    @Override
    public void close() {
        profileCounters.close();
        cloudData.close();
        atmosphereSettings.close();
        sceneSettings.close();
        rayDepth.close();
        canvasConfig.close();
        bufferSpp.close();
        randomSeed.close();
        normalBuffer.close();
        albedoBuffer.close();
        buffer.close();
    }
}
