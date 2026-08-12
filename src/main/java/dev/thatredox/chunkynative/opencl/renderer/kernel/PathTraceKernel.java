package dev.thatredox.chunkynative.opencl.renderer.kernel;

import static org.jocl.CL.*;

import dev.thatredox.chunkynative.opencl.util.ClMemory;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_event;
import org.jocl.cl_kernel;
import org.jocl.cl_program;

public class PathTraceKernel implements AutoCloseable {
    private final cl_kernel kernel;
    private final cl_command_queue queue;
    private final KernelArgBinder binder;
    private ClMemory randomSeed;
    private ClMemory bufferSpp;
    private final int[] seedValue = new int[1];
    private final int[] sppValue = new int[1];

    public PathTraceKernel(cl_program program, cl_command_queue queue) {
        this.kernel = clCreateKernel(program, "render", null);
        this.queue = queue;
        this.binder = new KernelArgBinder(kernel);
    }

    public void setStaticArgs(KernelBindings bindings) {
        this.randomSeed = bindings.getGpu().getRandomSeed();
        this.bufferSpp = bindings.getGpu().getBufferSpp();

        binder.reset();

        binder.setMem(bindings.getCamera().projectorType.get());
        binder.setMem(bindings.getCamera().cameraSettings.get());

        binder.setMem(bindings.getSceneLoader().getOctreeDepth().get());
        binder.setMem(bindings.getSceneLoader().getOctreeData().get());
        binder.setMem(bindings.getSceneLoader().getWaterOctreeDepth().get());
        binder.setMem(bindings.getSceneLoader().getWaterOctreeData().get());

        binder.setMem(bindings.getSceneLoader().getBlockPalette().get());
        binder.setMem(bindings.getSceneLoader().getQuadPalette().get());
        binder.setMem(bindings.getSceneLoader().getAabbPalette().get());
        binder.setMem(bindings.getSceneLoader().getWaterPalette().get());

        binder.setMem(bindings.getSceneLoader().getWorldBvh().get());
        binder.setMem(bindings.getSceneLoader().getActorBvh().get());
        binder.setMem(bindings.getSceneLoader().getTrigPalette().get());

        binder.setMem(bindings.getSceneLoader().getTexturePalette().getAtlas());
        binder.setMem(bindings.getSceneLoader().getMaterialPalette().get());
        binder.setMem(bindings.getSceneLoader().getBiomeMeta().get());
        binder.setMem(bindings.getSceneLoader().getBiomeGrid().get());
        binder.setMem(bindings.getSceneLoader().getBiomeGrass().get());
        binder.setMem(bindings.getSceneLoader().getBiomeFoliage().get());
        binder.setMem(bindings.getSceneLoader().getBiomeDryFoliage().get());
        binder.setMem(bindings.getSceneLoader().getBiomeWater().get());
        binder.setMem(bindings.getSceneLoader().getEmitterGridMeta().get());
        binder.setMem(bindings.getSceneLoader().getEmitterGridCells().get());
        binder.setMem(bindings.getSceneLoader().getEmitterGridIndexes().get());
        binder.setMem(bindings.getSceneLoader().getEmitterGridEmitters().get());

        binder.setMem(bindings.getSceneLoader().getSky().skyTexture.get());
        binder.setMem(bindings.getSceneLoader().getSun().get());

        binder.setMem(bindings.getGpu().getRandomSeed().get());
        binder.setMem(bindings.getGpu().getBufferSpp().get());
        binder.setMem(bindings.getGpu().getCanvasConfig().get());
        binder.setMem(bindings.getGpu().getRayDepth().get());
        binder.setMem(bindings.getGpu().getSceneSettings().get());
        binder.setMem(bindings.getGpu().getAtmosphereSettings().get());
        binder.setMem(bindings.getGpu().getCloudData().get());
        binder.setInt(bindings.getSceneConstants().getEmittersEnabled());
        binder.setFloat(bindings.getSceneConstants().getEmitterIntensity());
        binder.setInt(bindings.getSceneConstants().getEmitterSamplingStrategy());
        binder.setInt(bindings.getSceneConstants().getPreventNormalEmitterWithSampling());
        binder.setMem(bindings.getGpu().getAlbedoBuffer().get());
        binder.setMem(bindings.getGpu().getNormalBuffer().get());
        binder.setMem(bindings.getGpu().getBuffer().get());
    }

    public void setPerDispatchArgs(DispatchParams params) {
        seedValue[0] = params.getRngSeed();
        sppValue[0] = params.getBufferSpp();
        clEnqueueWriteBuffer(queue, randomSeed.get(), CL_TRUE, 0, Sizeof.cl_int,
                Pointer.to(seedValue), 0, null, null);
        clEnqueueWriteBuffer(queue, bufferSpp.get(), CL_TRUE, 0, Sizeof.cl_int,
                Pointer.to(sppValue), 0, null, null);
    }

    public cl_event dispatch(long globalSize, long[] localSize, cl_event[] waitEvents) {
        cl_event event = new cl_event();
        int waitCount = waitEvents == null ? 0 : waitEvents.length;
        clEnqueueNDRangeKernel(queue, kernel, 1, null, new long[] { globalSize }, localSize, waitCount, waitEvents, event);
        return event;
    }

    public cl_kernel getKernel() {
        return kernel;
    }

    @Override
    public void close() {
        clReleaseKernel(kernel);
    }
}
