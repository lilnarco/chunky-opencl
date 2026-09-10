package dev.thatredox.chunkynative.opencl.renderer.kernel;

import static org.jocl.CL.*;

import dev.thatredox.chunkynative.opencl.util.ClMemory;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_event;
import org.jocl.cl_kernel;
import org.jocl.cl_program;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class PathTraceKernel implements AutoCloseable {
    private final cl_kernel kernel;
    private final cl_command_queue queue;
    private final KernelArgBinder binder;
    private ClMemory randomSeed;
    private ClMemory bufferSpp;
    // Direct buffers: JOCL rejects heap arrays for non-blocking writes.
    private final ByteBuffer seedValue = ByteBuffer.allocateDirect(Sizeof.cl_int).order(ByteOrder.nativeOrder());
    private final ByteBuffer sppValue = ByteBuffer.allocateDirect(Sizeof.cl_int).order(ByteOrder.nativeOrder());
    private final int[] batchValue = new int[1];
    // Kernel arg index of the trailing sppPerBatch int, captured at the end of
    // setStaticArgs so per-dispatch updates can't drift from the signature order.
    private int batchArgIndex = -1;
    // Pending async per-dispatch writes, released once the next dispatch is
    // enqueued (the host always waits on the render event first, so by then the
    // writes are long complete) or when the kernel is closed.
    private cl_event pendingSeedWrite;
    private cl_event pendingSppWrite;

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
        binder.setInt(bindings.getSceneConstants().getProfileRender());
        binder.setMem(bindings.getGpu().getProfileCounters().get());
        binder.setInt(bindings.getSceneConstants().getGuidesEnabled());
        binder.setMem(bindings.getGpu().getAlbedoBuffer().get());
        binder.setMem(bindings.getGpu().getNormalBuffer().get());
        binder.setMem(bindings.getGpu().getBuffer().get());
        batchArgIndex = binder.getArgIndex();
    }

    public void setPerDispatchArgs(DispatchParams params) {
        releasePendingWrites();
        seedValue.putInt(0, params.getRngSeed());
        sppValue.putInt(0, params.getBufferSpp());
        // Async: the queue is in-order so kernel-after-writes is already guaranteed;
        // the event chain keeps it correct if that ever changes, and removes two
        // pipeline drains per launch.
        pendingSeedWrite = new cl_event();
        pendingSppWrite = new cl_event();
        clEnqueueWriteBuffer(queue, randomSeed.get(), CL_FALSE, 0, Sizeof.cl_int,
                Pointer.to(seedValue), 0, null, pendingSeedWrite);
        clEnqueueWriteBuffer(queue, bufferSpp.get(), CL_FALSE, 0, Sizeof.cl_int,
                Pointer.to(sppValue), 0, null, pendingSppWrite);
        batchValue[0] = params.getSppPerBatch();
        clSetKernelArg(kernel, batchArgIndex, Sizeof.cl_int, Pointer.to(batchValue));
    }

    /** Wait-list for the next dispatch: both async writes must land first. */
    public cl_event[] getWriteEvents() {
        return new cl_event[] { pendingSeedWrite, pendingSppWrite };
    }

    private void releasePendingWrites() {
        if (pendingSeedWrite != null) {
            clReleaseEvent(pendingSeedWrite);
            clReleaseEvent(pendingSppWrite);
            pendingSeedWrite = null;
            pendingSppWrite = null;
        }
    }

    public cl_event dispatch(long globalSize, long[] localSize, cl_event[] waitEvents) {
        cl_event event = new cl_event();
        int waitCount = waitEvents == null ? 0 : waitEvents.length;
        clEnqueueNDRangeKernel(queue, kernel, 1, null, new long[] { globalSize }, localSize, waitCount, waitEvents, event);
        return event;
    }

    /**
     * Elapsed device nanoseconds for a completed event. Only valid with a
     * profiling queue (-DchunkyClProfiling=1); call after completion, before
     * the event is released.
     */
    public static long eventNanos(cl_event event) {
        long[] start = new long[1];
        long[] end = new long[1];
        long[] sizeRet = new long[1];
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, Sizeof.cl_ulong, Pointer.to(start), sizeRet);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, Sizeof.cl_ulong, Pointer.to(end), sizeRet);
        return end[0] - start[0];
    }

    public cl_kernel getKernel() {
        return kernel;
    }

    @Override
    public void close() {
        releasePendingWrites();
        clReleaseKernel(kernel);
    }
}
