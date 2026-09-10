package dev.thatredox.chunkynative.opencl.renderer;

import static org.jocl.CL.*;

import dev.thatredox.chunkynative.opencl.context.ContextManager;
import dev.thatredox.chunkynative.opencl.context.ClContext;
import dev.thatredox.chunkynative.opencl.renderer.ClSceneLoader;
import dev.thatredox.chunkynative.opencl.renderer.kernel.DispatchParams;
import dev.thatredox.chunkynative.opencl.renderer.kernel.KernelDefines;
import dev.thatredox.chunkynative.opencl.renderer.kernel.KernelBindings;
import dev.thatredox.chunkynative.opencl.renderer.kernel.PathTraceKernel;
import dev.thatredox.chunkynative.opencl.renderer.kernel.SceneConstants;
import dev.thatredox.chunkynative.opencl.renderer.scene.*;
import dev.thatredox.chunkynative.opencl.ui.ChunkyClTab;
import dev.thatredox.chunkynative.opencl.ui.OpenClRenderTimer;
import org.jocl.*;

import se.llbit.chunky.main.Chunky;
import se.llbit.chunky.renderer.*;
import se.llbit.chunky.renderer.export.PictureExportFormats;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.log.Log;
import se.llbit.util.TaskTracker;

import java.io.File;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BooleanSupplier;


public class OpenClPathTracingRenderer implements Renderer {

    private BooleanSupplier postRender = () -> true;

    // Number of kernel operation counters; must match PROFILE_COUNT in rt.h.
    private static final int PROFILE_COUNT = 18;

    // Phase 1 spp-batching: samples traced per kernel launch when the profiler is
    // off. Disabled under the profiler because one batched launch would overflow
    // the 32-bit GPU counters mid-launch (2M px x 32 x 314 descents >> 2^32).
    private static final int SPP_PER_BATCH = 32;

    // Optional explicit OpenCL work-group size, e.g.
    // -DchunkyClWorkGroupSize=128. Null (default) leaves the choice to the driver.
    private static final long[] WORK_GROUP_SIZE = workGroupSize();

    private static long[] workGroupSize() {
        String v = System.getProperty("chunkyClWorkGroupSize");
        if (v == null) return null;
        try {
            long n = Long.parseLong(v);
            return n > 0 ? new long[] { n } : null;
        } catch (NumberFormatException e) {
            return null;
        }
    }

    @Override
    public String getId() {
        return "ChunkyClRenderer";
    }

    @Override
    public String getName() {
        return "ChunkyClRenderer";
    }

    @Override
    public String getDescription() {
        return "ChunkyClRenderer";
    }

    @Override
    public void setPostRender(BooleanSupplier callback) {
        postRender = callback;
    }

    @Override
    public void render(DefaultRenderManager manager) throws InterruptedException {
        ContextManager context = ContextManager.get();
        ClSceneLoader sceneLoader = context.sceneLoader;

        OpenClRenderTimer.start();
        try {
            ReentrantLock renderLock = new ReentrantLock();
            Scene scene = manager.bufferedScene;

            double[] sampleBuffer = scene.getSampleBuffer();
            float[] passBuffer = new float[sampleBuffer.length];

            // Ensure the scene is loaded
            sceneLoader.ensureLoad(manager.bufferedScene);

            // Stage 1 JIT: program specialized to this scene's feature set (cached
            // per define-set; first new set pays one compile). Stamped for the
            // tab's lap timer (render vs compile).
            long buildStart = System.nanoTime();
            org.jocl.cl_program renderProgram =
                    context.renderer.kernelFor(KernelDefines.forScene(scene, sceneLoader));
            OpenClRenderTimer.addCompileNanos(System.nanoTime() - buildStart);

            try (ClCamera camera = new ClCamera(scene, context.context);
                 GpuSceneResources gpu = new GpuSceneResources(context.context, scene, passBuffer);
                 PathTraceKernel kernel = new PathTraceKernel(renderProgram, context.context.queue)) {
                RenderScheduler scheduler = new RenderScheduler(context.context.queue);
                // Generate initial camera rays
                camera.generate(renderLock, true);
                kernel.setStaticArgs(new KernelBindings(camera, sceneLoader, gpu, SceneConstants.fromScene(scene)));

                // Start the profile counters from a clean zeroed state.
                if (ChunkyClTab.profileRender) {
                    clEnqueueWriteBuffer(context.context.queue, gpu.getProfileCounters().get(), CL_TRUE, 0,
                            (long) Sizeof.cl_int * PROFILE_COUNT, Pointer.to(new int[PROFILE_COUNT]), 0, null, null);
                }

                int bufferSppReal = 0;
                int logicalSpp = scene.spp;
                long[] profileTotals = new long[PROFILE_COUNT];
                int[] lastProfileCounters = new int[PROFILE_COUNT];
                final int[] sceneSpp = {scene.spp};
                long lastCallback = 0;
                // M4: accumulated device-side kernel nanoseconds (profiling queue only).
                long kernelNanosTotal = 0;
                int launchCount = 0;

                Random rand = new Random(0);

                ForkJoinTask<?> cameraGenTask = Chunky.getCommonThreads().submit(() -> 0);
                ForkJoinTask<?> bufferMergeTask = Chunky.getCommonThreads().submit(() -> 0);

                // This is the main rendering loop. This deals with dispatching rendering tasks. The majority of time is spent
                // waiting for the OpenCL renderer to complete.
                while (logicalSpp < scene.getTargetSpp()) {
                    renderLock.lock();
                    // Batch up to SPP_PER_BATCH samples per launch; the kernel
                    // averages them in registers. Remainder/exact-target safe, and
                    // overshoot-tolerant like the old +1 path (the drain handles it).
                    int batch = ChunkyClTab.profileRender ? 1
                            : Math.max(1, Math.min(SPP_PER_BATCH, scene.getTargetSpp() - scene.spp));
                    kernel.setPerDispatchArgs(new DispatchParams(rand.nextInt(), bufferSppReal, batch));
                    cl_event renderEvent = kernel.dispatch(passBuffer.length / 3, WORK_GROUP_SIZE,
                            kernel.getWriteEvents());
                    if (ClContext.QUEUE_PROFILING) {
                        // Read before scheduler.waitFor releases the event; the extra
                        // wait is free since the dispatch already completed.
                        clWaitForEvents(1, new cl_event[] { renderEvent });
                        kernelNanosTotal += PathTraceKernel.eventNanos(renderEvent);
                    }
                    scheduler.waitFor(renderEvent);
                    renderLock.unlock();
                    launchCount++;
                    bufferSppReal += batch;
                    scene.spp += batch;

                    // Accumulate the kernel profile counters every frame. The loop already
                    // blocks on each dispatch, so this read adds no latency to the render;
                    // per-frame deltas stay far below the 32-bit wrap limit, so the
                    // modular-delta math is always exact.
                    if (ChunkyClTab.profileRender) {
                        accumulateProfile(context.context.queue, gpu, profileTotals, lastProfileCounters);
                    }

                    if (camera.needGenerate && cameraGenTask.isDone()) {
                        cameraGenTask = Chunky.getCommonThreads().submit(() -> camera.generate(renderLock, true));
                    }

                    boolean saveEvent = isSaveEvent(manager.getSnapshotControl(), scene, logicalSpp + bufferSppReal);
                    if (bufferMergeTask.isDone() || saveEvent) {
                        // "Denoise now" button: run the denoiser on the current frame.
                        if (OidnDenoiser.triggerDenoise) {
                            OidnDenoiser.triggerDenoise = false;
                            denoiseFrame(manager.context.getSceneDirectory(), scene, sampleBuffer,
                                    scene.canvasConfig.getWidth(), scene.canvasConfig.getHeight());
                        }

                        if (!scene.shouldFinalizeBuffer() && !saveEvent) {
                            long time = System.currentTimeMillis();
                            if (time - lastCallback > 100 && !manager.shouldFinalize()) {
                                lastCallback = time;
                                if (postRender.getAsBoolean()) break;
                            }
                            if (bufferSppReal < 1024)
                                continue;
                        }

                        bufferMergeTask.join();
                        if (postRender.getAsBoolean()) break;
                        clEnqueueReadBuffer(context.context.queue, gpu.getBuffer().get(), CL_TRUE, 0,
                                (long) Sizeof.cl_float * passBuffer.length, Pointer.to(passBuffer),
                                0, null, null);

                        // Stash the current albedo/normal guides so "Denoise now" can work
                        // even when the GPU render is not active.
                        if (gpu.hasGuides() && (OidnDenoiser.enabled || OidnDenoiser.lastAlbedo == null)) {
                            stashGuides(context.context.queue, gpu, passBuffer.length);
                        }

                        int sampSpp = sceneSpp[0];
                        int passSpp = bufferSppReal;
                        double sinv = 1.0 / (sampSpp + passSpp);
                        bufferSppReal = 0;

                        bufferMergeTask = Chunky.getCommonThreads().submit(() -> {
                            Arrays.parallelSetAll(sampleBuffer, i -> (sampleBuffer[i] * sampSpp + passBuffer[i] * passSpp) * sinv);
                            sceneSpp[0] += passSpp;
                            scene.postProcessFrame(TaskTracker.Task.NONE);
                            manager.redrawScreen();
                        });
                        logicalSpp += passSpp;
                        if (saveEvent) {
                            bufferMergeTask.join();
                            if (postRender.getAsBoolean()) break;
                        }
                    }
                }

                cameraGenTask.join();
                bufferMergeTask.join();

                // Kernel operation counter breakdown (Spark-style profiling).
                if (ChunkyClTab.profileRender) {
                    profileLog(profileTotals);
                }

                // M4 kernel-vs-host split (needs -DchunkyClProfiling=1 at startup).
                if (ClContext.QUEUE_PROFILING) {
                    double wallMs = OpenClRenderTimer.getElapsedMillis();
                    Log.info(String.format(
                            "OpenCL kernel time: %.2f s of %.2f s wall (%.1f%% in kernel, %d launches)",
                            kernelNanosTotal / 1e9, wallMs / 1e3,
                            100.0 * kernelNanosTotal / 1e6 / Math.max(1.0, wallMs),
                            launchCount));
                }

                // End-of-render OIDN denoising: when the render completed (target spp
                // reached) or was deliberately stopped (paused). Merges any remaining
                // frames so the denoiser sees the complete image, then displays the
                // result and saves it to the scene's snapshots directory.
                if (OidnDenoiser.enabled &&
                        (logicalSpp + bufferSppReal >= scene.getTargetSpp() ||
                                scene.getMode() == RenderMode.PAUSED)) {
                    if (bufferSppReal > 0) {
                        clEnqueueReadBuffer(context.context.queue, gpu.getBuffer().get(), CL_TRUE, 0,
                                (long) Sizeof.cl_float * passBuffer.length, Pointer.to(passBuffer),
                                0, null, null);
                        int sampSpp = sceneSpp[0];
                        int passSpp = bufferSppReal;
                        double sinv = 1.0 / (sampSpp + passSpp);
                        Arrays.parallelSetAll(sampleBuffer, i -> (sampleBuffer[i] * sampSpp + passBuffer[i] * passSpp) * sinv);
                        sceneSpp[0] += passSpp;
                        logicalSpp += passSpp;
                        bufferSppReal = 0;
                    }
                    // Fresh guides straight from the GPU.
                    if (gpu.hasGuides()) {
                        stashGuides(context.context.queue, gpu, passBuffer.length);
                    }
                    denoiseFrame(manager.context.getSceneDirectory(), scene, sampleBuffer,
                            scene.canvasConfig.getWidth(), scene.canvasConfig.getHeight());
                }
            }

        } finally {
            OpenClRenderTimer.stop();
        }
    }

    private boolean isSaveEvent(SnapshotControl control, Scene scene, int spp) {
        return control.saveSnapshot(scene, spp) || control.saveRenderDump(scene, spp);
    }

    /**
     * Read the kernel operation counters and accumulate the deltas host-side into
     * exact 64-bit totals. Called once per rendered frame, so the 32-bit GPU counters
     * wrap many times over a long render but the per-frame delta can never approach
     * 2^32 (max ~25M rays/frame at 5K) — the modular delta is therefore always exact.
     * The slot order must match the PROFILE_* defines in rt.h.
     */
    private static void accumulateProfile(cl_command_queue queue, GpuSceneResources gpu,
                                          long[] totals, int[] last) {
        int[] counters = new int[totals.length];
        clEnqueueReadBuffer(queue, gpu.getProfileCounters().get(), CL_TRUE, 0,
                (long) Sizeof.cl_int * counters.length, Pointer.to(counters), 0, null, null);
        for (int i = 0; i < counters.length; i++) {
            long delta = (counters[i] - last[i]) & 0xFFFFFFFFL;
            totals[i] += delta;
            last[i] = counters[i];
        }
    }

    /**
     * Log the profiled operation breakdown, validating the counters' internal
     * invariants (hits >= branch sum, emitter steps >= emitter rays, occluder hits
     * within shadow-ray steps, lookups <= samples <= diffuse).
     */
    private static void profileLog(long[] totals) {
        String[] names = {
                "rays", "hits", "octree steps", "bvh node tests",
                "diffuse bounces", "specular bounces", "refraction bounces",
                "emitter grid lookups", "emitter samples", "emitter rays",
                "emitter ray steps", "sun rays", "sun ray steps",
                "occluder fast-path hits", "wave noise calls", "cloud steps",
                "water plane tests", "octree descent steps"
        };
        StringBuilder sb = new StringBuilder("Profile: rays=").append(totals[0]);
        for (int i = 1; i < totals.length; i++) {
            sb.append(String.format(" | %s=%d (%.2f/ray)",
                    names[i], totals[i], totals[0] > 0 ? (double) totals[i] / totals[0] : 0));
        }
        Log.warn(sb.toString());

        // Descent depth metric: average tree levels walked per octree step. A value
        // near the octree depth means the per-step root walk is long and the bitmask
        // fast-empty-skip experiment is worth it; a value near 1-3 means it is not.
        if (totals[2] > 0) {
            Log.warn(String.format("Profile descent depth: %d levels / %d octree steps = %.2f levels per step",
                    totals[17], totals[2], (double) totals[17] / totals[2]));
        }

        long hits = totals[1];
        long diffuse = totals[4];
        long specular = totals[5];
        long refraction = totals[6];
        long lookups = totals[7];
        long samples = totals[8];
        long emitterRays = totals[9];
        long emitterSteps = totals[10];
        long sunSteps = totals[12];
        long occluder = totals[13];
        if (hits < diffuse + specular + refraction) {
            Log.warn("Profile invariant violated: hits (" + hits + ") < branch sum ("
                    + (diffuse + specular + refraction) + ")");
        }
        if (emitterSteps < emitterRays) {
            Log.warn("Profile invariant violated: emitter ray steps (" + emitterSteps
                    + ") < emitter rays (" + emitterRays + ")");
        }
        if (occluder > emitterSteps + sunSteps) {
            Log.warn("Profile invariant violated: occluder fast-path hits (" + occluder
                    + ") > shadow-ray steps (" + (emitterSteps + sunSteps) + ")");
        }
        if (lookups > samples) {
            Log.warn("Profile invariant violated: emitter grid lookups (" + lookups
                    + ") > emitter samples (" + samples + ")");
        }
        if (samples > diffuse) {
            Log.warn("Profile invariant violated: emitter samples (" + samples
                    + ") > diffuse bounces (" + diffuse + ")");
        }
    }

    /**
     * Run the OIDN denoiser on the current accumulated frame (beauty from the CPU sample
     * buffer, guides from the stashed copies), display the result and save it to
     * {@code <scene directory>/snapshots/<scene name>-<spp>_denoised.png}.
     */
    private static void denoiseFrame(File sceneDirectory, Scene scene, double[] sampleBuffer,
                                     int width, int height) {
        if (OidnDenoiser.lastAlbedo == null || OidnDenoiser.lastNormal == null) {
            Log.warn("OIDN: no albedo/normal guides available yet - render a few frames first.");
            return;
        }
        float[] beauty = new float[sampleBuffer.length];
        float[] albedo = new float[sampleBuffer.length];
        float[] normal = new float[sampleBuffer.length];
        for (int i = 0; i < sampleBuffer.length; i++) {
            beauty[i] = (float) sampleBuffer[i];
        }
        System.arraycopy(OidnDenoiser.lastAlbedo, 0, albedo, 0, albedo.length);
        System.arraycopy(OidnDenoiser.lastNormal, 0, normal, 0, normal.length);

        float[] denoised = OidnDenoiser.denoiseFinal(beauty, albedo, normal, width, height);
        if (denoised == null) {
            return;
        }

        for (int i = 0; i < sampleBuffer.length; i++) {
            sampleBuffer[i] = denoised[i];
        }
        scene.postProcessFrame(TaskTracker.Task.NONE);
        scene.setSaveSnapshots(true);
        File snapshotsDir = new File(sceneDirectory, "snapshots");
        snapshotsDir.mkdirs();
        File snapshotFile = new File(snapshotsDir,
                String.format("%s-%d_denoised.png", scene.name, scene.spp));
        scene.saveFrame(snapshotFile, PictureExportFormats.PNG, TaskTracker.NONE);
        Log.warn("OIDN: saved denoised snapshot " + snapshotFile.getAbsolutePath());
    }

    /**
     * Copy the current albedo/normal GPU buffers into {@link OidnDenoiser}'s CPU stash.
     */
    private static void stashGuides(cl_command_queue queue, GpuSceneResources gpu, int length) {
        float[] albedo = new float[length];
        float[] normal = new float[length];
        clEnqueueReadBuffer(queue, gpu.getAlbedoBuffer().get(), CL_TRUE, 0,
                (long) Sizeof.cl_float * length, Pointer.to(albedo), 0, null, null);
        clEnqueueReadBuffer(queue, gpu.getNormalBuffer().get(), CL_TRUE, 0,
                (long) Sizeof.cl_float * length, Pointer.to(normal), 0, null, null);
        OidnDenoiser.lastAlbedo = albedo;
        OidnDenoiser.lastNormal = normal;
    }

    @Override
    public boolean autoPostProcess() {
        return false;
    }

    @Override
    public void sceneReset(DefaultRenderManager manager, ResetReason reason, int resetCount) {
        boolean fullClear = reason == ResetReason.SCENE_LOADED || reason == ResetReason.MATERIALS_CHANGED;
        synchronized (manager.bufferedScene) {
            Arrays.fill(manager.bufferedScene.getSampleBuffer(), 0.0);
            manager.bufferedScene.spp = 0;
            manager.bufferedScene.renderTime = 0;
            if (fullClear) {
                Arrays.fill(manager.bufferedScene.getBackBuffer().data, 0);
                manager.bufferedScene.postProcessFrame(TaskTracker.Task.NONE);
            }
        }
        if (fullClear) {
            manager.redrawScreen();
        }
        ContextManager.get().sceneLoader.load(resetCount, reason, manager.bufferedScene);
    }
}
