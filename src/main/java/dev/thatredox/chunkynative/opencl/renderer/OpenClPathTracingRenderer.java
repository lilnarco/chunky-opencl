package dev.thatredox.chunkynative.opencl.renderer;

import static org.jocl.CL.*;

import dev.thatredox.chunkynative.opencl.context.ContextManager;
import dev.thatredox.chunkynative.opencl.renderer.ClSceneLoader;
import dev.thatredox.chunkynative.opencl.renderer.kernel.DispatchParams;
import dev.thatredox.chunkynative.opencl.renderer.kernel.KernelBindings;
import dev.thatredox.chunkynative.opencl.renderer.kernel.PathTraceKernel;
import dev.thatredox.chunkynative.opencl.renderer.kernel.SceneConstants;
import dev.thatredox.chunkynative.opencl.renderer.scene.*;
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

            try (ClCamera camera = new ClCamera(scene, context.context);
                 GpuSceneResources gpu = new GpuSceneResources(context.context, scene, passBuffer);
                 PathTraceKernel kernel = new PathTraceKernel(context.renderer.kernel, context.context.queue)) {
                RenderScheduler scheduler = new RenderScheduler(context.context.queue);
                // Generate initial camera rays
                camera.generate(renderLock, true);
                kernel.setStaticArgs(new KernelBindings(camera, sceneLoader, gpu, SceneConstants.fromScene(scene)));

                int bufferSppReal = 0;
                int logicalSpp = scene.spp;
                final int[] sceneSpp = {scene.spp};
                long lastCallback = 0;

                Random rand = new Random(0);

                ForkJoinTask<?> cameraGenTask = Chunky.getCommonThreads().submit(() -> 0);
                ForkJoinTask<?> bufferMergeTask = Chunky.getCommonThreads().submit(() -> 0);

                // This is the main rendering loop. This deals with dispatching rendering tasks. The majority of time is spent
                // waiting for the OpenCL renderer to complete.
                while (logicalSpp < scene.getTargetSpp()) {
                    renderLock.lock();
                    kernel.setPerDispatchArgs(new DispatchParams(rand.nextInt(), bufferSppReal));
                    cl_event renderEvent = kernel.dispatch(passBuffer.length / 3, null, null);
                    scheduler.waitFor(renderEvent);
                    renderLock.unlock();
                    bufferSppReal += 1;
                    scene.spp += 1;

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
                        if (OidnDenoiser.enabled || OidnDenoiser.lastAlbedo == null) {
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
                    stashGuides(context.context.queue, gpu, passBuffer.length);
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
