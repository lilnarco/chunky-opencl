package dev.thatredox.chunkynative.opencl.renderer;

import dev.thatredox.chunkynative.opencl.context.ContextManager;
import dev.thatredox.chunkynative.opencl.renderer.ClSceneLoader;
import dev.thatredox.chunkynative.opencl.renderer.kernel.KernelDefines;
import dev.thatredox.chunkynative.opencl.renderer.scene.*;
import dev.thatredox.chunkynative.opencl.util.ClIntBuffer;
import dev.thatredox.chunkynative.opencl.util.ClMemory;
import org.jocl.*;
import se.llbit.chunky.renderer.DefaultRenderManager;
import se.llbit.chunky.renderer.Renderer;
import se.llbit.chunky.renderer.ResetReason;
import se.llbit.chunky.renderer.projection.ProjectionMode;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.util.TaskTracker;
import se.llbit.util.Mutable;

import java.util.Arrays;
import java.util.function.BooleanSupplier;

import static org.jocl.CL.*;

public class OpenClPreviewRenderer implements Renderer {
    private BooleanSupplier postRender = () -> true;

    // Persistent per-frame resources (this instance lives for the session, and
    // render() runs serially on the preview thread): kernel and pixel buffer are
    // rebuilt only on canvas/program/context change. The CAMERA is persistent
    // only for pre-generated (panoramic-family) projections, whose buffers are
    // full-res and whose generate() refreshes rays from the live camera every
    // frame; pinhole/parallel cameras snapshot position into ~10 floats at
    // construction, so they are recreated per frame (exactly like before) —
    // persisting them would freeze the preview on camera moves.
    private cl_kernel previewKernel = null;
    private String previewDefines = null;
    private ClCamera previewCamera = null;
    private ClMemory previewBuffer = null;
    private int previewPixels = -1;
    private ContextManager previewContext = null;

    @Override
    public String getId() {
        return "ChunkyClPreviewRenderer";
    }

    @Override
    public String getName() {
        return "Chunky CL Preview Renderer";
    }

    @Override
    public String getDescription() {
        return "A work in progress OpenCL renderer.";
    }

    @Override
    public void setPostRender(BooleanSupplier callback) {
        postRender = callback;
    }

    @Override
    public void render(DefaultRenderManager manager) throws InterruptedException {
        ContextManager context = ContextManager.get();
        ClSceneLoader sceneLoader = context.sceneLoader;

        cl_event[] renderEvent = new cl_event[1];
        Scene scene = manager.bufferedScene;
        int[] imageData = scene.getBackBuffer().data;

        // Ensure the scene is loaded
        sceneLoader.ensureLoad(manager.bufferedScene);

        // Rebuild persistent kernel+buffer only on canvas/program/context change.
        String defines = KernelDefines.forScene(scene, sceneLoader);
        if (previewKernel == null || previewContext != context
                || !defines.equals(previewDefines) || imageData.length != previewPixels) {
            releasePreview();
            previewKernel = clCreateKernel(context.renderer.kernelFor(defines), "preview", null);
            previewBuffer = new ClMemory(clCreateBuffer(context.context.context, CL_MEM_WRITE_ONLY,
                    (long) Sizeof.cl_int * imageData.length, null, null));
            previewDefines = defines;
            previewPixels = imageData.length;
            previewContext = context;
        }
        cl_kernel kernel = previewKernel;
        ClMemory buffer = previewBuffer;

        // Camera: panoramic pre-generated buffers are full-res, so the camera
        // persists (generate() refreshes rays from the live camera every frame;
        // canvas resizes drop it via releasePreview above). Pinhole/parallel
        // snapshots position into ~10 floats at construction, so a fresh one is
        // built per frame — exactly like before, and moves always apply.
        // A projection-mode switch drops the persisted camera either way.
        ProjectionMode projectionMode = scene.camera().getProjectionMode();
        boolean wantGenerated = projectionMode != ProjectionMode.PINHOLE
                && projectionMode != ProjectionMode.PARALLEL;
        ClCamera frameCamera = previewCamera;
        if (frameCamera == null || frameCamera.needGenerate != wantGenerated) {
            frameCamera = new ClCamera(scene, context.context);
            if (wantGenerated && frameCamera.needGenerate) {
                if (previewCamera != null) {
                    previewCamera.close();
                }
                previewCamera = frameCamera;
            }
        }
        boolean ownCamera = frameCamera != previewCamera;

        ClIntBuffer clCanvasConfig = new ClIntBuffer(new int[] {
                scene.canvasConfig.getWidth(), scene.canvasConfig.getHeight(),
                scene.canvasConfig.getCropWidth(), scene.canvasConfig.getCropHeight(),
                scene.canvasConfig.getCropX(), scene.canvasConfig.getCropY()
        }, context.context);

        try (ClIntBuffer ignored = clCanvasConfig;
             ClCamera owned = ownCamera ? frameCamera : null) {

            // Generate the camera rays
            frameCamera.generate(null, false);

            renderEvent[0] = new cl_event();

            try {
            int argIndex = 0;
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(frameCamera.projectorType.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(frameCamera.cameraSettings.get()));

            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getOctreeDepth().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getOctreeData().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getWaterOctreeDepth().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getWaterOctreeData().get()));

            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getBlockPalette().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getQuadPalette().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getAabbPalette().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getWaterPalette().get()));

            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getWorldBvh().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getActorBvh().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getTrigPalette().get()));

            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getTexturePalette().getAtlas()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getMaterialPalette().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getBiomeMeta().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getBiomeGrid().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getBiomeGrass().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getBiomeFoliage().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getBiomeDryFoliage().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getBiomeWater().get()));

            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getSky().skyTexture.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getSun().get()));

            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(clCanvasConfig.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(buffer.get()));
            clEnqueueNDRangeKernel(context.context.queue, kernel, 1, null,
                    new long[]{imageData.length}, null, 0, null,
                    renderEvent[0]);

            clEnqueueReadBuffer(context.context.queue, buffer.get(), CL_TRUE, 0,
                    (long) Sizeof.cl_int * imageData.length, Pointer.to(imageData),
                    1, renderEvent, null);
            } finally {
                clReleaseEvent(renderEvent[0]);
            }

            manager.redrawScreen();
            postRender.getAsBoolean();
        }
    }

    /** Release persistent frame resources (rebuild on next render). */
    private void releasePreview() {
        if (previewKernel != null) {
            clReleaseKernel(previewKernel);
            previewKernel = null;
        }
        if (previewCamera != null) {
            previewCamera.close();
            previewCamera = null;
        }
        if (previewBuffer != null) {
            previewBuffer.close();
            previewBuffer = null;
        }
        previewDefines = null;
        previewPixels = -1;
        previewContext = null;
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

