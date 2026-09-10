package dev.thatredox.chunkynative.opencl.context;

import dev.thatredox.chunkynative.opencl.renderer.ClSceneLoader;
import org.jocl.CLException;
import org.jocl.cl_program;
import se.llbit.log.Log;

public class ContextManager {
    public final Device device;
    public final ClContext context;
    public final Tonemap tonemap;
    public final Renderer renderer;

    public final ClSceneLoader sceneLoader;

    private static volatile ContextManager instance;

    private ContextManager(Device device) {
        this.device = device;
        this.context = new ClContext(device);
        this.tonemap = new Tonemap(context);
        this.renderer = new Renderer(context);
        this.sceneLoader = new ClSceneLoader(context);
    }

    /**
     * Lazily initialized so a failed device probe (e.g. the GPU not having recovered
     * after a system suspend) does not permanently poison the class.
     */
    public static ContextManager get() {
        ContextManager local = instance;
        if (local == null) {
            synchronized (ContextManager.class) {
                local = instance;
                if (local == null) {
                    instance = local = new ContextManager(Device.getPreferredDevice());
                }
            }
        }
        return local;
    }

    public static synchronized void setDevice(Device device) {
        try {
            instance = new ContextManager(device);
        } catch (CLException e) {
            Log.error("Failed to set device", e);
        }
    }

    public static synchronized void reload() {
        setDevice(instance.device);
    }

    public static class Tonemap {
        public final cl_program simpleFilter;

        private Tonemap(ClContext context) {
            this.simpleFilter = KernelLoader.loadProgram(context, "tonemap", "post_processing_filter.c");
        }
    }

    public static class Renderer {
        private final ClContext context;
        // Stage 1 JIT: programs specialized per define-set, built lazily on first
        // use (render or preview). Nothing compiles at startup anymore; a scene
        // pays one build when its feature set is first seen, then reuses the
        // binary (in-session map + on-disk cache, see KernelBinaryCache).
        private final java.util.Map<String, cl_program> specialized = new java.util.HashMap<>();

        // When non-null, a kernel build is in progress; the OpenCL tab shows this
        // instead of the render time so a ~1 min first-compile doesn't look frozen.
        // Written on the render thread, read on the FX ticker thread.
        public static volatile String compileStatus = null;
        // Last build outcome for the tab ("Kernel: cache, ready in 9 s" etc.).
        public static volatile String lastBuildOutcome = null;

        private Renderer(ClContext context) {
            this.context = context;
        }

        public synchronized cl_program kernelFor(String options) {
            cl_program program = specialized.get(options);
            if (program == null) {
                String label = options.isEmpty() ? "<default>" : options;
                Log.info("ChunkyCL: compiling OpenCL kernel [" + label + "]...");
                compileStatus = "Compiling OpenCL kernel… (one-time, up to ~1 min)";
                long t0 = System.nanoTime();
                try {
                    program = KernelBinaryCache.tryLoad(context, options);
                    if (program == null) {
                        program = KernelLoader.loadProgram(context, "kernel", "rayTracer.c", options);
                        KernelBinaryCache.store(context, program, options);
                        lastBuildOutcome = String.format("Kernel: compiled in %.0f s",
                                (System.nanoTime() - t0) / 1e9);
                    } else {
                        lastBuildOutcome = String.format("Kernel: cache, ready in %.0f s",
                                (System.nanoTime() - t0) / 1e9);
                    }
                    specialized.put(options, program);
                } finally {
                    compileStatus = null;
                }
            }
            return program;
        }
    }
}
