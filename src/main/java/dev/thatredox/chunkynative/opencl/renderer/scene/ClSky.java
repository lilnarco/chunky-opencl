package dev.thatredox.chunkynative.opencl.renderer.scene;


import static org.jocl.CL.*;

import dev.thatredox.chunkynative.opencl.context.ClContext;
import dev.thatredox.chunkynative.opencl.util.ClMemory;
import org.apache.commons.math3.util.FastMath;
import org.jocl.*;

import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.chunky.renderer.scene.sky.Sky;
import se.llbit.chunky.resources.Texture;
import se.llbit.math.Ray;

import java.lang.reflect.Field;
import java.util.stream.IntStream;

public class ClSky implements AutoCloseable {
    /** Maximum sky bake width. 8192x4096 (CL_FLOAT RGBA) = 512 MB VRAM. */
    private static final int MAX_SKY_RESOLUTION = 8192;
    /** Fraction of the device's global memory budgeted for the sky texture. */
    private static final double SKY_MEMORY_BUDGET = 0.25;
    /** Bytes per texel for a CL_FLOAT RGBA image. */
    private static final int BYTES_PER_TEXEL = 16;

    public final ClMemory skyTexture;
    private final ClContext context;

    public ClSky(Scene scene, ClContext context) {
        this.context = context;
        int textureResolution = getBakeResolution(scene, context);
        int height = textureResolution / 2;

        // Float (HDR) image so bright HDRI values survive the bake. Storing HDR values in
        // an 8-bit texture wraps them modulo 256 into garbage colors.
        cl_image_format fmt = new cl_image_format();
        fmt.image_channel_data_type = CL_FLOAT;
        fmt.image_channel_order = CL_RGBA;

        cl_image_desc desc = new cl_image_desc();
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = textureResolution;
        desc.image_height = height;

        float[] texture = new float[textureResolution * height * 4];
        IntStream.range(0, height).parallel().forEach(j -> {
            Ray ray = new Ray();
            for (int i = 0; i < textureResolution; i++) {
                int offset = 4 * (j * textureResolution + i);

                double theta = ((double) i / textureResolution) * 2 * FastMath.PI;
                double phi = ((double) j / height) * FastMath.PI - FastMath.PI / 2;
                double r = FastMath.cos(phi);
                ray.d.set(FastMath.cos(theta) * r, FastMath.sin(phi), FastMath.sin(theta) * r);

                scene.sky().getSkyColor(ray, false);
                texture[offset + 0] = (float) ray.color.x;
                texture[offset + 1] = (float) ray.color.y;
                texture[offset + 2] = (float) ray.color.z;
                texture[offset + 3] = 1.0f;
            }
        });

        this.skyTexture = new ClMemory(clCreateImage(context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                fmt, desc, Pointer.to(texture), null));
    }

    /**
     * Pick the sky bake resolution: the smallest of the skymap's own width, the device's
     * maximum image width, a memory budget derived from the device's global memory, and
     * the hard 8k cap. Falls back to 4096 for procedural skies.
     */
    private static int getBakeResolution(Scene scene, ClContext context) {
        int skymapWidth = getSkymapWidth(scene);
        long maxImageWidth = context.device.getDeviceLongs(CL_DEVICE_IMAGE2D_MAX_WIDTH, 1)[0];
        long globalMem = context.device.getDeviceLongs(CL_DEVICE_GLOBAL_MEM_SIZE, 1)[0];
        long memoryBudget = (long) Math.sqrt((globalMem * SKY_MEMORY_BUDGET) / BYTES_PER_TEXEL);
        long resolution = Math.min(MAX_SKY_RESOLUTION, Math.min(skymapWidth, Math.min(maxImageWidth, memoryBudget)));
        resolution &= ~15L;
        if (resolution < 256) {
            resolution = 256;
        }
        return (int) resolution;
    }

    private static int getSkymapWidth(Scene scene) {
        try {
            Sky sky = scene.sky();
            Field skymapField = sky.getClass().getDeclaredField("skymap");
            skymapField.setAccessible(true);
            Texture skymap = (Texture) skymapField.get(sky);
            int width = skymap.getWidth();
            return width > 0 ? width : 4096;
        } catch (NoSuchFieldException | IllegalAccessException e) {
            return 4096;
        }
    }

    @Override
    public void close() {
        skyTexture.close();
    }
}
