package dev.thatredox.chunkynative.opencl.renderer;

import se.llbit.log.Log;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

/**
 * Runs the Intel Open Image Denoise CLI (`oidnDenoise`) on the accumulated beauty,
 * albedo and normal passes, like the old Chunky denoiser plugin. The passes are
 * round-tripped through PFM temp files, which OIDN reads natively.
 */
public class OidnDenoiser {
    // Configuration, set from the ChunkyCl UI.
    public static boolean enabled = false;
    public static String binaryPath = "oidnDenoise";

    // Set by the "Denoise now" button; consumed by the renderer at the next merge.
    public static volatile boolean triggerDenoise = false;

    // Last accumulated albedo/normal guides, stashed by the renderer so denoising can
    // also run when the GPU render is not active (paused/stopped).
    public static volatile float[] lastAlbedo = null;
    public static volatile float[] lastNormal = null;

    private static final double MAX_PFM_VALUE = 1.0e6;

    /**
     * Denoise the given beauty buffer using the matching albedo and normal buffers.
     *
     * @return the denoised beauty buffer, or null if denoising failed.
     */
    public static float[] denoiseFinal(float[] beauty, float[] albedo, float[] normal, int width, int height) {
        if (beauty.length != width * height * 3) {
            Log.error("OIDN: buffer size " + beauty.length + " does not match canvas " + width + "x" + height);
            return null;
        }

        // Normalize the HDR beauty to a sane range for OIDN.
        float maxVal = 0.0f;
        for (int i = 0; i < beauty.length; i++) {
            float v = Math.abs(beauty[i]);
            if (v > maxVal) maxVal = v;
        }
        if (!(maxVal > 0.0f) || maxVal > MAX_PFM_VALUE) {
            Log.error("OIDN: invalid beauty buffer range (" + maxVal + ")");
            return null;
        }
        double scale = 1.0 / maxVal;
        Log.warn("OIDN: denoising " + width + "x" + height + " frame (range " + maxVal + ") with \"" + binaryPath + "\"");

        File dir;
        try {
            dir = java.nio.file.Files.createTempDirectory("chunkycl-oidn").toFile();
        } catch (IOException e) {
            Log.error("OIDN: failed to create temp directory", e);
            return null;
        }

        File beautyFile = new File(dir, "beauty.pfm");
        File albedoFile = new File(dir, "albedo.pfm");
        File normalFile = new File(dir, "normal.pfm");
        File outFile = new File(dir, "out.pfm");

        float[] albedoScaled = scaleBuffer(albedo, scale);
        float[] normalMapped = new float[normal.length];
        for (int i = 0; i < normal.length; i++) {
            normalMapped[i] = (float) ((normal[i] + 1.0) * 0.5);
        }

        try {
            writePfm(beautyFile, width, height, scaleBuffer(beauty, scale));
            writePfm(albedoFile, width, height, albedoScaled);
            writePfm(normalFile, width, height, normalMapped);
        } catch (IOException e) {
            Log.error("OIDN: failed to write PFM files", e);
            deleteDir(dir);
            return null;
        }

        try {
            ProcessBuilder pb = new ProcessBuilder(
                    binaryPath,
                    "--alb", albedoFile.getAbsolutePath(),
                    "--nrm", normalFile.getAbsolutePath(),
                    "--hdr", beautyFile.getAbsolutePath(),
                    "-o", outFile.getAbsolutePath()
            );
            pb.redirectErrorStream(true);
            Process process = pb.start();
            byte[] output = process.getInputStream().readAllBytes();
            if (!process.waitFor(60, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                Log.error("OIDN: denoising timed out");
                deleteDir(dir);
                return null;
            }
            if (process.exitValue() != 0) {
                Log.error("OIDN: denoising failed with exit code " + process.exitValue() +
                        "\n" + new String(output, StandardCharsets.UTF_8));
                deleteDir(dir);
                return null;
            }
        } catch (IOException e) {
            Log.error("OIDN: failed to run \"" + binaryPath + "\". Check the binary path in the OpenCL tab.", e);
            deleteDir(dir);
            return null;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            deleteDir(dir);
            return null;
        }

        float[] result;
        try {
            result = readPfm(outFile, width, height);
        } catch (IOException e) {
            Log.error("OIDN: failed to read denoised output", e);
            deleteDir(dir);
            return null;
        }
        deleteDir(dir);
        if (result == null) {
            return null;
        }

        float invScale = (float) (1.0 / scale);
        for (int i = 0; i < result.length; i++) {
            result[i] *= invScale;
        }
        return result;
    }

    private static float[] scaleBuffer(float[] buffer, double scale) {
        float[] out = new float[buffer.length];
        for (int i = 0; i < buffer.length; i++) {
            out[i] = (float) (buffer[i] * scale);
        }
        return out;
    }

    private static void writePfm(File file, int width, int height, float[] data) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(file)) {
            String header = "PF\n" + width + " " + height + "\n-1.0\n";
            fos.write(header.getBytes(StandardCharsets.US_ASCII));
            ByteBuffer buffer = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
            FloatBuffer floats = buffer.asFloatBuffer();
            // PFM rows are bottom-up: the first scanline in the file is the bottom row,
            // so the render buffer (top-down) must be written in reverse row order.
            for (int row = 0; row < height; row++) {
                floats.put(data, (height - 1 - row) * width * 3, width * 3);
            }
            fos.write(buffer.array());
        }
    }

    private static float[] readPfm(File file, int width, int height) throws IOException {
        try (DataInputStream in = new DataInputStream(new FileInputStream(file))) {
            String magic = readAsciiLine(in);
            if (!"PF".equals(magic)) {
                Log.error("OIDN: unexpected output format \"" + magic + "\"");
                return null;
            }
            String[] dims = readAsciiLine(in).trim().split("\\s+");
            if (dims.length != 2) {
                Log.error("OIDN: malformed PFM dimensions");
                return null;
            }
            int w = Integer.parseInt(dims[0]);
            int h = Integer.parseInt(dims[1]);
            double scale = Double.parseDouble(readAsciiLine(in).trim());
            if (w != width || h != height) {
                Log.error("OIDN: output dimensions " + w + "x" + h + " do not match " + width + "x" + height);
                return null;
            }

            float[] data = new float[w * h * 3];
            byte[] raw = in.readNBytes(data.length * 4);
            if (raw.length != data.length * 4) {
                Log.error("OIDN: truncated PFM output");
                return null;
            }
            ByteBuffer buffer = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
            buffer.asFloatBuffer().get(data);

            // PFM rows are bottom-up; flip to match the render buffer.
            float[] flipped = new float[data.length];
            for (int row = 0; row < h; row++) {
                System.arraycopy(data, (h - 1 - row) * w * 3, flipped, row * w * 3, w * 3);
            }
            return flipped;
        } catch (NumberFormatException e) {
            Log.error("OIDN: malformed PFM header", e);
            return null;
        }
    }

    private static String readAsciiLine(DataInputStream in) throws IOException {
        StringBuilder sb = new StringBuilder();
        int c;
        while ((c = in.read()) != -1 && c != '\n' && c != '\r') {
            sb.append((char) c);
        }
        if (c == '\r') {
            in.read();
        }
        return sb.toString();
    }

    /**
     * Round-trip a small buffer through the PFM writer/reader and verify it comes back
     * in the same orientation (catches bottom-up/upside-down regressions).
     */
    public static boolean selfTest() {
        int w = 3;
        int h = 2;
        float[] data = new float[w * h * 3];
        for (int i = 0; i < data.length; i++) {
            data[i] = (i % 7) * 0.1f + 0.01f;
        }
        File file = null;
        try {
            file = File.createTempFile("chunkycl-pfm-test", ".pfm");
            writePfm(file, w, h, data);
            float[] back = readPfm(file, w, h);
            if (back == null) {
                return false;
            }
            for (int i = 0; i < data.length; i++) {
                if (Math.abs(back[i] - data[i]) > 1.0e-5f) {
                    return false;
                }
            }
            return true;
        } catch (IOException e) {
            return false;
        } finally {
            if (file != null) {
                file.delete();
            }
        }
    }

    private static void deleteDir(File dir) {
        File[] files = dir.listFiles();
        if (files != null) {
            for (File f : files) {
                f.delete();
            }
        }
        dir.delete();
    }
}
