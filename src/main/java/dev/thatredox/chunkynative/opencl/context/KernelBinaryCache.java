package dev.thatredox.chunkynative.opencl.context;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_device_id;
import org.jocl.cl_program;
import se.llbit.log.Log;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

import static org.jocl.CL.*;

/**
 * On-disk cache for linked OpenCL program binaries (Stage 1b). The
 * specialized JIT compile can take ~1 min on first render; the binary is
 * deterministic for (options, device, driver, plugin build), so later sessions
 * load it in ~1 s.
 *
 * <p>Every failure — missing dir, stale driver, corrupt file, unsupported call —
 * falls back to the source compile path, so the cache can never brick
 * rendering. Disabled under {@code chunkyClHotReload} (dev sources change
 * without jar rebuilds, which would poison the key).
 */
final class KernelBinaryCache {
    private KernelBinaryCache() {
    }

    /** Directory for cached binaries, portable via {@code chunky.home}. */
    static File cacheDir() {
        String home = System.getProperty("chunky.home");
        File base = home != null ? new File(home) : new File(System.getProperty("java.io.tmpdir"), "chunkycl");
        File dir = new File(base, "kernel-cache");
        if (!dir.isDirectory() && !dir.mkdirs()) {
            return null;
        }
        return dir;
    }

    /** Cache key: options + device + driver + plugin build. */
    static String keyFor(ClContext ctx, String options) {
        return options + "\n" + deviceString(ctx.deviceArray[0], CL_DEVICE_NAME)
                + "\n" + deviceString(ctx.deviceArray[0], CL_DRIVER_VERSION)
                + "\n" + codeVersion();
    }

    static String fileNameFor(String key) {
        try {
            MessageDigest sha = MessageDigest.getInstance("SHA-256");
            byte[] digest = sha.digest(key.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder("kernel-");
            for (byte b : digest) {
                sb.append(String.format("%02x", b));
            }
            return sb.append(".bin").toString();
        } catch (NoSuchAlgorithmException e) {
            return "kernel-" + key.hashCode() + ".bin";
        }
    }

    /** Plugin build id: SHA-256 of the jar bytes (stable across copies of the
     * same build, unlike mtime — re-copying the jar must not invalidate). */
    static String codeVersion() {
        try {
            java.net.URL url = KernelBinaryCache.class.getProtectionDomain().getCodeSource().getLocation();
            File f = new File(url.toURI());
            if (f.isFile()) {
                byte[] jar = Files.readAllBytes(f.toPath());
                MessageDigest sha = MessageDigest.getInstance("SHA-256");
                byte[] digest = sha.digest(jar);
                StringBuilder sb = new StringBuilder("jar-");
                for (int i = 0; i < 8; i++) {
                    sb.append(String.format("%02x", digest[i]));
                }
                return sb.toString();
            }
        } catch (Exception e) {
            // Fall through to unknown; the source-compile fallback stays correct.
        }
        return "unknown";
    }

    static String deviceString(cl_device_id device, int param) {
        try {
            long[] size = new long[1];
            int code = clGetDeviceInfo(device, param, 0, null, size);
            if (code != CL_SUCCESS || size[0] <= 1) {
                return "unknown";
            }
            byte[] buf = new byte[(int) size[0]];
            code = clGetDeviceInfo(device, param, buf.length, Pointer.to(buf), null);
            if (code != CL_SUCCESS) {
                return "unknown";
            }
            return new String(buf, 0, buf.length - 1);
        } catch (Exception e) {
            return "unknown";
        }
    }

    /** Load a cached binary, or null (any failure means "compile from source"). */
    static cl_program tryLoad(ClContext ctx, String options) {
        if (KernelLoader.canHotReload()) {
            return null;
        }
        try {
            File dir = cacheDir();
            if (dir == null) {
                return null;
            }
            File file = new File(dir, fileNameFor(keyFor(ctx, options)));
            if (!file.isFile()) {
                return null;
            }
            byte[] binary = Files.readAllBytes(file.toPath());
            if (binary.length == 0) {
                return null;
            }
            int[] err = new int[1];
            int[] status = new int[1];
            cl_program program = clCreateProgramWithBinary(ctx.context, 1, ctx.deviceArray,
                    new long[] { binary.length }, new byte[][] { binary }, status, err);
            if (err[0] != CL_SUCCESS || status[0] != CL_SUCCESS || program == null) {
                return null;
            }
            int code = clBuildProgram(program, 1, ctx.deviceArray, "", null, null);
            if (code != CL_SUCCESS) {
                clReleaseProgram(program);
                return null;
            }
            Log.info("ChunkyCL: reused cached kernel binary (" + file.getName() + ").");
            return program;
        } catch (Exception e) {
            Log.warn("ChunkyCL: kernel binary cache miss (" + e.getMessage() + "), compiling from source.");
            return null;
        }
    }

    /** Store a freshly linked binary; failures are logged and ignored. */
    static void store(ClContext ctx, cl_program program, String options) {
        if (KernelLoader.canHotReload()) {
            return;
        }
        try {
            File dir = cacheDir();
            if (dir == null) {
                return;
            }
            int[] num = new int[1];
            int code = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, Sizeof.cl_uint,
                    Pointer.to(num), new long[1]);
            if (code != CL_SUCCESS || num[0] != 1) {
                return;
            }
            long[] sizes = new long[1];
            code = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, Sizeof.cl_ulong,
                    Pointer.to(sizes), new long[1]);
            if (code != CL_SUCCESS || sizes[0] <= 0 || sizes[0] > (1 << 30)) {
                return;
            }
            byte[] binary = new byte[(int) sizes[0]];
            Pointer binaryPtr = Pointer.to(binary);
            // Single device: param value is one pointer (8 bytes on 64-bit Chunky).
            code = clGetProgramInfo(program, CL_PROGRAM_BINARIES, 8L, Pointer.to(binaryPtr), new long[1]);
            if (code != CL_SUCCESS) {
                return;
            }
            File file = new File(dir, fileNameFor(keyFor(ctx, options)));
            Files.write(file.toPath(), binary);
            Log.info("ChunkyCL: cached kernel binary (" + file.getName() + ").");
        } catch (Exception e) {
            Log.warn("ChunkyCL: could not cache kernel binary (" + e.getMessage() + ").");
        }
    }
}
