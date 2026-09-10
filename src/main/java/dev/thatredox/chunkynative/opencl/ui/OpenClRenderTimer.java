package dev.thatredox.chunkynative.opencl.ui;

public final class OpenClRenderTimer {
    private static volatile boolean running = false;
    private static volatile long startNanos = 0L;
    private static volatile long lastElapsedNanos = 0L;
    // Lap timer: device-side kernel-build time inside the current render, so the
    // tab can show render vs compile separately.
    private static volatile long compileNanos = 0L;

    private OpenClRenderTimer() {}

    public static void start() {
        running = true;
        startNanos = System.nanoTime();
        lastElapsedNanos = 0L;
        compileNanos = 0L;
    }

    public static void stop() {
        if (running) {
            lastElapsedNanos = System.nanoTime() - startNanos;
            running = false;
        }
    }

    public static boolean isRunning() {
        return running;
    }

    public static long getElapsedMillis() {
        long elapsedNanos = running ? (System.nanoTime() - startNanos) : lastElapsedNanos;
        return elapsedNanos / 1_000_000L;
    }

    public static void addCompileNanos(long nanos) {
        compileNanos += nanos;
    }

    public static long getCompileMillis() {
        return compileNanos / 1_000_000L;
    }
}
