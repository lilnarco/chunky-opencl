package dev.thatredox.chunkynative.opencl.renderer.kernel;

public class DispatchParams {
    private final int rngSeed;
    private final int bufferSpp;
    private final int sppPerBatch;

    public DispatchParams(int rngSeed, int bufferSpp, int sppPerBatch) {
        this.rngSeed = rngSeed;
        this.bufferSpp = bufferSpp;
        this.sppPerBatch = sppPerBatch;
    }

    public int getRngSeed() {
        return rngSeed;
    }

    public int getBufferSpp() {
        return bufferSpp;
    }

    public int getSppPerBatch() {
        return sppPerBatch;
    }
}
