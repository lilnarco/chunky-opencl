package dev.thatredox.chunkynative.opencl.renderer.kernel;

import static org.jocl.CL.*;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;

public class KernelArgBinder {
    private final cl_kernel kernel;
    private int argIndex;
    private final int[] intValue = new int[1];
    private final float[] floatValue = new float[1];

    public KernelArgBinder(cl_kernel kernel) {
        this.kernel = kernel;
        this.argIndex = 0;
    }

    public void reset() {
        this.argIndex = 0;
    }

    /** Index the next setMem/setInt/setFloat call will bind (for per-dispatch args). */
    public int getArgIndex() {
        return argIndex;
    }

    public void setMem(cl_mem mem) {
        clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(mem));
    }

    public void setInt(int value) {
        intValue[0] = value;
        clSetKernelArg(kernel, argIndex++, Sizeof.cl_int, Pointer.to(intValue));
    }

    public void setFloat(float value) {
        floatValue[0] = value;
        clSetKernelArg(kernel, argIndex++, Sizeof.cl_float, Pointer.to(floatValue));
    }
}
