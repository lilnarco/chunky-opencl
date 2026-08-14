#ifndef CHUNKYCL_NOISE_H
#define CHUNKYCL_NOISE_H

#include "../opencl.h"

// 3D simplex noise with analytic derivatives, ported from Chunky's SimplexNoise
// (Stefan Gustavson / Keijiro Takahashi), used by the Simplex water shader.
#include "noise_tables.h"

typedef struct {
    float value;
    float ddx;
    float ddy;
} SimplexNoise;

SimplexNoise SimplexNoise_calculate(float x, float y, float z) {
    SimplexNoise n;
    const float F3 = 1.0f / 3.0f;
    const float G3 = 1.0f / 6.0f;

    float s = (x + y + z) * F3;
    float xs = x + s;
    float ys = y + s;
    float zs = z + s;
    int i = (int)floor(xs);
    int j = (int)floor(ys);
    int k = (int)floor(zs);
    float t = (i + j + k) * G3;
    float X0 = i - t;
    float Y0 = j - t;
    float Z0 = k - t;
    float x0 = x - X0;
    float y0 = y - Y0;
    float z0 = z - Z0;

    int i1, j1, k1, i2, j2, k2;
    if (x0 >= y0) {
        if (y0 >= z0) {
            i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
        } else if (x0 >= z0) {
            i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1;
        } else {
            i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1;
        }
    } else {
        if (y0 < z0) {
            i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1;
        } else if (x0 < z0) {
            i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1;
        } else {
            i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
        }
    }

    float x1 = x0 - i1 + G3;
    float y1 = y0 - j1 + G3;
    float z1 = z0 - k1 + G3;
    float x2 = x0 - i2 + 2 * G3;
    float y2 = y0 - j2 + 2 * G3;
    float z2 = z0 - k2 + 2 * G3;
    float x3 = x0 - 1 + 3 * G3;
    float y3 = y0 - 1 + 3 * G3;
    float z3 = z0 - 1 + 3 * G3;

    int ii = i & 0xff;
    int jj = j & 0xff;
    int kk = k & 0xff;

    float n0, n1, n2, n3;
    float gx0, gy0, gz0, gx1, gy1, gz1, gx2, gy2, gz2, gx3, gy3, gz3;
    float t20, t40, t21, t41, t22, t42, t23, t43;

    float t0 = 0.6f - x0 * x0 - y0 * y0 - z0 * z0;
    if (t0 < 0) {
        t40 = t20 = t0 = n0 = gx0 = gy0 = gz0 = 0;
    } else {
        int gi = Simplex_perm[ii + Simplex_perm[jj + Simplex_perm[kk]]];
        gx0 = Simplex_grad3x[gi & 15];
        gy0 = Simplex_grad3y[gi & 15];
        gz0 = Simplex_grad3z[gi & 15];
        t20 = t0 * t0;
        t40 = t20 * t20;
        n0 = t40 * (gx0 * x0 + gy0 * y0 + gz0 * z0);
    }

    float t1 = 0.6f - x1 * x1 - y1 * y1 - z1 * z1;
    if (t1 < 0) {
        t41 = t21 = t1 = n1 = gx1 = gy1 = gz1 = 0;
    } else {
        int gi = Simplex_perm[ii + i1 + Simplex_perm[jj + j1 + Simplex_perm[kk + k1]]];
        gx1 = Simplex_grad3x[gi & 15];
        gy1 = Simplex_grad3y[gi & 15];
        gz1 = Simplex_grad3z[gi & 15];
        t21 = t1 * t1;
        t41 = t21 * t21;
        n1 = t41 * (gx1 * x1 + gy1 * y1 + gz1 * z1);
    }

    float t2 = 0.6f - x2 * x2 - y2 * y2 - z2 * z2;
    if (t2 < 0) {
        t42 = t22 = t2 = n2 = gx2 = gy2 = gz2 = 0;
    } else {
        int gi = Simplex_perm[ii + i2 + Simplex_perm[jj + j2 + Simplex_perm[kk + k2]]];
        gx2 = Simplex_grad3x[gi & 15];
        gy2 = Simplex_grad3y[gi & 15];
        gz2 = Simplex_grad3z[gi & 15];
        t22 = t2 * t2;
        t42 = t22 * t22;
        n2 = t42 * (gx2 * x2 + gy2 * y2 + gz2 * z2);
    }

    float t3 = 0.6f - x3 * x3 - y3 * y3 - z3 * z3;
    if (t3 < 0) {
        t43 = t23 = t3 = n3 = gx3 = gy3 = gz3 = 0;
    } else {
        int gi = Simplex_perm[ii + 1 + Simplex_perm[jj + 1 + Simplex_perm[kk + 1]]];
        gx3 = Simplex_grad3x[gi & 15];
        gy3 = Simplex_grad3y[gi & 15];
        gz3 = Simplex_grad3z[gi & 15];
        t23 = t3 * t3;
        t43 = t23 * t23;
        n3 = t43 * (gx3 * x3 + gy3 * y3 + gz3 * z3);
    }

    n.value = 28 * (n0 + n1 + n2 + n3);

    float temp0 = t20 * t0 * (gx0 * x0 + gy0 * y0 + gz0 * z0);
    n.ddx = temp0 * x0;
    n.ddy = temp0 * y0;
    float temp1 = t21 * t1 * (gx1 * x1 + gy1 * y1 + gz1 * z1);
    n.ddx += temp1 * x1;
    n.ddy += temp1 * y1;
    float temp2 = t22 * t2 * (gx2 * x2 + gy2 * y2 + gz2 * z2);
    n.ddx += temp2 * x2;
    n.ddy += temp2 * y2;
    float temp3 = t23 * t3 * (gx3 * x3 + gy3 * y3 + gz3 * z3);
    n.ddx += temp3 * x3;
    n.ddy += temp3 * y3;

    n.ddx = -8 * n.ddx + (t40 * gx0 + t41 * gx1 + t42 * gx2 + t43 * gx3);
    n.ddy = -8 * n.ddy + (t40 * gy0 + t41 * gy1 + t42 * gy2 + t43 * gy3);
    n.ddx *= 28;
    n.ddy *= 28;
    return n;
}

// Fractal wave normal for the Simplex water shader (CPU-identical constants).
float3 SimplexWaterNormal(float x, float z, float time) {
    float frequency = 0.4f;
    float amplitude = 0.025f;
    float ddx = 0.0f;
    float ddz = 0.0f;
    for (int i = 0; i < 4; i++) {
        SimplexNoise noise = SimplexNoise_calculate(x * frequency, z * frequency, time);
        float ddxNext = ddx - amplitude * noise.ddx;
        float ddzNext = ddz - amplitude * noise.ddy;
        if (isnan(ddxNext + ddzNext)) {
            break;
        }
        ddx = ddxNext;
        ddz = ddzNext;
        frequency *= 2.0f;
        amplitude *= 0.5f;
    }
    return normalize((float3)(-ddx, 1.0f, -ddz));
}

#endif
