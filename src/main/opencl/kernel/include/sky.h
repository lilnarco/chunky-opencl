#ifndef CHUNKYCLPLUGIN_SKY_H
#define CHUNKYCLPLUGIN_SKY_H

#include "../opencl.h"
#include "rt.h"
#include "textureAtlas.h"
#include "random.h"
#include "material.h"

typedef struct {
    int flags;
    int textureSize;
    int texture;
    float intensity;
    float luminosity;
    float luminosityPdf;
    float3 su;
    float3 sv;
    float3 sw;
    float4 color;
} Sun;

Sun Sun_new(__global const int* data) {
    Sun sun;
    sun.flags = data[0];
    sun.textureSize = data[1];
    sun.texture = data[2];
    sun.intensity = as_float(data[3]);
    sun.luminosity = as_float(data[6]);
    sun.luminosityPdf = as_float(data[8]);
    sun.color = colorFromArgb(data[7]);
    
    float phi = as_float(data[4]);
    float theta = as_float(data[5]);
    float r = fabs(cos(phi));
    
    sun.sw = (float3) (cos(theta) * r, sin(phi), sin(theta) * r);
    if (fabs(sun.sw.x) > 0.1f) {
        sun.su = (float3) (0, 1, 0);
    } else {
        sun.su = (float3) (1, 0, 0);
    }
    sun.sv = normalize(cross(sun.sw, sun.su));
    sun.su = cross(sun.sv, sun.sw);
    
    return sun;
}

bool Sun_intersect(Sun self, image2d_array_t atlas, Ray ray, MaterialSample* sample) {
    float3 direction = ray.direction;

    if (!(self.flags & 1) || dot(direction, self.sw) < 0.5f) {
        return false;
    }

    float radius = 0.03;

    float width = radius * 4;
    float width2 = width * 2;
    float a = M_PI_2_F - acos(dot(direction, self.su)) + width;
    if (a >= 0 && a < width2) {
        float b = M_PI_2_F - acos(dot(direction, self.sv)) + width;
        if (b >= 0 && b < width2) {
            if (ray.flags & RAY_INDIRECT) {
                float4 color = self.color;
                color *= self.luminosity;
                sample->color += color;
            } else {
                float4 color = Atlas_read_uv(a / width2, b / width2,
                                             self.texture, self.textureSize, atlas);
                color *= self.intensity;
                sample->color += color;
            }
            return true;
        }
    }

    return false;
}

bool Sun_sampleDirection(Sun self, Ray* ray, Random random) {
    // Sun illumination is independent of the "draw sun" toggle (which only controls the
    // disk sprite), matching the CPU renderer. A sun below the horizon produces no light
    // anyway because the shadow ray hits the ground.
    float radius_cos = cos(0.03f);

    float x1 = Random_nextFloat(random);
    float x2 = Random_nextFloat(random);

    float cos_a = 1 - x1 + x1 * radius_cos;
    float sin_a = sqrt(1 - cos_a * cos_a);
    float phi = 2 * M_PI_F * x2;

    float3 u = self.su * (cos(phi) * sin_a);
    float3 v = self.sv * (sin(phi) * sin_a);
    float3 w = self.sw * cos_a;

    ray->direction = u + v + w;
    ray->direction = normalize(ray->direction);

    return true;
}

float3 Sun_emittance(Sun self) {
    return self.color.xyz * pow(self.intensity, 2.2f);
}

// The sky texture is a float (HDR) image; OpenCL 1.2 only guarantees nearest-neighbor
// filtering for float formats, so the sky is baked at a high resolution instead.
const sampler_t skySampler = CLK_NORMALIZED_COORDS_TRUE  | CLK_ADDRESS_MIRRORED_REPEAT | CLK_FILTER_NEAREST;

float3 Sky_sampleDirection(image2d_t skyTexture, float3 direction) {
    float theta = atan2(direction.z, direction.x);
    theta /= M_PI_F * 2;
    theta = fmod(fmod(theta, 1) + 1, 1);
    float phi = (asin(clamp(direction.y, -1.0f, 1.0f)) + M_PI_2_F) * M_1_PI_F;

    return read_imagef(skyTexture, skySampler, (float2) (theta, phi)).xyz;
}

void Sky_intersect(image2d_t skyTexture, Ray ray, MaterialSample* sample) {
    sample->color.xyz = Sky_sampleDirection(skyTexture, ray.direction);
    sample->color.w = 1.0;
    sample->emittance = 1.0;
}

#endif
