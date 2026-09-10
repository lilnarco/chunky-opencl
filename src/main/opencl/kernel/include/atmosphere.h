#ifndef CHUNKYCL_ATMOSPHERE_H
#define CHUNKYCL_ATMOSPHERE_H

#include "../opencl.h"
#include "rt.h"
#include "constants.h"
#include "utils.h"

// Reserved material palette index for the opaque white cloud material (air = 0).
#define CLOUD_MATERIAL 1

#define FOG_EXTINCTION_FACTOR 0.04f
#define CLOUD_LAYER_HEIGHT 5.0f
// CPU Fog.FOG_LIMIT: the distance used for the sky-fog y-span.
#define FOG_LIMIT 30000.0f

typedef struct {
    int fogMode;              // 0 = NONE, 1 = UNIFORM, 2 = LAYERED
    float3 fogColor;
    float uniformDensity;
    float skyFogDensity;
    bool fastFog;
    bool cloudsEnabled;
    float cloudSize;
    float3 cloudOffset;
    int3 origin;              // octree origin, used to convert to world coordinates
    __global const int* cloudData;
    float waterVisibility;    // underwater light attenuation distance
    bool waterPlaneEnabled;
    float waterPlaneY;        // octree coordinates
    int waterShader;          // 0 = still, 1 = simplex, 2 = legacy (treated as simplex)
    float animationTime;
    int waterMaterial;        // material palette index of water
    float waterOpacity;       // water surface alpha (CPU parity, default 0.42)
    bool hasWater;            // water octree non-empty (host-computed) - gates the
                              // always-run water octree march / medium lookup
    int fogLayerCount;        // layered fog: number of logistic layers
    __global const float* fogLayers; // layered fog: (yWithOrigin, breadthInv, density) x count
    bool profile;
    __global int* profileCounters;
} Atmosphere;

Atmosphere Atmosphere_new(__global const float* settings, __global const int* cloudData) {
    Atmosphere a;
    a.fogMode = (int)settings[0];
    a.fogColor = (float3)(settings[1], settings[2], settings[3]);
    a.uniformDensity = settings[4];
    a.skyFogDensity = settings[5];
    a.fastFog = settings[6] > 0.5f;
    a.cloudsEnabled = settings[7] > 0.5f;
    a.cloudSize = settings[8];
    a.cloudOffset = (float3)(settings[9], settings[10], settings[11]);
    a.origin = (int3)((int)settings[12], (int)settings[13], (int)settings[14]);
    a.cloudData = cloudData;
    a.waterVisibility = settings[15];
    a.waterPlaneEnabled = settings[16] > 0.5f;
    a.waterPlaneY = settings[17] - (float)a.origin.y;
    a.waterShader = (int)settings[18];
    a.animationTime = settings[19];
    a.waterMaterial = as_int(settings[20]);
    a.waterOpacity = settings[21];
    a.hasWater = settings[22] > 0.5f;
    a.fogLayerCount = (int)settings[23];
    a.fogLayers = settings + 24;
    a.profile = false;
    a.profileCounters = (__global int*)0;
    return a;
}

Atmosphere Atmosphere_empty() {
    Atmosphere a;
    a.fogMode = 0;
    a.fogColor = (float3)(0.0f, 0.0f, 0.0f);
    a.uniformDensity = 0.0f;
    a.skyFogDensity = 1.0f;
    a.fastFog = true;
    a.cloudsEnabled = false;
    a.cloudSize = 1.0f;
    a.cloudOffset = (float3)(0.0f, 0.0f, 0.0f);
    a.origin = (int3)(0, 0, 0);
    a.cloudData = (__global const int*)0;
    a.waterVisibility = 1.0f;
    a.waterPlaneEnabled = false;
    a.waterPlaneY = 0.0f;
    a.waterShader = 0;
    a.animationTime = 0.0f;
    a.waterMaterial = 0;
    a.waterOpacity = 1.0f;
    // The preview kernel has no settings buffer; conservatively keep the water march
    // enabled so water renders in the preview.
    a.hasWater = true;
    a.fogLayerCount = 0;
    a.fogLayers = (__global const float*)0;
    a.profile = false;
    a.profileCounters = (__global int*)0;
    return a;
}

// CPU Fog.clampDy: prevents numerical errors / division by 0 when dy is close to 0.
// Only layered fog (FOG_MODE == 2) uses these; guarding the definitions keeps the
// -Werror build clean when the call sites above are specialized away.
#if !defined(FOG_MODE) || FOG_MODE == 2
static inline float Fog_clampDy(float dy) {
    const float epsilon = 0.00001f;
    if (dy > 0.0f) {
        if (dy < epsilon) {
            return epsilon;
        }
    } else if (dy > -epsilon) {
        return -epsilon;
    }
    return dy;
}

// CPU Fog.addLayeredFog: logistic-CDF extinction over the y-span [y1, y2], with the
// fully-sunlit inscatter approximation (scatterLight = 1), matching the uniform fog's
// established simplification.
static inline float3 Fog_applyLayered(Atmosphere self, float dy, float y1, float y2, float3 color) {
    float total = 0.0f;
    for (int i = 0; i < self.fogLayerCount; i++) {
        float yWithOrigin = self.fogLayers[i * 3];
        float breadthInv = self.fogLayers[i * 3 + 1];
        float density = self.fogLayers[i * 3 + 2];
        // Stage 2: native_exp — feeds 8-bit output via extinction/inscatter mix,
        // so ~1e-3 error is invisible. Same below for the extinction.
        float cdf1 = 1.0f / (1.0f + native_exp((yWithOrigin - y1) * breadthInv));
        float cdf2 = 1.0f / (1.0f + native_exp((yWithOrigin - y2) * breadthInv));
        total += density * (cdf1 - cdf2);
    }
    float extinction = native_exp(total / Fog_clampDy(dy));
    float inscatter = 1.0f - extinction;
    color.x = color.x * extinction + inscatter * self.fogColor.x;
    color.y = color.y * extinction + inscatter * self.fogColor.y;
    color.z = color.z * extinction + inscatter * self.fogColor.z;
    return color;
}
#endif

// Cloud bitset: 32x32 longs covering a periodic 256x256 world grid, one bit per block.
int Cloud_get(__global const int* data, int x, int y) {
    x = ((x % 256) + 256) % 256;
    y = ((y % 256) + 256) % 256;
    int tilex = x / 8;
    int tiley = y / 8;
    int sub = (y & 7) * 8 + (x & 7);
    int idx = (tiley * 32 + tilex) * 2;
    unsigned int lo = (unsigned int)data[idx];
    unsigned int hi = (unsigned int)data[idx + 1];
    ulong v = ((ulong)hi << 32) | (ulong)lo;
    return (int)((v >> sub) & 1UL);
}

bool Cloud_inCloud(Atmosphere self, float x, float z) {
    return Cloud_get(self.cloudData, (int)floor(x), (int)floor(z)) == 1;
}

void Cloud_sample(MaterialSample* sample) {
    sample->color = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
    sample->emittance = 0.0f;
    sample->specular = 0.0f;
    sample->metalness = 0.0f;
    sample->roughness = 0.0f;
}

// Port of Sky.cloudIntersection: 2D DDA through the periodic cloud grid between
// [cloudOffset.y, cloudOffset.y + 5] in world coordinates. Handles both entering
// (target = 1) and exiting (target = 0) the cloud layer like the CPU renderer.
bool Cloud_intersect(Atmosphere self, Ray ray, IntersectionRecord* record, MaterialSample* sample) {
    float ox = ray.origin.x + self.origin.x;
    float oy = ray.origin.y + self.origin.y;
    float oz = ray.origin.z + self.origin.z;
    float offsetX = self.cloudOffset.x;
    float offsetY = self.cloudOffset.y;
    float offsetZ = self.cloudOffset.z;
    float invSize = 1.0f / self.cloudSize;
    float cloudTop = offsetY + CLOUD_LAYER_HEIGHT;
    int target = 1;
    float tOffset = 0.0f;

    if (oy < offsetY || oy > cloudTop) {
        if (fabs(ray.direction.y) < EPS) {
            // Ignore almost horizontal intersections (large/negative t_offset).
            return false;
        }
        if (ray.direction.y > 0.0f) {
            tOffset = (offsetY - oy) / ray.direction.y;
        } else {
            tOffset = (cloudTop - oy) / ray.direction.y;
        }
        if (tOffset < 0.0f) {
            return false;
        }
        // Ray is entering cloud.
        if (Cloud_inCloud(self, (ray.direction.x * tOffset + ox) * invSize + offsetX,
                (ray.direction.z * tOffset + oz) * invSize + offsetZ)) {
            record->distance = tOffset;
            record->normal = (float3)(0.0f, -sign(ray.direction.y), 0.0f);
            record->material = CLOUD_MATERIAL;
            record->block = 0;
            Cloud_sample(sample);
            return true;
        }
    } else if (Cloud_inCloud(self, ox * invSize + offsetX, oz * invSize + offsetZ)) {
        target = 0;
    }

    float tExit;
    if (ray.direction.y > 0.0f) {
        tExit = (cloudTop - oy) / ray.direction.y - tOffset;
    } else {
        tExit = (offsetY - oy) / ray.direction.y - tOffset;
    }
    if (record->distance < tExit) {
        tExit = record->distance;
    }

    float x0 = (ox + ray.direction.x * tOffset) * invSize + offsetX;
    float z0 = (oz + ray.direction.z * tOffset) * invSize + offsetZ;
    float xp = x0;
    float zp = z0;
    int ix = (int)floor(x0);
    int iz = (int)floor(z0);
    int xmod = (int)sign(ray.direction.x);
    int zmod = (int)sign(ray.direction.z);
    int xo = (1 + xmod) / 2;
    int zo = (1 + zmod) / 2;
    float dx = fabs(ray.direction.x) * invSize;
    float dz = fabs(ray.direction.z) * invSize;
    float t = 0.0f;
    int i = 0;
    int nx = 0;
    int nz = 0;

    if (dx > dz) {
        float m = dz / dx;
        float xrem = (float)xmod * (ix + xo - xp);
        float zlimit = xrem * m;
        while (t < tExit) {
            Profile_inc(self.profile, self.profileCounters, PROFILE_CLOUD_STEPS);
            float zrem = (float)zmod * (iz + zo - zp);
            if (zrem < zlimit) {
                iz += zmod;
                if (Cloud_get(self.cloudData, ix, iz) == target) {
                    t = i / dx + zrem / dz;
                    nx = 0;
                    nz = -zmod;
                    break;
                }
                ix += xmod;
                if (Cloud_get(self.cloudData, ix, iz) == target) {
                    t = (i + xrem) / dx;
                    nx = -xmod;
                    nz = 0;
                    break;
                }
            } else {
                ix += xmod;
                if (Cloud_get(self.cloudData, ix, iz) == target) {
                    t = (i + xrem) / dx;
                    nx = -xmod;
                    nz = 0;
                    break;
                }
                if (zrem <= m) {
                    iz += zmod;
                    if (Cloud_get(self.cloudData, ix, iz) == target) {
                        t = i / dx + zrem / dz;
                        nx = 0;
                        nz = -zmod;
                        break;
                    }
                }
            }
            t = i / dx;
            i += 1;
            zp = z0 + (float)zmod * i * m;
        }
    } else {
        float m = dx / dz;
        float zrem = (float)zmod * (iz + zo - zp);
        float xlimit = zrem * m;
        while (t < tExit) {
            Profile_inc(self.profile, self.profileCounters, PROFILE_CLOUD_STEPS);
            float xrem = (float)xmod * (ix + xo - xp);
            if (xrem < xlimit) {
                ix += xmod;
                if (Cloud_get(self.cloudData, ix, iz) == target) {
                    t = i / dz + xrem / dx;
                    nx = -xmod;
                    nz = 0;
                    break;
                }
                iz += zmod;
                if (Cloud_get(self.cloudData, ix, iz) == target) {
                    t = (i + zrem) / dz;
                    nx = 0;
                    nz = -zmod;
                    break;
                }
            } else {
                iz += zmod;
                if (Cloud_get(self.cloudData, ix, iz) == target) {
                    t = (i + zrem) / dz;
                    nx = 0;
                    nz = -zmod;
                    break;
                }
                if (xrem <= m) {
                    ix += xmod;
                    if (Cloud_get(self.cloudData, ix, iz) == target) {
                        t = i / dz + xrem / dx;
                        nx = -xmod;
                        nz = 0;
                        break;
                    }
                }
            }
            t = i / dz;
            i += 1;
            xp = x0 + (float)xmod * i * m;
        }
    }

    int ny = 0;
    if (target == 1) {
        if (t > tExit) {
            return false;
        }
        if (nx == 0 && ny == 0 && nz == 0) {
            return false;
        }
        record->distance = t + tOffset;
        record->normal = (float3)((float)nx, 0.0f, (float)nz);
        record->material = CLOUD_MATERIAL;
        record->block = 0;
        Cloud_sample(sample);
        return true;
    } else {
        if (t > tExit) {
            nx = 0;
            ny = (int)sign(ray.direction.y);
            nz = 0;
            t = tExit;
        } else {
            nx = -nx;
            nz = -nz;
        }
        if (nx == 0 && ny == 0 && nz == 0) {
            return false;
        }
        record->distance = t + tOffset;
        record->normal = (float3)((float)nx, (float)ny, (float)nz);
        record->material = 0;
        record->block = 0;
        Cloud_sample(sample);
        return true;
    }
}

#endif
