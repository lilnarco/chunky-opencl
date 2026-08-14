#ifndef CHUNKYCLPLUGIN_WAVEFRONT_H
#define CHUNKYCLPLUGIN_WAVEFRONT_H

#include "../opencl.h"

#define RAY_INDIRECT 0b01
#define RAY_PREVIEW  0b10
#define RAY_OCCLUDER 0b100

// Kernel operation counters (profiling). Order must match the host-side readback.
#define PROFILE_TOTAL_RAYS 0
#define PROFILE_TOTAL_HITS 1
#define PROFILE_OCTREE_STEPS 2
#define PROFILE_BVH_NODE_TESTS 3
#define PROFILE_DIFFUSE_BOUNCES 4
#define PROFILE_SPECULAR_BOUNCES 5
#define PROFILE_REFRACTION_BOUNCES 6
#define PROFILE_EMITTER_GRID_LOOKUPS 7
#define PROFILE_EMITTER_SAMPLES 8
#define PROFILE_EMITTER_RAYS 9
#define PROFILE_EMITTER_RAY_STEPS 10
#define PROFILE_SUN_RAYS 11
#define PROFILE_SUN_RAY_STEPS 12
#define PROFILE_OCCLUDER_FAST_PATH_HITS 13
#define PROFILE_WAVE_NOISE_CALLS 14
#define PROFILE_CLOUD_STEPS 15
#define PROFILE_WATER_PLANE_TESTS 16
#define PROFILE_COUNT 17

static inline void Profile_inc(bool enabled, __global int* counters, int slot) {
    if (enabled) {
        atomic_inc(&counters[slot]);
    }
}

typedef struct {
    float3 origin;
    float3 direction;
    int prevMaterial;
    int currentMaterial;
    int prevBlock;
    int currentBlock;
    int flags;
} Ray;

typedef struct {
    float distance;
    int material;
    int block;

    float3 normal;
    float2 texCoord;
} IntersectionRecord;

IntersectionRecord IntersectionRecord_new() {
    IntersectionRecord record;
    record.distance = HUGE_VALF;
    record.material = 0;
    record.block = 0;
    record.normal = (float3) (0, 1, 0);
    return record;
}

#endif
