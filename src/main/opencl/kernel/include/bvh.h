#ifndef CHUNKYCL_PLUGIN_BVH
#define CHUNKYCL_PLUGIN_BVH

#include "../opencl.h"
#include "material.h"
#include "primitives.h"

typedef struct {
    __global const int* bvh;
    __global const int* trigs;
    bool profile;
    __global int* profileCounters;
} Bvh;

static inline Bvh Bvh_profile(Bvh bvh, bool enabled, __global int* counters) {
    bvh.profile = enabled;
    bvh.profileCounters = counters;
    return bvh;
}

Bvh Bvh_new(__global const int* bvh, __global const int* trigs) {
    Bvh b;
    b.bvh = bvh;
    b.trigs = trigs;
    b.profile = false;
    b.profileCounters = (__global int*)0;
    return b;
}

bool Bvh_intersect(Bvh self, image2d_array_t atlas, MaterialPalette palette, BiomeColors biome, Ray ray, IntersectionRecord* record, MaterialSample* sample);

#endif
