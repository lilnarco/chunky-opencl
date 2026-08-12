#ifndef CHUNKYCLPLUGIN_OCTREE_H
#define CHUNKYCLPLUGIN_OCTREE_H

#include "../opencl.h"
#include "rt.h"
#include "constants.h"
#include "primitives.h"
#include "block.h"
#include "utils.h"

float Ray_dynamicOffset(float maxCoord) {
    if (maxCoord <= 0.0f) {
        return OFFSET;
    }
    int exp = ilogb(maxCoord);
    float spacing = ldexp(1.0f, exp - 23);
    float dyn = spacing * 2.0f;
    return dyn < OFFSET ? OFFSET : dyn;
}

typedef struct {
    __global const int* treeData;
    AABB bounds;
    int depth;
    int virtualDepth;
} Octree;

Octree Octree_create(__global const int* treeData, int depth, int virtualDepth) {
    Octree octree;
    octree.treeData = treeData;
    octree.depth = depth;
    // virtualDepth 僅用於計算動態 Offset 以保證大座標下的精度
    octree.virtualDepth = max(depth, virtualDepth);
    // 重要：碰撞邊界必須使用真實深度 (depth)，
    // 這樣 AABB_quick_intersect 才能將平行射線正確推送到實體方塊區域，解決 Y-Clip 精度遺失問題。
    octree.bounds = AABB_new(0, 1<<depth, 0, 1<<depth, 0, 1<<depth);
    return octree;
}

int Octree_get(Octree* self, int x, int y, int z) {
    int3 bp = (int3) (x, y, z);

    // 1. 檢查是否在實體數據範圍內 (depth)
    int3 rlv = bp >> self->depth;
    if ((rlv.x != 0) | (rlv.y != 0) | (rlv.z != 0))
        return 0;

    int level = self->depth;
    int data = self->treeData[0];
    while (data > 0) {
        level--;
        int3 lv = 1 & (bp >> level);
        data = self->treeData[data + ((lv.x << 2) | (lv.y << 1) | lv.z)];
    }
    return -data;
}

bool Octree_octreeIntersect(Octree self, image2d_array_t atlas, BlockPalette palette, MaterialPalette materialPalette, BiomeColors biome, int drawDepth, bool emittersEnabled, Ray ray, IntersectionRecord* record, MaterialSample* sample);

#endif
