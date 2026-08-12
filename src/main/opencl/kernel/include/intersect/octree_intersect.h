#include "octree.h"

bool Octree_octreeIntersect(Octree self, image2d_array_t atlas, BlockPalette palette, MaterialPalette materialPalette, BiomeColors biome, int drawDepth, bool emittersEnabled, Ray ray, IntersectionRecord* record, MaterialSample* sample) {
    float distMarch = 0;

    float3 invD = 1 / ray.direction;
    // 使用虛擬深度的最大值來計算 Offset，確保在大範圍下不會因為浮點數精度導致射線停滯。
    float rayOffset = Ray_dynamicOffset((float)(1 << self.virtualDepth));
    float3 offsetD = ray.direction * rayOffset;

    int depth = self.depth;

    // Check if we are in bounds
        if (!AABB_inside(self.bounds, ray.origin)) {
            // Attempt to intersect with the octree
            float dist = AABB_quick_intersect(self.bounds, ray.origin, invD);
            if (isnan(dist) || dist < 0) {
                return false;
            } else {
                distMarch += dist + rayOffset;
            }
        }

    for (int i = 0; i < drawDepth; i++) {
        if (distMarch > record->distance) {
            // There's already been a closer intersection!
            return false;
        }

        float3 pos = ray.origin + ray.direction * distMarch;
        int3 bp = intFloorFloat3(pos + offsetD);

        // Check inbounds
        int3 lv = bp >> depth;
        if (lv.x != 0 || lv.y != 0 || lv.z != 0) {
            return false;
        }

        // Read the octree with depth
        int level = depth;
        int data = self.treeData[0];
        while (data > 0) {
            level--;
            lv = 1 & (bp >> level);
            data = self.treeData[data + ((lv.x << 2) | (lv.y << 1) | lv.z)];
        }
        data = -data;
        lv = bp >> level;

        // Get block data if there is an intersection
        if (data != 0) {
            IntersectionRecord tempRecord = *record;
            MaterialSample tempSample;
            if (BlockPalette_intersectNormalizedBlock(palette, atlas, materialPalette, biome, emittersEnabled, data, bp, ray, &tempRecord, &tempSample)) {
                if (ray.currentMaterial != 0 && tempRecord.material == ray.currentMaterial) {
                    distMarch += tempRecord.distance + OFFSET;
                    continue;
                }
                if (data != ray.currentBlock || tempRecord.distance > rayOffset * 4.0f) {
                    *record = tempRecord;
                    *sample = tempSample;
                    return true;
                }
                distMarch += tempRecord.distance + rayOffset;
                continue;
            }
        }

        // Exit the current leaf
        AABB box = AABB_new(lv.x << level, (lv.x + 1) << level,
                            lv.y << level, (lv.y + 1) << level,
                            lv.z << level, (lv.z + 1) << level);
        distMarch += AABB_exit(box, pos + offsetD, invD) + rayOffset;
    }
    return false;
}
