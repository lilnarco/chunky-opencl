#include "kernel.h"

bool closestIntersect(Scene self, image2d_array_t atlas, Ray ray, IntersectionRecord* record, MaterialSample* sample, Material* mat) {
    bool hit = false;

    // 0. 雲層優先測試 (與 CPU 的 nextIntersection 一致)
    if (self.atmosphere.cloudsEnabled) {
        if (Cloud_intersect(self.atmosphere, ray, record, sample)) {
            hit = true;
        }
    }
    
    // 1. 優先測試 Octree (通常是場景中最密集的物體)
    if (Octree_octreeIntersect(self.octree, atlas, self.blockPalette, self.materialPalette, self.biome, self.drawDepth, self.emittersEnabled, ray, record, sample)) {
        hit = true;
    }
    
    // 2. 測試水面 Octree (只有在距離比目前撞到的更短時才有意義)
    if (Octree_octreeIntersect(self.waterOctree, atlas, self.blockPalette, self.materialPalette, self.biome, self.drawDepth, self.emittersEnabled, ray, record, sample)) {
        hit = true;
    }

    // 3. 測試 BVH (同樣只在更短的情況下更新 hit)
    // 注意：如果場景沒有實體，這部分會很快返回
    if (Bvh_intersect(self.worldBvh, atlas, self.materialPalette, self.biome, ray, record, sample)) {
        hit = true;
    }
    
    if (Bvh_intersect(self.actorBvh, atlas, self.materialPalette, self.biome, ray, record, sample)) {
        hit = true;
    }

    if (hit) {
        *mat = Material_get(self.materialPalette, record->material);
        return true;
    }
    
    return false;
}

void initialize_ray_medium(Scene scene, Ray* ray) {
    int3 blockPos = intFloorFloat3(ray->origin);
    int block = Octree_get(&scene.octree, blockPos.x, blockPos.y, blockPos.z);
    if (block == 0) {
        int waterBlock = Octree_get(&scene.waterOctree, blockPos.x, blockPos.y, blockPos.z);
        if (waterBlock != 0) {
            int waterMaterial = BlockPalette_primaryMaterial(scene.blockPalette, waterBlock);
            Material waterMat = Material_get(scene.materialPalette, waterMaterial);
            if (!Material_isOpaque(waterMat)) {
                ray->prevMaterial = 0;
                ray->currentMaterial = waterMaterial;
                ray->prevBlock = 0;
                ray->currentBlock = waterBlock;
                return;
            }
        }
        ray->prevMaterial = 0;
        ray->currentMaterial = 0;
        ray->prevBlock = 0;
        ray->currentBlock = 0;
        return;
    }

    int material = BlockPalette_primaryMaterial(scene.blockPalette, block);
    Material currentMat = Material_get(scene.materialPalette, material);
    if (Material_isRefractive(currentMat) && !Material_isOpaque(currentMat)) {
        ray->prevMaterial = 0;
        ray->currentMaterial = material;
        ray->prevBlock = 0;
        ray->currentBlock = block;
    } else {
        ray->prevMaterial = 0;
        ray->currentMaterial = 0;
        ray->prevBlock = 0;
        ray->currentBlock = 0;
    }
}
