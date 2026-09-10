#include "kernel.h"

bool closestIntersect(Scene self, image2d_array_t atlas, Ray ray, IntersectionRecord* record, MaterialSample* sample, Material* mat) {
    bool hit = false;

    // Clouds first (matches the CPU nextIntersection order).
#ifdef HAS_CLOUDS
    if (self.atmosphere.cloudsEnabled) {
        if (Cloud_intersect(self.atmosphere, ray, record, sample)) {
            hit = true;
        }
    }
#endif

    // Water plane (CPU's waterPlaneIntersection): infinite horizontal plane below the
    // loaded chunks, entering from above with the water material and exiting from
    // below with air.
#ifdef HAS_WATERPLANE
    if (self.atmosphere.waterPlaneEnabled && fabs(ray.direction.y) > EPS) {
        Profile_inc(self.profile, self.profileCounters, PROFILE_WATER_PLANE_TESTS);
        float t = (self.atmosphere.waterPlaneY - ray.origin.y) / ray.direction.y;
        if (t > EPS && t < record->distance) {
            IntersectionRecord planeRecord = *record;
            planeRecord.distance = t;
            planeRecord.block = 0;
            float3 wp = ray.origin + ray.direction * t;
            float2 uv = (float2)(fmod(fabs(wp.x), 1.0f), fmod(fabs(wp.z), 1.0f));
            if (ray.direction.y < 0.0f) {
                planeRecord.normal = (float3)(0.0f, 1.0f, 0.0f);
                planeRecord.material = self.atmosphere.waterMaterial;
                Material waterMat = Material_get(self.materialPalette, planeRecord.material);
                if (Material_sample_mode(waterMat, atlas, uv, false, intFloorFloat3(wp), self.biome, sample, (ray.flags & RAY_OCCLUDER) != 0)) {
                    sample->color.w = self.atmosphere.waterOpacity;
                    *record = planeRecord;
                    hit = true;
                }
            } else {
                // Exiting the water plane from below.
                planeRecord.normal = (float3)(0.0f, -1.0f, 0.0f);
                planeRecord.material = 0;
                Material waterMat = Material_get(self.materialPalette, self.atmosphere.waterMaterial);
                if (Material_sample_mode(waterMat, atlas, uv, false, intFloorFloat3(wp), self.biome, sample, (ray.flags & RAY_OCCLUDER) != 0)) {
                    sample->color.w = self.atmosphere.waterOpacity;
                    *record = planeRecord;
                    hit = true;
                }
            }
        }
    }
#endif
    
    // Solid octree first (usually the densest geometry in the scene).
    if (Octree_octreeIntersect(self.octree, atlas, self.blockPalette, self.materialPalette, self.biome, self.drawDepth, self.emittersEnabled, ray, record, sample)) {
        hit = true;
    }
    
    // Water octree (only matters when closer than the current hit).
#ifdef HAS_WATER
    if (self.atmosphere.hasWater &&
            Octree_octreeIntersect(self.waterOctree, atlas, self.blockPalette, self.materialPalette, self.biome, self.drawDepth, self.emittersEnabled, ray, record, sample)) {
        hit = true;
    }
#endif

    // BVHs (likewise only update on a closer hit).
    // Note: returns quickly when the scene has no entities.
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
#ifdef HAS_WATER
        if (scene.atmosphere.hasWater) {
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
        }
#endif
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
