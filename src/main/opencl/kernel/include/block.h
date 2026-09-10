// This includes stuff regarding blocks and block palettes

#ifndef CHUNKYCLPLUGIN_BLOCK_H
#define CHUNKYCLPLUGIN_BLOCK_H

#include "../opencl.h"
#include "rt.h"
#include "utils.h"
#include "constants.h"
#include "textureAtlas.h"
#include "material.h"
#include "primitives.h"

typedef struct {
    __global const int* blockPalette;
    __global const int* quadModels;
    __global const int* aabbModels;
    __global const int* waterModels;
} BlockPalette;

BlockPalette BlockPalette_new(__global const int* blockPalette, __global const int* quadModels, __global const int* aabbModels, __global const int* waterModels) {
    BlockPalette p;
    p.blockPalette = blockPalette;
    p.quadModels = quadModels;
    p.aabbModels = aabbModels;
    p.waterModels = waterModels;
    return p;
}

int BlockPalette_modelType(BlockPalette self, int block) {
    if (block == 0x7FFFFFFE || block == 0) {
        return 0;
    }
    return self.blockPalette[block + 0];
}

int BlockPalette_primaryMaterial(BlockPalette self, int block) {
    if (block == 0x7FFFFFFE || block == 0) {
        return 0;
    }

    int modelType = self.blockPalette[block + 0];
    int modelPointer = self.blockPalette[block + 1];

    switch (modelType) {
        default:
        case 0:
            return 0;
        case 1:
        case 4:
            return modelPointer;
        case 5:
            return self.waterModels[modelPointer + 0];
        case 2: {
            int boxes = self.aabbModels[modelPointer];
            for (int i = 0; i < boxes; i++) {
                int offset = modelPointer + 1 + i * TEX_AABB_SIZE;
                TexturedAABB box = TexturedAABB_new(self.aabbModels, offset);
                if (box.mn != 0) return box.mn;
                if (box.me != 0) return box.me;
                if (box.ms != 0) return box.ms;
                if (box.mw != 0) return box.mw;
                if (box.mt != 0) return box.mt;
                if (box.mb != 0) return box.mb;
            }
            return 0;
        }
        case 3: {
            int quads = self.quadModels[modelPointer];
            if (quads > 0) {
                int offset = modelPointer + 1;
                Quad q = Quad_new(self.quadModels, offset);
                return q.material;
            }
            return 0;
        }
    }
}

int BlockPalette_emitterFaceCount(BlockPalette self, int block) {
    if (block == 0x7FFFFFFE || block == 0) {
        return 0;
    }

    int modelType = self.blockPalette[block + 0];
    int modelPointer = self.blockPalette[block + 1];

    switch (modelType) {
        default:
        case 0:
        case 5:
            return 0;
        case 1:
        case 4:
            return 6;
        case 2: {
            int count = 0;
            int boxes = self.aabbModels[modelPointer];
            for (int i = 0; i < boxes; i++) {
                int offset = modelPointer + 1 + i * TEX_AABB_SIZE;
                TexturedAABB box = TexturedAABB_new(self.aabbModels, offset);
                if (((box.flags >> 0) & 0b1000) == 0) count++;
                if (((box.flags >> 4) & 0b1000) == 0) count++;
                if (((box.flags >> 8) & 0b1000) == 0) count++;
                if (((box.flags >> 12) & 0b1000) == 0) count++;
                if (((box.flags >> 16) & 0b1000) == 0) count++;
                if (((box.flags >> 20) & 0b1000) == 0) count++;
            }
            return count;
        }
        case 3:
            return self.quadModels[modelPointer];
    }
}

void sample_unit_block_face(int faceIndex, float2 uv, float3* outPos, float3* outNormal, float* outArea) {
    switch (faceIndex) {
        case 0:
            *outPos = (float3)(uv.x, uv.y, 0.0f);
            *outNormal = (float3)(0.0f, 0.0f, -1.0f);
            break;
        case 1:
            *outPos = (float3)(1.0f, uv.y, uv.x);
            *outNormal = (float3)(1.0f, 0.0f, 0.0f);
            break;
        case 2:
            *outPos = (float3)(1.0f - uv.x, uv.y, 1.0f);
            *outNormal = (float3)(0.0f, 0.0f, 1.0f);
            break;
        case 3:
            *outPos = (float3)(0.0f, uv.y, 1.0f - uv.x);
            *outNormal = (float3)(-1.0f, 0.0f, 0.0f);
            break;
        case 4:
            *outPos = (float3)(uv.x, 1.0f, 1.0f - uv.y);
            *outNormal = (float3)(0.0f, 1.0f, 0.0f);
            break;
        default:
            *outPos = (float3)(uv.x, 0.0f, uv.y);
            *outNormal = (float3)(0.0f, -1.0f, 0.0f);
            break;
    }
    *outArea = 1.0f;
}

bool sample_textured_aabb_face(TexturedAABB box, int faceIndex, float2 uv, float3* outPos, float3* outNormal, float* outArea) {
    float dx = box.box.xmax - box.box.xmin;
    float dy = box.box.ymax - box.box.ymin;
    float dz = box.box.zmax - box.box.zmin;
    switch (faceIndex) {
        case 0:
            if ((box.flags & 0b1000) != 0) return false;
            *outPos = (float3)(box.box.xmin + uv.x * dx, box.box.ymin + uv.y * dy, box.box.zmin);
            *outNormal = (float3)(0.0f, 0.0f, -1.0f);
            *outArea = dx * dy;
            return true;
        case 1:
            if (((box.flags >> 4) & 0b1000) != 0) return false;
            *outPos = (float3)(box.box.xmax, box.box.ymin + uv.y * dy, box.box.zmin + uv.x * dz);
            *outNormal = (float3)(1.0f, 0.0f, 0.0f);
            *outArea = dz * dy;
            return true;
        case 2:
            if (((box.flags >> 8) & 0b1000) != 0) return false;
            *outPos = (float3)(box.box.xmax - uv.x * dx, box.box.ymin + uv.y * dy, box.box.zmax);
            *outNormal = (float3)(0.0f, 0.0f, 1.0f);
            *outArea = dx * dy;
            return true;
        case 3:
            if (((box.flags >> 12) & 0b1000) != 0) return false;
            *outPos = (float3)(box.box.xmin, box.box.ymin + uv.y * dy, box.box.zmax - uv.x * dz);
            *outNormal = (float3)(-1.0f, 0.0f, 0.0f);
            *outArea = dz * dy;
            return true;
        case 4:
            if (((box.flags >> 16) & 0b1000) != 0) return false;
            *outPos = (float3)(box.box.xmin + uv.x * dx, box.box.ymax, box.box.zmax - uv.y * dz);
            *outNormal = (float3)(0.0f, 1.0f, 0.0f);
            *outArea = dx * dz;
            return true;
        default:
            if (((box.flags >> 20) & 0b1000) != 0) return false;
            *outPos = (float3)(box.box.xmin + uv.x * dx, box.box.ymin, box.box.zmin + uv.y * dz);
            *outNormal = (float3)(0.0f, -1.0f, 0.0f);
            *outArea = dx * dz;
            return true;
    }
}

bool BlockPalette_sampleEmitterFace(BlockPalette self, int block, int faceIndex, float2 uv, float3* outPos, float3* outNormal, float* outArea) {
    if (block == 0x7FFFFFFE || block == 0) {
        return false;
    }

    int modelType = self.blockPalette[block + 0];
    int modelPointer = self.blockPalette[block + 1];

    switch (modelType) {
        default:
        case 0:
            return false;
        case 1:
        case 4:
            if (faceIndex < 0 || faceIndex >= 6) return false;
            sample_unit_block_face(faceIndex, uv, outPos, outNormal, outArea);
            return true;
        case 2: {
            int boxes = self.aabbModels[modelPointer];
            for (int i = 0; i < boxes; i++) {
                int offset = modelPointer + 1 + i * TEX_AABB_SIZE;
                TexturedAABB box = TexturedAABB_new(self.aabbModels, offset);
                for (int face = 0; face < 6; face++) {
                    float3 pos;
                    float3 normal;
                    float area;
                    if (!sample_textured_aabb_face(box, face, uv, &pos, &normal, &area)) {
                        continue;
                    }
                    if (faceIndex == 0) {
                        *outPos = pos;
                        *outNormal = normal;
                        *outArea = area;
                        return true;
                    }
                    faceIndex--;
                }
            }
            return false;
        }
        case 3: {
            int quads = self.quadModels[modelPointer];
            if (faceIndex < 0 || faceIndex >= quads) {
                return false;
            }
            int offset = modelPointer + 1 + faceIndex * QUAD_SIZE;
            Quad q = Quad_new(self.quadModels, offset);
            *outPos = q.origin + uv.x * q.xv + uv.y * q.yv;
            *outNormal = normalize(cross(q.xv, q.yv));
            *outArea = length(cross(q.xv, q.yv));
            return true;
        }
    }
}

float WaterModel_cornerHeight(int level) {
    switch (level & 7) {
        case 0: return 14.0f / 16.0f;
        case 1: return 12.25f / 16.0f;
        case 2: return 10.5f / 16.0f;
        case 3: return 8.75f / 16.0f;
        case 4: return 7.0f / 16.0f;
        case 5: return 5.25f / 16.0f;
        case 6: return 3.5f / 16.0f;
        default: return 1.75f / 16.0f;
    }
}

bool WaterModel_sampleTriangle(
        float3 a,
        float3 b,
        float3 c,
        float2 ta,
        float2 tb,
        float2 tc,
        Material material,
        image2d_array_t atlas,
        Ray ray,
        int3 blockPos,
        BiomeColors biome,
        IntersectionRecord* record,
        MaterialSample* sample
) {
    float3 e1 = b - a;
    float3 e2 = c - a;
    float3 pvec = cross(ray.direction, e2);
    float det = dot(e1, pvec);
    if (fabs(det) <= EPS) {
        return false;
    }

    float recip = 1.0f / det;
    float3 tvec = ray.origin - a;
    float u = dot(tvec, pvec) * recip;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    float3 qvec = cross(tvec, e1);
    float v = dot(ray.direction, qvec) * recip;
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    float t = dot(e2, qvec) * recip;
    if (t <= EPS || t >= record->distance) {
        return false;
    }

    float w = 1.0f - u - v;
    float2 texCoord = ta * w + tb * u + tc * v;
    MaterialSample tempSample;
    if (!Material_sample_mode(material, atlas, texCoord, false, blockPos, biome, &tempSample, (ray.flags & RAY_OCCLUDER) != 0)) {
        return false;
    }

    float3 normal = normalize(cross(e1, e2));
    if (dot(normal, ray.direction) > 0.0f) {
        normal = -normal;
    }

    record->distance = t;
    record->texCoord = texCoord;
    record->normal = normal;
    *sample = tempSample;
    return true;
}

bool WaterModel_intersect(
        __global const int* waterModels,
        image2d_array_t atlas,
        MaterialPalette materialPalette,
        int modelPointer,
        Ray ray,
        int3 blockPos,
        BiomeColors biome,
        IntersectionRecord* record,
        MaterialSample* sample
) {
    int materialId = waterModels[modelPointer + 0];
    int data = waterModels[modelPointer + 1];
    Material material = Material_get(materialPalette, materialId);

    if (((data >> 16) & 1) != 0) {
        IntersectionRecord tempRecord = *record;
        if (AABB_full_intersect_map_2(AABB_new(0, 1, 0, 1, 0, 1), ray, &tempRecord) &&
                Material_sample_mode(material, atlas, tempRecord.texCoord, false, blockPos, biome, sample, (ray.flags & RAY_OCCLUDER) != 0)) {
            tempRecord.material = materialId;
            *record = tempRecord;
            return true;
        }
        return false;
    }

    bool hit = false;
    IntersectionRecord tempRecord = *record;
    float c0 = WaterModel_cornerHeight((data >> 0) & 0xF);
    float c1 = WaterModel_cornerHeight((data >> 4) & 0xF);
    float c2 = WaterModel_cornerHeight((data >> 8) & 0xF);
    float c3 = WaterModel_cornerHeight((data >> 12) & 0xF);

    hit |= WaterModel_sampleTriangle((float3)(0.0f, 0.0f, 0.0f), (float3)(1.0f, 0.0f, 0.0f), (float3)(0.0f, 0.0f, 1.0f),
            (float2)(0.0f, 0.0f), (float2)(1.0f, 0.0f), (float2)(0.0f, 1.0f), material, atlas, ray, blockPos, biome, &tempRecord, sample);
    hit |= WaterModel_sampleTriangle((float3)(0.0f, 0.0f, 1.0f), (float3)(1.0f, 0.0f, 0.0f), (float3)(1.0f, 0.0f, 1.0f),
            (float2)(0.0f, 1.0f), (float2)(1.0f, 0.0f), (float2)(1.0f, 1.0f), material, atlas, ray, blockPos, biome, &tempRecord, sample);
    hit |= WaterModel_sampleTriangle((float3)(0.0f, c0, 1.0f), (float3)(1.0f, c1, 1.0f), (float3)(1.0f, c2, 0.0f),
            (float2)(0.0f, 0.0f), (float2)(1.0f, 0.0f), (float2)(1.0f, 1.0f), material, atlas, ray, blockPos, biome, &tempRecord, sample);
    hit |= WaterModel_sampleTriangle((float3)(0.0f, c3, 0.0f), (float3)(0.0f, c0, 1.0f), (float3)(1.0f, c2, 0.0f),
            (float2)(0.0f, 1.0f), (float2)(0.0f, 0.0f), (float2)(1.0f, 1.0f), material, atlas, ray, blockPos, biome, &tempRecord, sample);
    hit |= WaterModel_sampleTriangle((float3)(0.0f, c3, 0.0f), (float3)(0.0f, 0.0f, 0.0f), (float3)(0.0f, c0, 1.0f),
            (float2)(0.0f, 1.0f), (float2)(0.0f, 0.0f), (float2)(1.0f, 1.0f), material, atlas, ray, blockPos, biome, &tempRecord, sample);
    hit |= WaterModel_sampleTriangle((float3)(0.0f, 0.0f, 1.0f), (float3)(0.0f, c0, 1.0f), (float3)(0.0f, 0.0f, 0.0f),
            (float2)(1.0f, 0.0f), (float2)(1.0f, 1.0f), (float2)(0.0f, 0.0f), material, atlas, ray, blockPos, biome, &tempRecord, sample);
    hit |= WaterModel_sampleTriangle((float3)(1.0f, c2, 0.0f), (float3)(1.0f, c1, 1.0f), (float3)(1.0f, 0.0f, 0.0f),
            (float2)(0.0f, 1.0f), (float2)(1.0f, 1.0f), (float2)(0.0f, 0.0f), material, atlas, ray, blockPos, biome, &tempRecord, sample);
    hit |= WaterModel_sampleTriangle((float3)(1.0f, c1, 1.0f), (float3)(1.0f, 0.0f, 1.0f), (float3)(1.0f, 0.0f, 0.0f),
            (float2)(1.0f, 1.0f), (float2)(1.0f, 0.0f), (float2)(0.0f, 0.0f), material, atlas, ray, blockPos, biome, &tempRecord, sample);
    hit |= WaterModel_sampleTriangle((float3)(0.0f, c3, 0.0f), (float3)(1.0f, c2, 0.0f), (float3)(0.0f, 0.0f, 0.0f),
            (float2)(0.0f, 1.0f), (float2)(1.0f, 1.0f), (float2)(0.0f, 0.0f), material, atlas, ray, blockPos, biome, &tempRecord, sample);
    hit |= WaterModel_sampleTriangle((float3)(1.0f, 0.0f, 0.0f), (float3)(0.0f, 0.0f, 0.0f), (float3)(1.0f, c2, 0.0f),
            (float2)(1.0f, 0.0f), (float2)(0.0f, 0.0f), (float2)(1.0f, 1.0f), material, atlas, ray, blockPos, biome, &tempRecord, sample);
    hit |= WaterModel_sampleTriangle((float3)(0.0f, c0, 1.0f), (float3)(0.0f, 0.0f, 1.0f), (float3)(1.0f, c1, 1.0f),
            (float2)(0.0f, 1.0f), (float2)(0.0f, 0.0f), (float2)(1.0f, 1.0f), material, atlas, ray, blockPos, biome, &tempRecord, sample);
    hit |= WaterModel_sampleTriangle((float3)(1.0f, 0.0f, 1.0f), (float3)(1.0f, c1, 1.0f), (float3)(0.0f, 0.0f, 1.0f),
            (float2)(1.0f, 0.0f), (float2)(1.0f, 1.0f), (float2)(0.0f, 0.0f), material, atlas, ray, blockPos, biome, &tempRecord, sample);

    if (hit) {
        tempRecord.material = materialId;
        *record = tempRecord;
    }
    return hit;
}

bool BlockPalette_intersectNormalizedBlock(BlockPalette self, image2d_array_t atlas, MaterialPalette materialPalette, BiomeColors biome, bool emittersEnabled, int block, int3 blockPosition, Ray ray, IntersectionRecord* record, MaterialSample* sample) {
    // ANY_TYPE. Should not be intersected.
    if (block == 0x7FFFFFFE) {
        return false;
    }
    
    int modelType = self.blockPalette[block + 0];
    int modelPointer = self.blockPalette[block + 1];

    bool hit = false;
    Ray tempRay = ray;
    tempRay.origin = ray.origin - int3toFloat3(blockPosition);

    IntersectionRecord tempRecord = *record;

    switch (modelType) {
        default:
        case 0: {
            return false;
        }
        case 1: {
            AABB box = AABB_new(0, 1, 0, 1, 0, 1);
            bool insideBlock = AABB_inside(box, tempRay.origin);
            hit = AABB_full_intersect(box, tempRay, &tempRecord);
            tempRecord.material = modelPointer;
            if (hit) {
                if (tempRecord.normal.x > 0 || tempRecord.normal.z < 0) {
                    tempRecord.texCoord.x = 1 - tempRecord.texCoord.x;
                }
                if (tempRecord.normal.y > 0) {
                    tempRecord.texCoord.y = 1 - tempRecord.texCoord.y;
                }

                Material material = Material_get(materialPalette, tempRecord.material);
                if ((ray.flags & RAY_OCCLUDER) && Material_isOpaqueOccluder(material)) {
                    // Shadow ray through an opaque non-emissive block: the texel never
                    // needs to be sampled (the ray stops here anyway).
                    MaterialSample_opaque(sample);
                    tempRecord.block = block;
                    *record = tempRecord;
                    return true;
                }
                hit = Material_sample_mode(material, atlas, tempRecord.texCoord, true, blockPosition, biome, sample, (ray.flags & RAY_OCCLUDER) != 0);
                if (hit) {
                    if (insideBlock && Material_isRefractive(material) && !Material_isOpaque(material) &&
                            dot(tempRecord.normal, tempRay.direction) > 0.0f &&
                            sample->color.w <= EPS) {
                        // Exiting clear glass should keep the medium boundary for refraction,
                        // but fully transparent texels should not render a visible back-face frame.
                        sample->color = (float4)(1.0f, 1.0f, 1.0f, 0.0f);
                    }
                    tempRecord.block = block;
                    *record =tempRecord;
                    return true;
                } else {
                    return false;
                }
            }
            return false;
        }
        case 2: {
            int boxes = self.aabbModels[modelPointer];
            for (int i = 0; i < boxes; i++) {
                int offset = modelPointer + 1 + i * TEX_AABB_SIZE;
                TexturedAABB box = TexturedAABB_new(self.aabbModels, offset);
                hit |= TexturedAABB_intersect(box, atlas, materialPalette, tempRay, blockPosition, biome, record, sample);
            }
            if (hit) {
                record->block = block;
            }
            return hit;
        }
        case 3: {
            int quads = self.quadModels[modelPointer];
            for (int i = 0; i < quads; i++) {
                int offset = modelPointer + 1 + i * QUAD_SIZE;
                Quad q = Quad_new(self.quadModels, offset);
                hit |= Quad_intersect(q, atlas, materialPalette, tempRay, blockPosition, biome, record, sample);
            }
            if (hit) {
                record->block = block;
            }
            return hit;
        }
        case 4: {
            // Invisible light block: never contributes visible surface color, but rays still
            // intersect it so the emitter's light can be picked up directly and so it occludes
            // shadow rays, matching the CPU renderer's LightBlockModel. Like the CPU, light
            // blocks are fully transparent when emitters are disabled.
            if (!emittersEnabled) {
                return false;
            }
            AABB box = AABB_new(0.125f, 0.875f, 0.125f, 0.875f, 0.125f, 0.875f);
            hit = AABB_full_intersect(box, tempRay, &tempRecord);
            tempRecord.material = modelPointer;
            if (hit) {
                if (tempRecord.normal.x > 0 || tempRecord.normal.z < 0) {
                    tempRecord.texCoord.x = 1 - tempRecord.texCoord.x;
                }
                if (tempRecord.normal.y > 0) {
                    tempRecord.texCoord.y = 1 - tempRecord.texCoord.y;
                }

                Material material = Material_get(materialPalette, tempRecord.material);
                hit = Material_sample_mode(material, atlas, tempRecord.texCoord, true, blockPosition, biome, sample, (ray.flags & RAY_OCCLUDER) != 0);
                if (hit) {
                    tempRecord.block = block;
                    *record = tempRecord;
                    return true;
                }
                return false;
            }
            return false;
        }
        case 5: {
            hit = WaterModel_intersect(self.waterModels, atlas, materialPalette, modelPointer, tempRay, blockPosition, biome, &tempRecord, sample);
            if (hit) {
                tempRecord.block = block;
                *record = tempRecord;
            }
            return hit;
        }
    }
}

#endif
