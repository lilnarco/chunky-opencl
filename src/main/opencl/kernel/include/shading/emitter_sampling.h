#include "kernel.h"

float3 sampleEmitterFace(
        Scene scene,
        image2d_array_t textureAtlas,
        float3 hitPoint,
        float3 shadingNormal,
        int4 emitter,
        int face,
        float faceScaler,
        float emitterIntensity,
        bool fancierTranslucency,
        float transmissivityCap,
        Random random
) {
    float3 localPos;
    float3 emitterNormal;
    float emitterArea;
    float2 uv = (float2)(Random_nextFloat(random), Random_nextFloat(random));
    if (!BlockPalette_sampleEmitterFace(scene.blockPalette, emitter.w, face, uv, &localPos, &emitterNormal, &emitterArea)) {
        return (float3)(0.0f);
    }

    float3 target = convert_float3((int3)(emitter.x, emitter.y, emitter.z)) + localPos;
    float3 toEmitter = target - hitPoint;
    float distance = length(toEmitter);
    if (distance <= EPS) {
        return (float3)(0.0f);
    }

    float3 direction = toEmitter / distance;
    if (dot(direction, shadingNormal) <= 0.0f) {
        return (float3)(0.0f);
    }

    Ray shadowRay;
    shadowRay.origin = hitPoint + direction * OFFSET;
    shadowRay.direction = direction;
    shadowRay.prevMaterial = 0;
    shadowRay.currentMaterial = 0;
    shadowRay.prevBlock = 0;
    shadowRay.currentBlock = 0;
    shadowRay.flags = RAY_INDIRECT | RAY_OCCLUDER;

    float traveled = OFFSET;
    float3 attenuation = (float3)(1.0f, 1.0f, 1.0f);
    Profile_inc(scene.profile, scene.profileCounters, PROFILE_EMITTER_RAYS);
    while (traveled < distance) {
        Profile_inc(scene.profile, scene.profileCounters, PROFILE_EMITTER_RAY_STEPS);
        IntersectionRecord record = IntersectionRecord_new();
        MaterialSample sample;
        Material material;
        if (!closestIntersect(scene, textureAtlas, shadowRay, &record, &sample, &material)) {
            return (float3)(0.0f);
        }

        traveled += record.distance;
        if (fabs(traveled - distance) <= (2.0f * OFFSET)) {
            if (sample.emittance <= EPS) {
                return (float3)(0.0f);
            }
            float e = fabs(dot(direction, record.normal));
            e /= fmax(distance * distance, 1.0f);
            e *= emitterArea;
            e *= sample.emittance;
            e *= emitterIntensity;
            e *= faceScaler;
            return attenuation * sample.color.xyz * e;
        }

        // Water surface alpha is the scene's water opacity (CPU parity).
        if (Material_isWater(material)) {
            sample.color.w = scene.atmosphere.waterOpacity;
        }
        if (Material_isOpaque(material)) {
            Profile_inc(scene.profile, scene.profileCounters, PROFILE_OCCLUDER_FAST_PATH_HITS);
        }

        float3 transmittance = Material_translucentTransmission(
                sample,
                sample.color.w,
                transmissivityCap,
                fancierTranslucency
        );
        if (Material_isRefractive(material) && !Material_isOpaque(material) && sample.color.w > EPS) {
            float colorAvg = (sample.color.x + sample.color.y + sample.color.z) / 3.0f;
            if (colorAvg > EPS) {
                float3 tintBoost = clamp(sample.color.xyz / colorAvg, (float3)(0.0f), (float3)(4.0f));
                float tintMix = clamp(sample.color.w * 2.25f, 0.0f, 1.0f);
                transmittance *= mix((float3)(1.0f), tintBoost, tintMix);
            }
        }
        attenuation *= transmittance;
        if (fmax(attenuation.x, fmax(attenuation.y, attenuation.z)) <= 1.0e-3f) {
            return (float3)(0.0f);
        }

        shadowRay.prevMaterial = shadowRay.currentMaterial;
        shadowRay.prevBlock = shadowRay.currentBlock;
        shadowRay.currentMaterial = record.material;
        shadowRay.currentBlock = record.block;
        shadowRay.origin += shadowRay.direction * (record.distance + OFFSET);
        traveled += OFFSET;
    }

    return (float3)(0.0f);
}

float3 sampleEmitters(
        Scene scene,
        image2d_array_t textureAtlas,
        float3 hitPoint,
        float3 shadingNormal,
        int strategy,
        float emitterIntensity,
        bool fancierTranslucency,
        float transmissivityCap,
        Random random
) {
    Profile_inc(scene.profile, scene.profileCounters, PROFILE_EMITTER_SAMPLES);
    int start;
    int count;
    if (!EmitterGrid_cellRange(scene.emitterGrid, intFloorFloat3(hitPoint), &start, &count) || count <= 0) {
        return (float3)(0.0f);
    }
    Profile_inc(scene.profile, scene.profileCounters, PROFILE_EMITTER_GRID_LOOKUPS);

    float3 result = (float3)(0.0f);
    switch (strategy) {
        case 1:
        case 2: {
            int emitterListIndex = start + (int)(Random_nextFloat(random) * count);
            if (emitterListIndex >= start + count) {
                emitterListIndex = start + count - 1;
            }
            int emitterIndex = scene.emitterGrid.indexes[emitterListIndex];
            int4 emitter = EmitterGrid_getEmitter(scene.emitterGrid, emitterIndex);
            int faceCount = BlockPalette_emitterFaceCount(scene.blockPalette, emitter.w);
            if (faceCount <= 0) {
                return (float3)(0.0f);
            }
            // Fast reject: if the emitter's block center is behind the surface plane,
            // every face is too.
            float3 toCenter = convert_float3((int3)(emitter.x, emitter.y, emitter.z)) +
                    (float3)(0.5f, 0.5f, 0.5f) - hitPoint;
            if (dot(toCenter, shadingNormal) <= 0.0f) {
                return (float3)(0.0f);
            }
            if (strategy == 1) {
                int face = (int)(Random_nextFloat(random) * faceCount);
                if (face >= faceCount) {
                    face = faceCount - 1;
                }
                result += sampleEmitterFace(scene, textureAtlas, hitPoint, shadingNormal, emitter, face, M_PI_F, emitterIntensity, fancierTranslucency, transmissivityCap, random);
            } else {
                float scaler = M_PI_F / (float) faceCount;
                for (int face = 0; face < faceCount; face++) {
                    result += sampleEmitterFace(scene, textureAtlas, hitPoint, shadingNormal, emitter, face, scaler, emitterIntensity, fancierTranslucency, transmissivityCap, random);
                }
            }
            break;
        }
        case 3: {
            float emitterScaler = M_PI_F / (float) count;
            for (int i = 0; i < count; i++) {
                int emitterIndex = scene.emitterGrid.indexes[start + i];
                int4 emitter = EmitterGrid_getEmitter(scene.emitterGrid, emitterIndex);
                int faceCount = BlockPalette_emitterFaceCount(scene.blockPalette, emitter.w);
                if (faceCount <= 0) {
                    continue;
                }
                float3 toCenter = convert_float3((int3)(emitter.x, emitter.y, emitter.z)) +
                        (float3)(0.5f, 0.5f, 0.5f) - hitPoint;
                if (dot(toCenter, shadingNormal) <= 0.0f) {
                    continue;
                }
                float faceScaler = emitterScaler / (float) faceCount;
                for (int face = 0; face < faceCount; face++) {
                    result += sampleEmitterFace(scene, textureAtlas, hitPoint, shadingNormal, emitter, face, faceScaler, emitterIntensity, fancierTranslucency, transmissivityCap, random);
                }
            }
            break;
        }
        default:
            break;
    }
    return result;
}
