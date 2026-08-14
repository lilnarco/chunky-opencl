#include "kernel.h"
#include "camera.h"
#include "material.h"
#include "sky.h"
#include "noise.h"

// The albedo/normal auxiliary passes converge within a few dozen samples (their only
// noise is edge anti-aliasing), so stop accumulating them after this many samples.
#define GUIDE_SPP_CAP 32

bool closestIntersect(Scene self, image2d_array_t atlas, Ray ray, IntersectionRecord* record, MaterialSample* sample, Material* mat);
void initialize_ray_medium(Scene scene, Ray* ray);
float computeDiffuseProbability(float4 color, bool fancierTranslucency);
float computeAbsorption(float4 color, float pDiffuse, bool fancierTranslucency);
float4 getDirectLightAttenuation(Scene scene, image2d_array_t textureAtlas, Ray ray, bool strictDirectLight);
float3 sampleEmitters(Scene scene, image2d_array_t textureAtlas, float3 hitPoint, float3 shadingNormal, int strategy, float emitterIntensity, bool fancierTranslucency, float transmissivityCap, Random random);
void intersectSky(image2d_t skyTexture, Sun sun, image2d_array_t atlas, Atmosphere atmosphere, Ray ray, MaterialSample* sample);

Ray ray_to_camera(
        const __global int* projectorType,
        const __global float* cameraSettings,
        const __global int* canvasConfig,
        int gid,
        Random random
) {
    Ray ray;
    if (*projectorType != -1) {
        float3 cameraPos = vload3(0, cameraSettings);
        float3 m1s = vload3(1, cameraSettings);
        float3 m2s = vload3(2, cameraSettings);
        float3 m3s = vload3(3, cameraSettings);

        int width = canvasConfig[0];
        int height = canvasConfig[1];
        int fullWidth = canvasConfig[2];
        int fullHeight = canvasConfig[3];
        int cropX = canvasConfig[4];
        int cropY = canvasConfig[5];

        float halfWidth = fullWidth / (2.0 * fullHeight);
        float invHeight = 1.0 / fullHeight;
        float x = -halfWidth + ((gid % width) + Random_nextFloat(random) + cropX) * invHeight;
        float y = -0.5 + ((gid / width) + Random_nextFloat(random) + cropY) * invHeight;

        switch (*projectorType) {
            case 0:
                ray = Camera_pinHole(x, y, random, cameraSettings + 12);
                break;
            case 1:
                ray = Camera_parallel(x, y, cameraSettings + 12);
                break;
        }

        ray.direction = normalize((float3) (
                dot(m1s, ray.direction),
                        dot(m2s, ray.direction),
                        dot(m3s, ray.direction)
        ));
        ray.origin = (float3) (
                dot(m1s, ray.origin),
                        dot(m2s, ray.origin),
                        dot(m3s, ray.origin)
        );

        ray.origin += cameraPos;
    } else {
        ray = Camera_preGenerated(cameraSettings, gid);
    }
    return ray;
}

__kernel void render(
    __global const int* projectorType,
    __global const float* cameraSettings,

    __global const int* octreeDepth,
    __global const int* octreeData,
    __global const int* waterOctreeDepth,
    __global const int* waterOctreeData,

    __global const int* bPalette,
    __global const int* quadModels,
    __global const int* aabbModels,
    __global const int* waterModels,

    __global const int* worldBvhData,
    __global const int* actorBvhData,
    __global const int* bvhTrigs,

    image2d_array_t textureAtlas,
    __global const int* matPalette,
    __global const int* biomeMeta,
    __global const int* biomeGrid,
    __global const float* biomeGrass,
    __global const float* biomeFoliage,
    __global const float* biomeDryFoliage,
    __global const float* biomeWater,
    __global const int* emitterGridMeta,
    __global const int* emitterGridCells,
    __global const int* emitterGridIndexes,
    __global const int* emitterGridEmitters,

    image2d_t skyTexture,
    __global const int* sunData,

    __global const int* randomSeed,
    __global const int* bufferSpp,
    __global const int* canvasConfig,
    __global const int* rayDepth,
    __global const float* sceneSettings,
    __global const float* atmosphereSettings,
    __global const int* cloudData,
    int emittersEnabled,
    float emitterIntensity,
    int emitterSamplingStrategy,
    int preventNormalEmitterWithSampling,
    int profileEnabled,
    __global int* profileCounters,
    __global float* albedoRes,
    __global float* normalRes,
    __global float* res

) {
    int gid = get_global_id(0);

    float maxCoord = sceneSettings[6];

    Scene scene;
    scene.materialPalette = MaterialPalette_new(matPalette);
    scene.octree = Octree_create(octreeData, *octreeDepth, maxCoord);
    scene.waterOctree = Octree_create(waterOctreeData, *waterOctreeDepth, maxCoord);
    scene.worldBvh = Bvh_new(worldBvhData, bvhTrigs);
    scene.actorBvh = Bvh_new(actorBvhData, bvhTrigs);
    scene.blockPalette = BlockPalette_new(bPalette, quadModels, aabbModels, waterModels);
    scene.biome = BiomeColors_new(biomeMeta, biomeGrid, biomeGrass, biomeFoliage, biomeDryFoliage, biomeWater);
    scene.emitterGrid = EmitterGrid_new(emitterGridMeta, emitterGridCells, emitterGridIndexes, emitterGridEmitters);
    scene.atmosphere = Atmosphere_new(atmosphereSettings, cloudData);
    scene.drawDepth = 256;
    scene.emittersEnabled = emittersEnabled != 0;
    scene.profile = profileEnabled != 0;
    scene.profileCounters = profileCounters;
    scene.octree = Octree_profile(scene.octree, scene.profile, profileCounters);
    scene.waterOctree = Octree_profile(scene.waterOctree, scene.profile, profileCounters);
    scene.worldBvh = Bvh_profile(scene.worldBvh, scene.profile, profileCounters);
    scene.actorBvh = Bvh_profile(scene.actorBvh, scene.profile, profileCounters);
    scene.atmosphere.profile = scene.profile;
    scene.atmosphere.profileCounters = profileCounters;

    Sun sun = Sun_new(sunData);

    unsigned int randomState = *randomSeed + gid;
    Random random = &randomState;
    Random_nextState(random);
    Ray ray = ray_to_camera(projectorType, cameraSettings, canvasConfig, gid, random);

    initialize_ray_medium(scene, &ray);
    ray.flags = 0;

    Profile_inc(scene.profile, scene.profileCounters, PROFILE_TOTAL_RAYS);

    float3 color = (float3) (0.0);
    float3 throughput = (float3) (1.0);
    float traveled = 0.0f;
    float airDistance = 0.0f;
    float waterDistance = 0.0f;
    float3 firstAlbedo = (float3) (0.0f);
    float3 firstNormal = (float3) (0.0f, 1.0f, 0.0f);
    float transmissivityCap = sceneSettings[0];
    bool fancierTranslucency = sceneSettings[1] > 0.5f;
    bool doSunSampling = sceneSettings[2] > 0.5f;
    bool sunLuminosity = sceneSettings[3] > 0.5f;
    bool strictDirectLight = sceneSettings[4] > 0.5f;
    float rrThreshold = sceneSettings[5] / 100.0f; // 俄羅斯輪盤閾值 (0.0 ~ 1.0)
    // NONE (0) means no emitter-grid sampling, exactly like the CPU renderer.
    int effectiveEmitterSamplingStrategy = emitterSamplingStrategy;

    for (int depth = 0; depth < *rayDepth; depth++) {
        // 實作俄羅斯輪盤 (Russian Roulette)
        // 在前 3 跳之後，如果路徑能量過低，則機率性終止，以提升 GPU 效率。
        if (depth > 2) {
            float p = fmax(throughput.x, fmax(throughput.y, throughput.z));
            if (p < rrThreshold) {
                if (Random_nextFloat(random) > p) {
                    break;
                }
                throughput /= p; // 能量補償，保持渲染無偏
            }
        }

        IntersectionRecord record = IntersectionRecord_new();
        MaterialSample sample;
        Material material;

        if (closestIntersect(scene, textureAtlas, ray, &record, &sample, &material)) {
            traveled += record.distance;

            // Capture the first *visible* surface's albedo and world-space normal for the
            // auxiliary render passes (and OIDN denoising). March past semi-transparent
            // texels (alpha < 0.5) so the passes show the object behind leaves/glass.
            if (depth == 0) {
                Ray auxRay = ray;
                IntersectionRecord auxRecord = record;
                MaterialSample auxSample = sample;
                Material auxMaterial = material;
                for (int step = 0; step < 8; step++) {
                    if (auxSample.color.w >= 0.5f) {
                        break;
                    }
                    auxRay.origin += auxRay.direction * (auxRecord.distance + OFFSET);
                    auxRay.prevMaterial = auxRay.currentMaterial;
                    auxRay.prevBlock = auxRay.currentBlock;
                    auxRay.currentMaterial = auxRecord.material;
                    auxRay.currentBlock = auxRecord.block;
                    if (!closestIntersect(scene, textureAtlas, auxRay, &auxRecord, &auxSample, &auxMaterial)) {
                        auxRecord.normal = (float3)(0.0f, 1.0f, 0.0f);
                        intersectSky(skyTexture, sun, textureAtlas, scene.atmosphere, auxRay, &auxSample);
                        break;
                    }
                }
                firstAlbedo = auxSample.color.xyz;
                firstNormal = auxRecord.normal;
            }

            ray.prevMaterial = ray.currentMaterial;
            ray.prevBlock = ray.currentBlock;
            ray.currentMaterial = record.material;
            ray.currentBlock = record.block;

            Profile_inc(scene.profile, scene.profileCounters, PROFILE_TOTAL_HITS);

            Material currentMat = Material_get(scene.materialPalette, ray.currentMaterial);
            Material prevMat = Material_get(scene.materialPalette, ray.prevMaterial);

            // Track the distance traveled through air (or water) for uniform fog, like
            // the CPU's `if (prevMat == Air || prevMat.isWater()) airDistance = ray.distance`.
            if (ray.prevMaterial == 0 || Material_isWater(prevMat)) {
                airDistance = traveled;
            }
            // Underwater visibility: accumulate the distance traveled through water.
            if (Material_isWater(prevMat)) {
                waterDistance += record.distance;
            }

            // Water surface alpha is the scene's water opacity (CPU parity).
            if (Material_isWater(currentMat) || Material_isWater(prevMat)) {
                sample.color.w = scene.atmosphere.waterOpacity;
            }

            float pSpecular = sample.specular;
            float pDiffuse = computeDiffuseProbability(sample.color, fancierTranslucency);
            float pAbsorb = computeAbsorption(sample.color, pDiffuse, fancierTranslucency);
            float n1 = Material_ior(prevMat);
            float n2 = Material_ior(currentMat);
            float3 hitPoint = ray.origin + ray.direction * record.distance;

            // Animated water surface (Simplex shader): perturb the surface normal at
            // water-air boundaries, like the CPU's SimplexWaterShader.
            if (scene.atmosphere.waterShader == 1 &&
                    (Material_isWater(currentMat) != Material_isWater(prevMat)) &&
                    fabs(record.normal.y) > 0.1f) {
                record.normal = SimplexWaterNormal(hitPoint.x, hitPoint.z, scene.atmosphere.animationTime);
                Profile_inc(scene.profile, scene.profileCounters, PROFILE_WAVE_NOISE_CALLS);
            }

            if (sample.color.w + pSpecular < EPS && fabs(n1 - n2) < EPS) {
                ray.origin = hitPoint + ray.direction * OFFSET;
                continue;
            }

            bool didSpecularBounce = true;
            // Translucent texels (alpha < 1) get a proportionally smaller chance of a
            // metal/specular bounce so the alpha-aware diffuse/transmission branches keep
            // the material translucent (cherry leaves with metalness stay holey/airy).
            float surfaceAlpha = sample.color.w;
            bool doMetal = sample.metalness > EPS && sample.metalness * surfaceAlpha > Random_nextFloat(random);
            if (doMetal) {
                Profile_inc(scene.profile, scene.profileCounters, PROFILE_SPECULAR_BOUNCES);
                throughput *= sample.color.xyz;
                ray.origin = hitPoint;
                ray.direction = _Material_specularReflection(record, sample, ray, random);
                ray.origin += ray.direction * OFFSET;
                ray.currentMaterial = ray.prevMaterial;
                ray.currentBlock = ray.prevBlock;
            } else if (pSpecular > EPS && pSpecular * surfaceAlpha > Random_nextFloat(random)) {
                Profile_inc(scene.profile, scene.profileCounters, PROFILE_SPECULAR_BOUNCES);
                ray.origin = hitPoint;
                ray.direction = _Material_specularReflection(record, sample, ray, random);
                ray.origin += ray.direction * OFFSET;
                ray.currentMaterial = ray.prevMaterial;
                ray.currentBlock = ray.prevBlock;
            } else if (Random_nextFloat(random) < pDiffuse) {
                Profile_inc(scene.profile, scene.profileCounters, PROFILE_DIFFUSE_BOUNCES);
                bool allowNormalEmitter = emittersEnabled != 0 &&
                        (!preventNormalEmitterWithSampling || effectiveEmitterSamplingStrategy == 0 || depth == 0);
                if (allowNormalEmitter && sample.emittance > EPS) {
                    color += throughput * sample.color.xyz * sample.color.xyz * sample.emittance * emitterIntensity;
                } else if (emittersEnabled != 0 &&
                        effectiveEmitterSamplingStrategy != 0 &&
                        emitterIntensity > EPS &&
                        sample.emittance <= EPS) {
                    float3 emitterLight = sampleEmitters(
                            scene,
                            textureAtlas,
                            hitPoint,
                            record.normal,
                            effectiveEmitterSamplingStrategy,
                            emitterIntensity,
                            fancierTranslucency,
                            transmissivityCap,
                            random
                    );
                    color += throughput * sample.color.xyz * emitterLight;
                }

                if (doSunSampling && sun.intensity > EPS && sun.sw.y >= 0.0f) {
                    Ray sunRay = ray;
                    sunRay.origin = hitPoint;
                    sunRay.currentMaterial = ray.prevMaterial;
                    sunRay.currentBlock = ray.prevBlock;
                    sunRay.prevMaterial = ray.prevMaterial;
                    sunRay.prevBlock = ray.prevBlock;
                    sunRay.flags = RAY_INDIRECT | RAY_OCCLUDER;

                    if (Sun_sampleDirection(sun, &sunRay, random)) {
                        float frontLight = dot(sunRay.direction, record.normal);
                        if (frontLight > 0.0f) {
                            float4 attenuation = getDirectLightAttenuation(
                                    scene,
                                    textureAtlas,
                                    sunRay,
                                    strictDirectLight
                            );
                            if (attenuation.w > 0.0f) {
                                float mult = fabs(frontLight) * (sunLuminosity ? sun.luminosityPdf : 1.0f);
                                float3 directLight = attenuation.xyz * attenuation.w * mult;
                                color += throughput * sample.color.xyz * directLight * Sun_emittance(sun);
                            }
                        }
                    }
                }

                throughput *= sample.color.xyz;
                ray.origin = hitPoint;
                ray.direction = _Material_diffuseReflection(record, random);
                ray.origin += ray.direction * OFFSET;
                ray.currentMaterial = ray.prevMaterial;
                ray.currentBlock = ray.prevBlock;
                didSpecularBounce = false;
            } else if (fabs(n1 - n2) >= EPS) {
                Profile_inc(scene.profile, scene.profileCounters, PROFILE_REFRACTION_BOUNCES);
                bool doRefraction = Material_isRefractive(currentMat) || Material_isRefractive(prevMat);
                float n1n2 = n1 / n2;
                float cosTheta = -dot(record.normal, ray.direction);
                float radicand = 1 - n1n2 * n1n2 * (1 - cosTheta * cosTheta);

                if (doRefraction && radicand < EPS) {
                    ray.origin = hitPoint;
                    ray.direction = _Material_specularReflection(record, sample, ray, random);
                    ray.origin += ray.direction * OFFSET;
                    ray.currentMaterial = ray.prevMaterial;
                    ray.currentBlock = ray.prevBlock;
                } else {
                    float a = n1n2 - 1;
                    float b = n1n2 + 1;
                    float R0 = (a * a) / (b * b);
                    float c = 1 - cosTheta;
                    float Rtheta = R0 + (1 - R0) * (c * c * c * c * c);

                    if (Random_nextFloat(random) < Rtheta) {
                        ray.origin = hitPoint;
                        ray.direction = _Material_specularReflection(record, sample, ray, random);
                        ray.origin += ray.direction * OFFSET;
                        ray.currentMaterial = ray.prevMaterial;
                        ray.currentBlock = ray.prevBlock;
                    } else {
                        throughput *= Material_translucentTransmission(sample, pAbsorb, transmissivityCap, fancierTranslucency);
                        ray.origin = hitPoint;
                        if (doRefraction) {
                            ray.direction = Material_refractDirection(record, ray, n1, n2);
                        }
                        ray.origin += ray.direction * OFFSET;
                    }
                }
            } else {
                throughput *= Material_translucentTransmission(sample, pAbsorb, transmissivityCap, fancierTranslucency);
                ray.origin = hitPoint + ray.direction * OFFSET;
            }

            if (!didSpecularBounce) {
                ray.flags |= RAY_INDIRECT;
            }
        } else {
            // A ray that misses while traveling through water ends black instead of
            // showing the sky (CPU parity — no sky underlay beneath the water world).
            if (ray.currentMaterial != 0 &&
                    Material_isWater(Material_get(scene.materialPalette, ray.currentMaterial))) {
                break;
            }
            intersectSky(skyTexture, sun, textureAtlas, scene.atmosphere, ray, &sample);
            if (depth == 0) {
                firstAlbedo = sample.color.xyz;
            }
            throughput *= sample.color.xyz;
            color += sample.emittance * throughput;
            break;
        }
    }

    // Uniform ground fog: distance-based blend toward the sky-tinted haze color.
    // Sun-independent so the fog never darkens when the sun is occluded, and identical
    // to the CPU's extinction+inscatter result when the sun is fully visible.
    if (scene.atmosphere.fogMode == 1 && airDistance > 0.0f) {
        float fogDensity = scene.atmosphere.uniformDensity * FOG_EXTINCTION_FACTOR;
        float fogFactor = 1.0f - exp(-airDistance * fogDensity);
        fogFactor = clamp(fogFactor, 0.0f, 1.0f);

        // Flat fog color (CPU parity): a sky-tinted haze makes far objects converge
        // toward the sky behind them, ghosting the skymap through geometry.
        float3 hazeColor = scene.atmosphere.fogColor;

        color = mix(color, hazeColor, fogFactor);
    }

    // Underwater visibility attenuation (CPU parity): exp(-waterDistance / visibility),
    // black when the visibility is zero.
    if (waterDistance > 0.0f) {
        if (scene.atmosphere.waterVisibility <= EPS) {
            color *= 0.0f;
        } else {
            color *= exp(-waterDistance / scene.atmosphere.waterVisibility);
        }
    }

    int spp = *bufferSpp;
    float3 bufferColor = vload3(gid, res);
    bufferColor = (bufferColor * spp + color) / (spp + 1);
    vstore3(bufferColor, gid, res);

    // The albedo/normal guides only carry edge anti-aliasing noise, so they converge
    // within GUIDE_SPP_CAP samples. Freeze them there — they are OIDN inputs only.
    if (spp < GUIDE_SPP_CAP) {
        float3 albedoColor = vload3(gid, albedoRes);
        float3 normalColor = vload3(gid, normalRes);
        albedoColor = (albedoColor * spp + firstAlbedo) / (spp + 1);
        normalColor = (normalColor * spp + firstNormal) / (spp + 1);
        vstore3(albedoColor, gid, albedoRes);
        vstore3(normalColor, gid, normalRes);
    }
}

__kernel void preview(
    __global const int* projectorType,
    __global const float* cameraSettings,

    __global const int* octreeDepth,
    __global const int* octreeData,
    __global const int* waterOctreeDepth,
    __global const int* waterOctreeData,

    __global const int* bPalette,
    __global const int* quadModels,
    __global const int* aabbModels,
    __global const int* waterModels,

    __global const int* worldBvhData,
    __global const int* actorBvhData,
    __global const int* bvhTrigs,

    image2d_array_t textureAtlas,
    __global const int* matPalette,
    __global const int* biomeMeta,
    __global const int* biomeGrid,
    __global const float* biomeGrass,
    __global const float* biomeFoliage,
    __global const float* biomeDryFoliage,
    __global const float* biomeWater,

    image2d_t skyTexture,
    __global const int* sunData,

    __global const int* canvasConfig,
    __global int* res
) {
    int gid = get_global_id(0);

    int px = gid % canvasConfig[0] + canvasConfig[4];
    int py = gid / canvasConfig[0] + canvasConfig[5];

    // Crosshairs?
    if ((px == canvasConfig[2] / 2 && (py >= canvasConfig[3] / 2 - 5 && py <= canvasConfig[3] / 2 + 5)) ||
        (py == canvasConfig[3] / 2 && (px >= canvasConfig[2] / 2 - 5 && px <= canvasConfig[2] / 2 + 5))) {
        res[gid] = 0xFFFFFFFF;
        return;
    }

    Scene scene;
    scene.materialPalette = MaterialPalette_new(matPalette);
    scene.octree = Octree_create(octreeData, *octreeDepth, (float)(1 << max(*octreeDepth, 10)));
    scene.waterOctree = Octree_create(waterOctreeData, *waterOctreeDepth, (float)(1 << max(*waterOctreeDepth, 10)));
    scene.worldBvh = Bvh_new(worldBvhData, bvhTrigs);
    scene.actorBvh = Bvh_new(actorBvhData, bvhTrigs);
    scene.blockPalette = BlockPalette_new(bPalette, quadModels, aabbModels, waterModels);
    scene.biome = BiomeColors_new(biomeMeta, biomeGrid, biomeGrass, biomeFoliage, biomeDryFoliage, biomeWater);
    scene.emitterGrid = EmitterGrid_new(bPalette, bPalette, bPalette, bPalette);
    scene.atmosphere = Atmosphere_empty();
    scene.drawDepth = 256;
    scene.emittersEnabled = false;
    scene.profile = false;
    scene.profileCounters = (__global int*)0;

    Sun sun = Sun_new(sunData);

    // Per-pixel seed so depth-of-field and pixel jitter vary spatially. With a constant
    // seed every pixel would get the identical lens offset and DOF would vanish.
    unsigned int randomState = gid * 2654435761u + 0x9E3779B9u;
    Random random = &randomState;
    Random_nextState(random);

    Ray ray = ray_to_camera(projectorType, cameraSettings, canvasConfig, gid, random);

    initialize_ray_medium(scene, &ray);
    ray.flags = RAY_PREVIEW;

    // March past low-alpha texels so translucent blocks (cherry leaves, tinted glass)
    // render as holes instead of opaque white-ish blobs.
    float3 color = (float3)(0.0f);
    bool hitAnything = false;
    for (int step = 0; step < 8; step++) {
        IntersectionRecord record = IntersectionRecord_new();
        MaterialSample sample;
        Material material;

        if (closestIntersect(scene, textureAtlas, ray, &record, &sample, &material)) {
            if (sample.color.w < 0.35f) {
                ray.currentMaterial = record.material;
                ray.currentBlock = record.block;
                ray.origin += ray.direction * (record.distance + OFFSET);
                continue;
            }
            float shading = dot(record.normal, (float3) (0.25, 0.866, 0.433));
            shading = fmax(0.3f, shading);
            color = sample.color.xyz * shading;
            hitAnything = true;
            break;
        } else {
            break;
        }
    }

    if (!hitAnything) {
        // Underwater misses end black (no sky underlay beneath the water world).
        if (ray.currentMaterial != 0 &&
                Material_isWater(Material_get(scene.materialPalette, ray.currentMaterial))) {
            color = (float3)(0.0f);
        } else {
            MaterialSample sample;
            intersectSky(skyTexture, sun, textureAtlas, scene.atmosphere, ray, &sample);
            color = sample.color.xyz;
        }
    }

    color = sqrt(color);
    int3 rgb = intFloorFloat3(clamp(color * 255.0f, 0.0f, 255.0f));
    res[gid] = 0xFF000000 | (rgb.x << 16) | (rgb.y << 8) | rgb.z;
}
