#include "kernel.h"
#include "sky.h"

float4 getDirectLightAttenuation(
        Scene scene,
        image2d_array_t textureAtlas,
        Ray ray,
        bool strictDirectLight
) {
    float4 attenuation = (float4) (1.0f, 1.0f, 1.0f, 1.0f);

    while (attenuation.w > 0.0f) {
        ray.origin += ray.direction * OFFSET;

        IntersectionRecord record = IntersectionRecord_new();
        MaterialSample sample;
        Material material;

        if (!closestIntersect(scene, textureAtlas, ray, &record, &sample, &material)) {
            break;
        }

        ray.prevMaterial = ray.currentMaterial;
        ray.prevBlock = ray.currentBlock;
        ray.currentMaterial = record.material;
        ray.currentBlock = record.block;

        float mult = 1.0f - sample.color.w;
        attenuation.x *= sample.color.x * sample.color.w + mult;
        attenuation.y *= sample.color.y * sample.color.w + mult;
        attenuation.z *= sample.color.z * sample.color.w + mult;
        attenuation.w *= mult;

        if (strictDirectLight) {
            Material prevMat = Material_get(scene.materialPalette, ray.prevMaterial);
            Material currentMat = Material_get(scene.materialPalette, ray.currentMaterial);
            if (fabs(Material_ior(prevMat) - Material_ior(currentMat)) >= EPS) {
                attenuation.w = 0.0f;
            }
        }

        ray.origin += ray.direction * record.distance;
    }

    return attenuation;
}

void intersectSky(image2d_t skyTexture, Sun sun, image2d_array_t atlas, Atmosphere atmosphere, Ray ray, MaterialSample* sample) {
    Sky_intersect(skyTexture, ray, sample);
    Sun_intersect(sun, atlas, ray, sample);

    // Uniform fog: blend the sky color toward the fog color, strongest at the horizon.
    if (atmosphere.fogMode == 1 && atmosphere.uniformDensity > 0.0f) {
        float fog;
        if (ray.direction.y > 0.0f) {
            float dy = 1.0f - ray.direction.y;
            fog = dy * dy;
        } else {
            fog = 1.0f;
        }
        fog *= atmosphere.skyFogDensity;
        fog = clamp(fog, 0.0f, 1.0f);
        sample->color.xyz = mix(sample->color.xyz, atmosphere.fogColor, fog);
    }
}
