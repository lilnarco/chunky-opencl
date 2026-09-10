// This includes stuff regarding materials

#ifndef CHUNKYCLPLUGIN_MATERIAL_H
#define CHUNKYCLPLUGIN_MATERIAL_H

#include "../opencl.h"
#include "rt.h"
#include "textureAtlas.h"
#include "utils.h"
#include "constants.h"
#include "random.h"
#include "biome.h"

typedef struct {
    __global const int* palette;
} MaterialPalette;

MaterialPalette MaterialPalette_new(__global const int* palette) {
    MaterialPalette p;
    p.palette = palette;
    return p;
}

typedef struct {
    unsigned int flags;
    unsigned int tint;
    unsigned int textureSize;
    unsigned int color;
    unsigned int normal_emittance;
    unsigned int specular_metalness_roughness;
    unsigned int ior;
} Material;

Material Material_get(MaterialPalette self, int material) {
    Material m;
    m.flags = self.palette[material + 0];
    m.tint = self.palette[material + 1];
    m.textureSize = self.palette[material + 2];
    m.color = self.palette[material + 3];
    m.normal_emittance = self.palette[material + 4];
    m.specular_metalness_roughness = self.palette[material + 5];
    m.ior = self.palette[material + 6];
    return m;
}

typedef struct {
    float4 color;
    float emittance;
    float specular;
    float metalness;
    float roughness;
} MaterialSample;

bool Material_isRefractive(Material self);

bool Material_sample_mode(Material self, image2d_array_t atlas, float2 uv, bool allowTransparentHit, int3 worldPos, BiomeColors biome, MaterialSample* sample);

bool Material_sample(Material self, image2d_array_t atlas, float2 uv, int3 worldPos, BiomeColors biome, MaterialSample* sample) {
    return Material_sample_mode(self, atlas, uv, false, worldPos, biome, sample);
}

bool Material_sample_mode(Material self, image2d_array_t atlas, float2 uv, bool allowTransparentHit, int3 worldPos, BiomeColors biome, MaterialSample* sample) {
    // Color
    float4 color;
    if (self.flags & 0b00001)
        color = Atlas_read_uv(uv.x, uv.y, self.color, self.textureSize, atlas);
    else
        color = colorFromArgb(self.color);
    
    if (self.tint == 0xFE000000) {
        // Light block: CPU sets ray.color = (1,1,1,1) on hit; the quadratic emittance
        // mapping (color^2 * emittance) then matches the CPU contribution exactly.
        sample->color.xyz = 1.0;
        sample->color.w = 1.0;
    } else if (color.w > EPS) {
        sample->color = color;
    } else if (allowTransparentHit && Material_isRefractive(self)) {
        // Fully transparent texel on glass: keep the medium interface for refraction,
        // but do not render a visible back-face frame.
        sample->color = (float4)(1.0f, 1.0f, 1.0f, 0.0f);
    } else {
        // Fully transparent texel on a non-refractive translucent block (cherry leaves,
        // tinted glass, etc.): no hit, matching the CPU's alpha check in Block.intersect.
        return false;
    }

    // Tint
    switch (self.tint >> 24) {
        case 0xFF:
            sample->color *= colorFromArgb(self.tint);
            break;
        case 1:
            sample->color.xyz *= BiomeColors_getFoliage(biome, worldPos);
            break;
        case 2:
            sample->color.xyz *= BiomeColors_getGrass(biome, worldPos);
            break;
        case 3:
            sample->color.xyz *= BiomeColors_getWater(biome, worldPos);
            break;
        case 4:
            sample->color.xyz *= BiomeColors_getDryFoliage(biome, worldPos);
            break;
    }

    if ((self.flags & 0b01000) && (self.flags & 0b10000) == 0 &&
            sample->color.w > EPS && sample->color.w < 0.999f) {
        // Keep stained/partial glass surface details readable without affecting fully
        // transparent texels that should still disappear on the back face.
        sample->color.w = fmin(1.0f, sample->color.w + 0.15f);
    }

    // (Normal) emittance — packed as full float bits to preserve intensities > 1.
    if (self.flags & 0b00010)
        sample->emittance = Atlas_read_uv(uv.x, uv.y, self.normal_emittance, self.textureSize, atlas).w;
    else
        sample->emittance = as_float(self.normal_emittance);

    // specular, metalness, roughness
    if (self.flags & 0b00100) {
        float3 smr = Atlas_read_uv(uv.x, uv.y, self.specular_metalness_roughness, self.textureSize, atlas).xyz;
        sample->specular = smr.x;
        sample->metalness = smr.y;
        sample->roughness = smr.z;
    } else {
        sample->specular = (self.specular_metalness_roughness & 0xFF) / 255.0;
        sample->metalness = ((self.specular_metalness_roughness >> 8) & 0xFF) / 255.0;
        sample->roughness = ((self.specular_metalness_roughness >> 16) & 0xFF) / 255.0;
    }
    
    return true;
}

bool Material_isRefractive(Material self) {
    return (self.flags & 0b01000) != 0;
}

bool Material_isOpaque(Material self) {
    return (self.flags & 0b10000) != 0;
}

bool Material_isWater(Material self) {
    return (self.flags & 0b100000) != 0;
}

// Opaque, non-emissive materials stop shadow rays immediately; their texels never need
// to be sampled, so shadow rays can skip the texture read for them.
bool Material_isOpaqueOccluder(Material self) {
    return Material_isOpaque(self) && as_float(self.normal_emittance) <= EPS;
}

void MaterialSample_opaque(MaterialSample* sample) {
    sample->color = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
    sample->emittance = 0.0f;
    sample->specular = 0.0f;
    sample->metalness = 0.0f;
    sample->roughness = 0.0f;
}

float Material_ior(Material self) {
    return as_float(self.ior);
}

float3 _Material_diffuseReflection(IntersectionRecord record, Random random) {
    float x1 = Random_nextFloat(random);
    float x2 = Random_nextFloat(random);
    // Stage 2: natives — uniform hemisphere map; small radial/angular error
    // averages out over SPP.
    float r = native_sqrt(x1);
    float theta = 2 * M_PI_F * x2;

    float tx = r * native_cos(theta);
    float ty = r * native_sin(theta);
    float tz = native_sqrt(1 - x1);

    // Transform from tangent space to world space
    float xx, xy, xz;
    float ux, uy, uz;
    float vx, vy, vz;

    if (fabs(record.normal.x) > 0.1) {
        xx = 0;
        xy = 1;
    } else {
        xx = 1;
        xy = 0;
    }
    xz = 0;

    ux = xy * record.normal.z - xz * record.normal.y;
    uy = xz * record.normal.x - xx * record.normal.z;
    uz = xx * record.normal.y - xy * record.normal.x;

    // Stage 2: native_rsqrt — textbook reciprocal normalize.
    r = native_rsqrt(ux*ux + uy*uy + uz*uz);

    ux *= r;
    uy *= r;
    uz *= r;

    vx = uy * record.normal.z - uz * record.normal.y;
    vy = uz * record.normal.x - ux * record.normal.z;
    vz = ux * record.normal.y - uy * record.normal.x;

    return (float3) (
        ux * tx + vx * ty + record.normal.x * tz,
        uy * tx + vy * ty + record.normal.y * tz,
        uz * tx + vz * ty + record.normal.z * tz
    );
}

float3 _Material_specularReflection(IntersectionRecord record, MaterialSample sample, Ray ray, Random random) {
    float3 direction = ray.direction + (record.normal * (-2 * dot(ray.direction, record.normal)));

    if (sample.roughness > 0) {
        float3 diffuseDirection = _Material_diffuseReflection(record, random);
        diffuseDirection *= sample.roughness;
        direction = diffuseDirection + direction * (1 - sample.roughness);
    }

    if (signbit(dot(record.normal, direction)) == signbit(dot(record.normal, ray.direction))) {
        float factor = copysign(dot(record.normal, ray.direction), -EPS - dot(record.normal, direction));
        direction += factor * record.normal;
    }

    // Stage 2: fast normalize idiom — input is already ~unit (reflection of a
    // unit vector plus a small roughness perturb).
    return direction * native_rsqrt(dot(direction, direction));
}

float3 Material_refractDirection(IntersectionRecord record, Ray ray, float n1, float n2) {
    float n1n2 = n1 / n2;
    float cosTheta = -dot(record.normal, ray.direction);
    float t2 = sqrt(fmax(0.0f, 1 - n1n2 * n1n2 * (1 - cosTheta * cosTheta)));
    float3 direction;

    if (cosTheta > 0) {
        direction = n1n2 * ray.direction + (n1n2 * cosTheta - t2) * record.normal;
    } else {
        direction = n1n2 * ray.direction - (-n1n2 * cosTheta - t2) * record.normal;
    }

    direction = normalize(direction);
    if (signbit(dot(record.normal, direction)) != signbit(dot(record.normal, ray.direction))) {
        float factor = copysign(dot(record.normal, ray.direction), -EPS - dot(record.normal, direction));
        direction += factor * record.normal;
        direction = normalize(direction);
    }
    return direction;
}

float3 Material_translucentTransmission(MaterialSample sample, float absorption, float transmissivityCap, bool fancierTranslucency) {
    if (!fancierTranslucency) {
        float3 rgbTrans = (float3) (1 - absorption);
        return rgbTrans + absorption * sample.color.xyz;
    }

    float colorTrans = (sample.color.x + sample.color.y + sample.color.z) / 3.0f;
    float shouldTrans = 1.0f - absorption;
    float3 rgbTrans = (float3) (shouldTrans);

    if (colorTrans > 0) {
        rgbTrans = sample.color.xyz * (shouldTrans / colorTrans);
    }

    float maxTrans = fmax(rgbTrans.x, fmax(rgbTrans.y, rgbTrans.z));
    if (maxTrans > transmissivityCap && maxTrans > 0) {
        rgbTrans *= transmissivityCap / maxTrans;
    }
    return rgbTrans;
}

typedef struct {
    float3 direction;
    float3 spectrum;
    bool specular;
} MaterialPdfSample;

MaterialPdfSample Material_samplePdf(Material self, IntersectionRecord record, MaterialSample sample, Ray ray, Random random) {
    MaterialPdfSample out;

    if (sample.metalness > 0 && sample.metalness > Random_nextFloat(random)) {
        // Metal reflection
        out.direction = _Material_specularReflection(record, sample, ray, random);
        out.spectrum = sample.color.xyz;
        out.specular = true;
        return out;
    } else if (sample.specular > 0 && sample.specular > Random_nextFloat(random)) {
        // Specular reflection
        out.direction = _Material_specularReflection(record, sample, ray, random);
        out.spectrum = 1;
        out.specular = true;
        return out;
    } else {
        // Diffuse reflection
        out.direction = _Material_diffuseReflection(record, random);
        out.spectrum = sample.color.xyz;
        out.specular = false;
        return out;
    }
}

#endif
