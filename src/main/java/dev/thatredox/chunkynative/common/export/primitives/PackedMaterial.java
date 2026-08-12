package dev.thatredox.chunkynative.common.export.primitives;

import dev.thatredox.chunkynative.common.export.texture.AbstractTextureLoader;
import dev.thatredox.chunkynative.common.export.Packer;

import it.unimi.dsi.fastutil.ints.IntArrayList;
import se.llbit.chunky.PersistentSettings;
import se.llbit.chunky.model.Tint;
import se.llbit.chunky.resources.Texture;
import se.llbit.chunky.world.Material;
import se.llbit.log.Log;
import se.llbit.math.ColorUtil;

public class PackedMaterial implements Packer {
    private static final float AIR_IOR = 1.000293f;
    /**
     * The size of a packed material in 32 bit words.
     */
    public static final int MATERIAL_DWORD_SIZE = 7;

    public static final int FLAG_HAS_COLOR_TEXTURE = 0b00001;
    public static final int FLAG_HAS_NORMAL_EMITTANCE_TEXTURE = 0b00010;
    public static final int FLAG_HAS_SPECULAR_METALNESS_ROUGHNESS_TEXTURE = 0b00100;
    public static final int FLAG_REFRACTIVE = 0b01000;
    public static final int FLAG_OPAQUE = 0b10000;
    public static final int FLAG_WATER = 0b100000;

    public final boolean hasColorTexture;
    public final boolean hasNormalEmittanceTexture;
    public final boolean hasSpecularMetalnessRoughnessTexture;
    public final boolean refractive;
    public final boolean opaque;
    public final boolean water;

    public final int blockTint;

    public final long colorTexture;
    public final int normalEmittanceTexture;
    public final int specularMetalnessRoughnessTexture;
    public final int ior;

    public static PackedMaterial air() {
        return new PackedMaterial();
    }

    /**
     * Opaque white material used by the GPU cloud renderer. It is reserved at palette
     * index 1 so cloud intersection records can reference it by a fixed id.
     */
    public static PackedMaterial cloud() {
        return new PackedMaterial(true);
    }

    private PackedMaterial() {
        this.hasColorTexture = false;
        this.hasNormalEmittanceTexture = false;
        this.hasSpecularMetalnessRoughnessTexture = false;
        this.refractive = false;
        this.opaque = false;
        this.water = false;
        this.blockTint = 0;
        this.colorTexture = 0;
        this.normalEmittanceTexture = 0;
        this.specularMetalnessRoughnessTexture = 0;
        this.ior = Float.floatToIntBits(1.000293f);
    }

    private PackedMaterial(boolean isCloud) {
        this.hasColorTexture = false;
        this.hasNormalEmittanceTexture = false;
        this.hasSpecularMetalnessRoughnessTexture = false;
        this.refractive = false;
        this.opaque = true;
        this.water = false;
        this.blockTint = 0;
        this.colorTexture = 0xFFFFFFFFL;
        this.normalEmittanceTexture = Float.floatToIntBits(0.0f);
        this.specularMetalnessRoughnessTexture = 0;
        this.ior = Float.floatToIntBits(1.0f);
    }

    public PackedMaterial(Texture texture, Tint tint, Material material, AbstractTextureLoader texturePalette) {
        this(texture, getTint(tint), material.emittance, material.specular, material.metalness, material.roughness,
                material.refractive, material.opaque, material.ior, texturePalette);
    }

    public PackedMaterial(Material material, Tint tint, AbstractTextureLoader texMap) {
        this(material.texture, getTint(tint), material.emittance, material.specular, material.metalness, material.roughness,
                material.refractive, material.opaque, material.ior, texMap);
    }

    public PackedMaterial(Texture texture, Tint tint, float emittance, float specular, float metalness, float roughness, AbstractTextureLoader texMap) {
        this(texture, getTint(tint), emittance, specular, metalness, roughness, false, false, 1.000293f, texMap);
    }

    private static int getTint(Tint tint) {
        if (tint != null) {
            switch (tint.type) {
                default:
                    Log.warn("Unsupported tint type " + tint.type);
                case NONE:
                    return 0;
                case CONSTANT:
                    return ColorUtil.getRGB(tint.tint) | 0xFF000000;
                case BIOME_FOLIAGE:
                    return 1 << 24;
                case BIOME_DRY_FOLIAGE:
                    return 4 << 24;
                case BIOME_GRASS:
                    return 2 << 24;
                case BIOME_WATER:
                    return 3 << 24;
            }
        } else {
            return 0;
        }
    }

    public PackedMaterial(Texture texture, int blockTint, float emittance, float specular, float metalness, float roughness, AbstractTextureLoader texMap) {
        this(texture, blockTint, emittance, specular, metalness, roughness, false, false, 1.000293f, texMap);
    }

    public PackedMaterial(Texture texture, int blockTint, float emittance, float specular, float metalness, float roughness,
                          boolean refractive, boolean opaque, float ior, AbstractTextureLoader texMap) {
        // Transparent/refractive materials need per-pixel alpha; averaging the texture
        // collapses glass into an effectively opaque surface.
        boolean effectivelyRefractive = refractive || (!opaque && Math.abs(ior - AIR_IOR) > 1.0e-4f);
        this.hasColorTexture = effectivelyRefractive || !opaque || !PersistentSettings.getSingleColorTextures();
        this.hasNormalEmittanceTexture = false;
        this.hasSpecularMetalnessRoughnessTexture = false;
        this.refractive = effectivelyRefractive;
        this.opaque = opaque;
        this.water = texture == Texture.water;
        this.blockTint = blockTint;
        this.colorTexture = this.hasColorTexture ? texMap.get(texture).get() : texture.getAvgColor();
        this.normalEmittanceTexture = Float.floatToIntBits(emittance);
        this.specularMetalnessRoughnessTexture = (int) (specular * 255.0) |
                ((int) (metalness * 255.0) << 8) |
                ((int) (roughness * 255.0) << 16);
        this.ior = Float.floatToIntBits(ior);
    }

    /**
     * Materials are packed into 6 consecutive integers:
     * 0: Flags - 0b00001 = has color texture
     *            0b00010 = has normal emittance texture
     *            0b00100 = has specular metalness roughness texture
     *            0b01000 = refractive
     *            0b10000 = opaque
     *            0b100000 = water
     * 1: Block tint - the top 8 bits control which type of tint:
     *                 0xFF = lower 24 bits should be interpreted as RGB color
     *                 0x01 = foliage color
     *                 0x02 = grass color
     *                 0x03 = water color
     *                 0x04 = dry foliage color
     * 2 & 3: Color texture reference
     * 4: Emittance as float bits (no 8-bit quantization, supports intensities > 1).
     * 5: First 8 bits represent the specularness. Next 8 bits represent the metalness. Next 8 bits represent the roughness.
     * 6: IoR as float bits.
     */
    @Override
    public IntArrayList pack() {
        IntArrayList packed = new IntArrayList(7);
        packed.add((this.hasColorTexture ? FLAG_HAS_COLOR_TEXTURE : 0) |
                   (this.hasNormalEmittanceTexture ? FLAG_HAS_NORMAL_EMITTANCE_TEXTURE : 0) |
                   (this.hasSpecularMetalnessRoughnessTexture ? FLAG_HAS_SPECULAR_METALNESS_ROUGHNESS_TEXTURE : 0) |
                   (this.refractive ? FLAG_REFRACTIVE : 0) |
                   (this.opaque ? FLAG_OPAQUE : 0) |
                   (this.water ? FLAG_WATER : 0));
        packed.add(this.blockTint);
        packed.add((int) (this.colorTexture >>> 32));
        packed.add((int) this.colorTexture);
        packed.add(this.normalEmittanceTexture);
        packed.add(this.specularMetalnessRoughnessTexture);
        packed.add(this.ior);
        return packed;
    }
}
