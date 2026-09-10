package dev.thatredox.chunkynative.opencl.renderer;

import dev.thatredox.chunkynative.common.export.AbstractSceneLoader;
import dev.thatredox.chunkynative.common.export.ResourcePalette;
import dev.thatredox.chunkynative.common.export.models.PackedAabbModel;
import dev.thatredox.chunkynative.common.export.models.PackedQuadModel;
import dev.thatredox.chunkynative.common.export.models.PackedTriangleModel;
import dev.thatredox.chunkynative.common.export.models.PackedWaterModel;
import dev.thatredox.chunkynative.common.export.primitives.PackedBlock;
import dev.thatredox.chunkynative.common.export.primitives.PackedMaterial;
import dev.thatredox.chunkynative.common.export.primitives.PackedSun;
import dev.thatredox.chunkynative.common.export.texture.AbstractTextureLoader;
import dev.thatredox.chunkynative.common.state.SkyState;
import dev.thatredox.chunkynative.opencl.context.ClContext;
import dev.thatredox.chunkynative.opencl.context.ContextManager;
import dev.thatredox.chunkynative.opencl.renderer.export.ClPackedResourcePalette;
import dev.thatredox.chunkynative.opencl.renderer.export.ClTextureLoader;
import dev.thatredox.chunkynative.opencl.renderer.scene.ClSky;
import dev.thatredox.chunkynative.opencl.util.ClIntBuffer;
import dev.thatredox.chunkynative.opencl.util.ClMemory;
import dev.thatredox.chunkynative.util.FunctionCache;
import dev.thatredox.chunkynative.util.Reflection;
import se.llbit.chunky.renderer.ResetReason;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.chunky.world.ChunkPosition;
import se.llbit.chunky.world.biome.Biomes;
import se.llbit.math.Grid;
import se.llbit.math.Vector3i;
import org.jocl.Pointer;
import org.jocl.Sizeof;

import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collection;
import java.lang.ref.WeakReference;

import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.clCreateBuffer;

public class ClSceneLoader extends AbstractSceneLoader {
    protected final FunctionCache<int[], ClIntBuffer> clWorldBvh;
    protected final FunctionCache<int[], ClIntBuffer> clActorBvh;
    protected final FunctionCache<PackedSun, ClIntBuffer> clPackedSun;
    protected ClSky clSky = null;
    protected SkyState skyState = null;

    protected ClIntBuffer octreeData = null;
    protected ClIntBuffer octreeDepth = null;
    protected ClIntBuffer waterOctreeData = null;
    protected ClIntBuffer waterOctreeDepth = null;
    protected ClIntBuffer emitterGridMeta = null;
    protected ClIntBuffer emitterGridCells = null;
    protected ClIntBuffer emitterGridIndexes = null;
    protected ClIntBuffer emitterGridEmitters = null;
    protected WeakReference<Grid> prevEmitterGrid = new WeakReference<>(null, null);
    protected int[] prevEmitterGridBlockMapping = null;
    private int waterMaterialId = 0;
    protected ClIntBuffer biomeMeta = null;
    protected ClIntBuffer biomeGrid = null;
    protected ClMemory biomeGrass = null;
    protected ClMemory biomeFoliage = null;
    protected ClMemory biomeDryFoliage = null;
    protected ClMemory biomeWater = null;
    private final ClContext context;

    public ClSceneLoader(ClContext context) {
        this.context = context;
        this.clWorldBvh = new FunctionCache<>(i -> new ClIntBuffer(i, context), ClIntBuffer::close, null);
        this.clActorBvh = new FunctionCache<>(i -> new ClIntBuffer(i, context), ClIntBuffer::close, null);
        this.clPackedSun = new FunctionCache<>(i -> new ClIntBuffer(i, context), ClIntBuffer::close, null);
    }

    @Override
    public boolean ensureLoad(Scene scene) {
        return this.ensureLoad(scene, clSky == null);
    }

    @Override
    public boolean load(int modCount, ResetReason resetReason, Scene scene) {
        boolean loadSuccess = super.load(modCount, resetReason, scene);
        if (this.modCount != modCount) {
            SkyState newSky = new SkyState(scene.sky(), scene.sun());
            if (!newSky.equals(skyState)) {
                if (clSky != null) clSky.close();
                clSky = new ClSky(scene, context);
                skyState = newSky;
                packedSun = new PackedSun(scene.sun(), getTexturePalette());
            }
            loadEmitterGrid(scene);
            loadBiomeColors(scene);
            // The first packed water model's material is shared by all water (the material
            // palette dedups by content), so it represents the water surface material.
            if (waterPalette instanceof ClPackedResourcePalette) {
                ClPackedResourcePalette<PackedWaterModel> packed = (ClPackedResourcePalette<PackedWaterModel>) waterPalette;
                waterMaterialId = packed.size() > 0 ? packed.getData(0) : 0;
            }
        }
        return loadSuccess;
    }

    /** Material palette index of the water material (0 = air when no water is present). */
    public int getWaterMaterialId() {
        return waterMaterialId;
    }

    /** Whether any water blocks exist in the loaded scene (water palette non-empty). */
    public boolean hasWater() {
        return waterPalette instanceof ClPackedResourcePalette
                && ((ClPackedResourcePalette<?>) waterPalette).size() > 0;
    }

    private void loadEmitterGrid(Scene scene) {
        Grid grid = scene.getEmitterGrid();
        // The grid only changes when the world/palette changes, not on camera moves or
        // lighting edits. Cache the GPU buffers on the Grid instance + block mapping so
        // ordinary scene resets don't rebuild and re-upload them every frame.
        if (grid == prevEmitterGrid.get() && prevEmitterGridBlockMapping == blockMapping && emitterGridMeta != null) {
            return;
        }
        prevEmitterGrid = new WeakReference<>(grid, null);
        prevEmitterGridBlockMapping = blockMapping;

        if (emitterGridMeta != null) emitterGridMeta.close();
        if (emitterGridCells != null) emitterGridCells.close();
        if (emitterGridIndexes != null) emitterGridIndexes.close();
        if (emitterGridEmitters != null) emitterGridEmitters.close();

        if (grid == null || blockMapping == null) {
            emitterGridMeta = new ClIntBuffer(new int[] {0, 0, 0, 0, 0, 0, 0}, context);
            emitterGridCells = new ClIntBuffer(new int[] {0, 0}, context);
            emitterGridIndexes = new ClIntBuffer(new int[] {0}, context);
            emitterGridEmitters = new ClIntBuffer(new int[] {0, 0, 0, 0}, context);
            return;
        }

        int cellSize = Reflection.getFieldValue(grid, "cellSize", Integer.class);
        int offsetX = Reflection.getFieldValue(grid, "offsetX", Integer.class);
        int sizeX = Reflection.getFieldValue(grid, "sizeX", Integer.class);
        int offsetY = Reflection.getFieldValue(grid, "offsetY", Integer.class);
        int sizeY = Reflection.getFieldValue(grid, "sizeY", Integer.class);
        int offsetZ = Reflection.getFieldValue(grid, "offsetZ", Integer.class);
        int sizeZ = Reflection.getFieldValue(grid, "sizeZ", Integer.class);
        int[] constructedGrid = Reflection.getFieldValue(grid, "constructedGrid", int[].class);
        int[] positionIndexes = Reflection.getFieldValue(grid, "positionIndexes", int[].class);
        @SuppressWarnings("unchecked")
        List<Grid.EmitterPosition> positions = Reflection.getFieldValue(grid, "emitterPositions", List.class);

        int[] emitters = new int[Math.max(positions.size(), 1) * 4];
        for (int i = 0; i < positions.size(); i++) {
            Grid.EmitterPosition pos = positions.get(i);
            int paletteIndex = scene.getPalette().getPalette().indexOf(pos.block);
            int packedBlock = paletteIndex >= 0 && paletteIndex < blockMapping.length ? blockMapping[paletteIndex] : 0;
            emitters[i * 4] = pos.x;
            emitters[i * 4 + 1] = pos.y;
            emitters[i * 4 + 2] = pos.z;
            emitters[i * 4 + 3] = packedBlock;
        }

        emitterGridMeta = new ClIntBuffer(new int[] {
                cellSize,
                offsetX, sizeX,
                offsetY, sizeY,
                offsetZ, sizeZ
        }, context);
        emitterGridCells = new ClIntBuffer(constructedGrid == null || constructedGrid.length == 0 ? new int[] {0, 0} : constructedGrid, context);
        emitterGridIndexes = new ClIntBuffer(positionIndexes == null || positionIndexes.length == 0 ? new int[] {0} : positionIndexes, context);
        emitterGridEmitters = new ClIntBuffer(emitters, context);
    }

    private void loadBiomeColors(Scene scene) {
        if (biomeMeta != null) biomeMeta.close();
        if (biomeGrid != null) biomeGrid.close();
        if (biomeGrass != null) biomeGrass.close();
        if (biomeFoliage != null) biomeFoliage.close();
        if (biomeDryFoliage != null) biomeDryFoliage.close();
        if (biomeWater != null) biomeWater.close();

        float[] defaultGrass = Biomes.biomesPrePalette[0].grassColorLinear;
        float[] defaultFoliage = Biomes.biomesPrePalette[0].foliageColorLinear;
        float[] defaultDryFoliage = Biomes.biomesPrePalette[0].dryFoliageColorLinear;
        float[] defaultWater = Biomes.biomesPrePalette[0].waterColorLinear;

        int[] meta = new int[18];
        Vector3i origin = scene.getOrigin();
        meta[4] = origin.x;
        meta[5] = origin.z;

        meta[6] = Float.floatToIntBits(defaultGrass[0]);
        meta[7] = Float.floatToIntBits(defaultGrass[1]);
        meta[8] = Float.floatToIntBits(defaultGrass[2]);
        meta[9] = Float.floatToIntBits(defaultFoliage[0]);
        meta[10] = Float.floatToIntBits(defaultFoliage[1]);
        meta[11] = Float.floatToIntBits(defaultFoliage[2]);
        meta[12] = Float.floatToIntBits(defaultDryFoliage[0]);
        meta[13] = Float.floatToIntBits(defaultDryFoliage[1]);
        meta[14] = Float.floatToIntBits(defaultDryFoliage[2]);
        meta[15] = Float.floatToIntBits(defaultWater[0]);
        meta[16] = Float.floatToIntBits(defaultWater[1]);
        meta[17] = Float.floatToIntBits(defaultWater[2]);

        Collection<ChunkPosition> sceneChunks = scene.getChunks();
        boolean useBiome = scene.biomeColorsEnabled();
        if (!useBiome || sceneChunks == null || sceneChunks.isEmpty()) {
            biomeMeta = new ClIntBuffer(meta, context);
            biomeGrid = new ClIntBuffer(new int[] {0}, context);
            biomeGrass = createFloatBuffer(new float[] {0});
            biomeFoliage = createFloatBuffer(new float[] {0});
            biomeDryFoliage = createFloatBuffer(new float[] {0});
            biomeWater = createFloatBuffer(new float[] {0});
            return;
        }

        List<ChunkPosition> chunks = new ArrayList<>(sceneChunks);
        int minChunkX = Integer.MAX_VALUE;
        int maxChunkX = Integer.MIN_VALUE;
        int minChunkZ = Integer.MAX_VALUE;
        int maxChunkZ = Integer.MIN_VALUE;
        for (ChunkPosition pos : chunks) {
            minChunkX = Math.min(minChunkX, pos.x);
            maxChunkX = Math.max(maxChunkX, pos.x);
            minChunkZ = Math.min(minChunkZ, pos.z);
            maxChunkZ = Math.max(maxChunkZ, pos.z);
        }

        int sizeX = maxChunkX - minChunkX + 1;
        int sizeZ = maxChunkZ - minChunkZ + 1;
        if (sizeX <= 0 || sizeZ <= 0) {
            biomeMeta = new ClIntBuffer(meta, context);
            biomeGrid = new ClIntBuffer(new int[] {0}, context);
            biomeGrass = createFloatBuffer(new float[] {0});
            biomeFoliage = createFloatBuffer(new float[] {0});
            biomeDryFoliage = createFloatBuffer(new float[] {0});
            biomeWater = createFloatBuffer(new float[] {0});
            return;
        }

        meta[0] = minChunkX;
        meta[1] = minChunkZ;
        meta[2] = sizeX;
        meta[3] = sizeZ;

        int[] grid = new int[sizeX * sizeZ];
        Arrays.fill(grid, -1);

        int chunkCount = chunks.size();
        int perChunk = 16 * 16 * 3;
        float[] grass = new float[chunkCount * perChunk];
        float[] foliage = new float[chunkCount * perChunk];
        float[] dryFoliage = new float[chunkCount * perChunk];
        float[] water = new float[chunkCount * perChunk];

        for (int i = 0; i < chunkCount; i++) {
            ChunkPosition pos = chunks.get(i);
            int gx = pos.x - minChunkX;
            int gz = pos.z - minChunkZ;
            if (gx >= 0 && gx < sizeX && gz >= 0 && gz < sizeZ) {
                grid[gz * sizeX + gx] = i;
            }

            int base = i * perChunk;
            int worldBaseX = pos.x << 4;
            int worldBaseZ = pos.z << 4;
            for (int z = 0; z < 16; z++) {
                int worldZ = worldBaseZ + z;
                int octreeZ = worldZ - origin.z;
                for (int x = 0; x < 16; x++) {
                    int worldX = worldBaseX + x;
                    int octreeX = worldX - origin.x;
                    int offset = base + (z * 16 + x) * 3;

                    float[] g = scene.getGrassColor(octreeX, 0, octreeZ);
                    grass[offset] = g[0];
                    grass[offset + 1] = g[1];
                    grass[offset + 2] = g[2];

                    float[] f = scene.getFoliageColor(octreeX, 0, octreeZ);
                    foliage[offset] = f[0];
                    foliage[offset + 1] = f[1];
                    foliage[offset + 2] = f[2];

                    float[] d = scene.getDryFoliageColor(octreeX, 0, octreeZ);
                    dryFoliage[offset] = d[0];
                    dryFoliage[offset + 1] = d[1];
                    dryFoliage[offset + 2] = d[2];

                    float[] w = scene.getWaterColor(octreeX, 0, octreeZ);
                    water[offset] = w[0];
                    water[offset + 1] = w[1];
                    water[offset + 2] = w[2];
                }
            }
        }

        biomeMeta = new ClIntBuffer(meta, context);
        biomeGrid = new ClIntBuffer(grid, context);
        biomeGrass = createFloatBuffer(grass);
        biomeFoliage = createFloatBuffer(foliage);
        biomeDryFoliage = createFloatBuffer(dryFoliage);
        biomeWater = createFloatBuffer(water);
    }

    private ClMemory createFloatBuffer(float[] data) {
        if (data.length == 0) {
            data = new float[] {0.0f};
        }
        return new ClMemory(clCreateBuffer(context.context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                (long) Sizeof.cl_float * data.length,
                Pointer.to(data), null));
    }

    private int[] mapOctree(int[] octree, int[] blockMapping) {
        return Arrays.stream(octree)
                .map(i -> i > 0 || -i >= blockMapping.length ? i : -blockMapping[-i])
                .toArray();
    }

    @Override
    protected boolean loadWorldOctree(int[] octree, int depth, int[] blockMapping, ResourcePalette<PackedBlock> blockPalette) {
        if (octreeData != null) octreeData.close();
        if (octreeDepth != null) octreeDepth.close();

        int[] mappedOctree = mapOctree(octree, blockMapping);
        octreeData = new ClIntBuffer(mappedOctree, context);
        octreeDepth = new ClIntBuffer(depth, context);
        return true;
    }

    @Override
    protected boolean loadWaterOctree(int[] octree, int depth, int[] blockMapping, ResourcePalette<PackedBlock> blockPalette) {
        if (waterOctreeData != null) waterOctreeData.close();
        if (waterOctreeDepth != null) waterOctreeDepth.close();

        int[] mappedOctree = mapOctree(octree, blockMapping);
        waterOctreeData = new ClIntBuffer(mappedOctree, context);
        waterOctreeDepth = new ClIntBuffer(depth, context);
        return true;
    }

    @Override
    protected AbstractTextureLoader createTextureLoader() {
        return new ClTextureLoader(context);
    }

    @Override
    protected ResourcePalette<PackedBlock> createBlockPalette() {
        return new ClPackedResourcePalette<>(context);
    }

    @Override
    protected ResourcePalette<PackedMaterial> createMaterialPalette() {
        return new ClPackedResourcePalette<>(context);
    }

    @Override
    protected ResourcePalette<PackedAabbModel> createAabbModelPalette() {
        return new ClPackedResourcePalette<>(context);
    }

    @Override
    protected ResourcePalette<PackedQuadModel> createQuadModelPalette() {
        return new ClPackedResourcePalette<>(context);
    }

    @Override
    protected ResourcePalette<PackedWaterModel> createWaterModelPalette() {
        return new ClPackedResourcePalette<>(context);
    }

    @Override
    protected ResourcePalette<PackedTriangleModel> createTriangleModelPalette() {
        return new ClPackedResourcePalette<>(context);
    }

    public ClIntBuffer getOctreeData() {
        assert octreeData != null;
        return octreeData;
    }

    public ClIntBuffer getOctreeDepth() {
        assert octreeDepth != null;
        return octreeDepth;
    }

    public ClIntBuffer getWaterOctreeData() {
        assert waterOctreeData != null;
        return waterOctreeData;
    }

    public ClIntBuffer getWaterOctreeDepth() {
        assert waterOctreeDepth != null;
        return waterOctreeDepth;
    }

    public ClTextureLoader getTexturePalette() {
        assert texturePalette instanceof ClTextureLoader;
        return (ClTextureLoader) texturePalette;
    }

    public ClPackedResourcePalette<PackedBlock> getBlockPalette() {
        assert blockPalette instanceof ClPackedResourcePalette;
        return (ClPackedResourcePalette<PackedBlock>) blockPalette;
    }

    public ClPackedResourcePalette<PackedMaterial> getMaterialPalette() {
        assert materialPalette.palette instanceof ClPackedResourcePalette;
        return (ClPackedResourcePalette<PackedMaterial>) materialPalette.palette;
    }

    public ClPackedResourcePalette<PackedAabbModel> getAabbPalette() {
        assert aabbPalette instanceof ClPackedResourcePalette;
        return (ClPackedResourcePalette<PackedAabbModel>) aabbPalette;
    }

    public ClPackedResourcePalette<PackedQuadModel> getQuadPalette() {
        assert quadPalette instanceof ClPackedResourcePalette;
        return (ClPackedResourcePalette<PackedQuadModel>) quadPalette;
    }

    public ClPackedResourcePalette<PackedWaterModel> getWaterPalette() {
        assert waterPalette instanceof ClPackedResourcePalette;
        return (ClPackedResourcePalette<PackedWaterModel>) waterPalette;
    }

    public ClPackedResourcePalette<PackedTriangleModel> getTrigPalette() {
        assert trigPalette instanceof ClPackedResourcePalette;
        return (ClPackedResourcePalette<PackedTriangleModel>) trigPalette;
    }

    public ClIntBuffer getWorldBvh() {
        return clWorldBvh.apply(this.worldBvh);
    }

    public ClIntBuffer getActorBvh() {
        return clActorBvh.apply(this.actorBvh);
    }

    public ClSky getSky() {
        assert clSky != null;
        return clSky;
    }

    public ClIntBuffer getSun() {
        return clPackedSun.apply(packedSun);
    }

    public ClIntBuffer getEmitterGridMeta() {
        assert emitterGridMeta != null;
        return emitterGridMeta;
    }

    public ClIntBuffer getEmitterGridCells() {
        assert emitterGridCells != null;
        return emitterGridCells;
    }

    public ClIntBuffer getEmitterGridIndexes() {
        assert emitterGridIndexes != null;
        return emitterGridIndexes;
    }

    public ClIntBuffer getEmitterGridEmitters() {
        assert emitterGridEmitters != null;
        return emitterGridEmitters;
    }

    public ClIntBuffer getBiomeMeta() {
        assert biomeMeta != null;
        return biomeMeta;
    }

    public ClIntBuffer getBiomeGrid() {
        assert biomeGrid != null;
        return biomeGrid;
    }

    public ClMemory getBiomeGrass() {
        assert biomeGrass != null;
        return biomeGrass;
    }

    public ClMemory getBiomeFoliage() {
        assert biomeFoliage != null;
        return biomeFoliage;
    }

    public ClMemory getBiomeDryFoliage() {
        assert biomeDryFoliage != null;
        return biomeDryFoliage;
    }

    public ClMemory getBiomeWater() {
        assert biomeWater != null;
        return biomeWater;
    }
}
