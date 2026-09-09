# Virtual-depth automation

The octree march offset is derived from a host-computed maximum coordinate magnitude (`maxCoord` from octree extent and camera distance) instead of a user-facing virtual-depth slider, which was removed. Collision bounds always use the real octree depth so parallel rays are pushed into solid regions (the parallel-projection Y-clip fix). This supersedes the deleted `PARALLEL_YCLIP_BUGFIX.md` note.
