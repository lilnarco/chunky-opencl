#!/usr/bin/env bash
# Nsight Compute wizard for ChunkyCL on RTX 3070 (Debian, driver 550.x, ncu 2025.1).
# Interactive: checks permissions, then prints the minimal capture recipe.
# Does NOT launch Chunky itself — attach to a short headless/profile run.
set -euo pipefail
NCU=/usr/local/cuda/bin/ncu
echo "== ChunkyCL ncu wizard =="
"$NCU" --version | head -2
nvidia-smi --query-gpu=name,driver_version --format=csv
echo ""
echo "-- perf permissions --"
if [ -r /proc/sys/kernel/perf_event_paranoid ]; then
  echo "perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid) (want <=2 for full counters)"
fi
echo ""
echo "If counters fail with ERR_NVGPUCTRPERM, run once as root:"
echo "  sudo $NCU --mode launch-and-attach --help | head -5"
echo ""
echo "-- minimal capture recipe (1080p x 32spp, profile-render ON) --"
cat <<'EOF'
# 1. Start Chunky with a small scene (e.g. hamhut at 1920x1080, target 32 spp,
#    Profile render ON so the kernel exposes its counters).
# 2. In another shell, find the java pid:
#      pgrep -af chunky | head
# 3. Capture ONE kernel launch (the `render` megakernel):
/usr/local/cuda/bin/ncu \
  --target-processes all \
  --kernel-name render \
  --launch-count 1 \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes.sum.per_second,\
lts__t_bytes.sum.per_second,\
dram__bytes.sum.per_second,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__occupancy.avg.pct_of_peak_sustained_active \
  -o /tmp/opencode/ncu-render \
  -p <java-pid>

# 4. Read back:
#      /usr/local/cuda/bin/ncu -i /tmp/opencode/ncu-render.ncu-rep --summary auto
#
# What to look at:
#  - sm__throughput < 60% + dram/lts high  -> memory/latency bound (our hypothesis;
#    matches 2026 wavefront paper: VRAM 2x is the win, not SM).
#  - sm__occupancy low (<40%)               -> register-bound megakernel; justifies
#    Phase 2 JIT defines + Phase 4 wavefront-lite split.
#  - sm__throughput already 80%+            -> wavefront premise collapses; stay
#    on host-batching + micro-opts only.
EOF
echo ""
echo "-- full-sweep recipe (only after lite looks promising) --"
cat <<'EOF'
/usr/local/cuda/bin/ncu \
  --target-processes all --kernel-name render --launch-count 3 \
  --set detailed \
  -o /tmp/opencode/ncu-render-full -p <java-pid>
EOF
