#!/usr/bin/env bash
# M6 — true occupancy + registers/launch. ncu cannot attach to an already-running
# Chunky (lesson from the oneshot), so this script launches Chunky UNDER ncu.
# Save your work first: you will quit Chunky at the end to finalize the report.
set -euo pipefail
NCU=/usr/local/cuda/bin/ncu
OUT=/tmp/opencode/ncu-render-full
JAVA=/usr/lib/jvm/java-25-amazon-corretto/bin/java

[ -x "$NCU" ] || { echo "ncu not found at $NCU"; exit 1; }
[ -x "$JAVA" ] || { echo "java not found at $JAVA"; exit 1; }
if pgrep -f "se.llbit.chunky.main.Chunky" >/dev/null; then
  echo "Chunky is already running. Quit it first (ncu must do the launching)."
  exit 1
fi
mkdir -p /tmp/opencode

# Classpath: everything the launcher puts on it (wildcard, version-proof).
CP=$(ls /home/zaddy/.chunky/lib/*.jar | tr '\n' ':')
echo "Launching Chunky under ncu..."
echo "  1. Load 1080p hamhut, target 32 spp, Profile OFF, OIDN off."
echo "  2. Hit RENDER once (32 spp batched = exactly 1 launch — perfect capture)."
echo "  3. QUIT Chunky to finalize the report."
echo ""
"$NCU" \
  --mode launch-and-attach \
  --target-processes all \
  --kernel-name render \
  --launch-count 1 \
  --set basic \
  --metrics launch__registers_per_thread \
  -o "$OUT" \
  -- "$JAVA" -Xmx24000m \
    -Dchunky.home=/home/zaddy/.chunky \
    -Djava.library.path=/usr/lib/x86_64-linux-gnu/jni \
    --enable-native-access=javafx.graphics \
    --module-path /usr/share/openjfx/lib \
    --add-modules javafx.controls,javafx.fxml \
    -classpath "$CP" se.llbit.chunky.main.Chunky

echo ""
echo "== RESULT — paste this back =="
"$NCU" -i "${OUT}.ncu-rep" --summary auto 2>&1 | tail -40
echo ""
echo "Also paste: achieved occupancy % and registers/thread if visible above."
echo "Reading guide: occupancy <40% = register-bound megakernel (JIT will help a"
echo "lot). Occupancy >60% = wavefront-lite premise is dead, stay the course."
