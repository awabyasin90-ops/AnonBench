#!/bin/zsh
# One-click launcher for Custom X-AnyLabeling with all Hacks & Optimizations
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/X-AnyLabeling"

if [ ! -d "$TARGET_DIR" ]; then
    TARGET_DIR="/Users/mazeny/Library/CloudStorage/GoogleDrive-awab.yasin90@gmail.com/My Drive/Research_Papers_2026/Idea's_Hub/03-Anonymization Failure Patterns Across Real Drives and CARLA Simulation/3_Systems/X/X-AnyLabeling"
fi

cd "$TARGET_DIR" || exit 1
source .venv/bin/activate

echo "=========================================================="
echo " 🚀 Launching Customized X-AnyLabeling (AWAB Optimized) "
echo "=========================================================="
echo " - Shortcut [D]: Auto-Save & Next Image"
echo " - Shortcut [F]: Select Face & Rectangle (Neon Green)"
echo " - Shortcut [P]: Select Plate & Rectangle (Deep Sky Blue)"
echo " - Mouse Scroll: Direct Zoom without Ctrl"
echo " - New Shapes  : No Dialog Popup (Silent Auto-Save)"
echo " - Memory Cache: RAM Background Pre-fetching (0 ms delay)"
echo " - AI Engine   : MobileSAM Auto-loaded"
echo "=========================================================="

python anylabeling/app.py

