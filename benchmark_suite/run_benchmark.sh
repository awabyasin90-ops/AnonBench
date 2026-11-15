#!/usr/bin/env bash
# ==============================================================================
# Master 1-Click Automotive Privacy Anonymization Benchmark Runner
# ==============================================================================
# Usage:
#   bash run_benchmark.sh [INPUT_DIR] [OUTPUT_DIR] [MAX_FRAMES] [NUM_COMPARISONS] [DEVICE]
#
# Examples:
#   1. Run 100 sample frames:
#      bash run_benchmark.sh "/Users/mazeny/Desktop/test models/not blur" "./results_100" 100 10 cpu
#
#   2. Run Big Evaluation (7,200 frames on GPU / CUDA):
#      bash run_benchmark.sh "/path/to/full_dataset" "./results_7200" 7200 20 cuda
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_DIR="${1:-/Users/mazeny/Desktop/test models/not blur}"
OUTPUT_DIR="${2:-$SCRIPT_DIR/outputs}"
MAX_FRAMES="${3:-100}"
NUM_COMPARISONS="${4:-10}"
DEVICE="${5:-cpu}"

echo "======================================================================"
echo "  AUTOMOTIVE PRIVACY ANONYMIZATION BENCHMARK SUITE"
echo "======================================================================"
echo "  Script Directory : $SCRIPT_DIR"
echo "  Input Directory  : $INPUT_DIR"
echo "  Output Directory : $OUTPUT_DIR"
echo "  Max Frames       : $MAX_FRAMES (Set 7200 for full evaluation)"
echo "  Comparison Cards : $NUM_COMPARISONS"
echo "  Compute Device   : $DEVICE"
echo "======================================================================"

python3 "$SCRIPT_DIR/master_benchmark_runner.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --max-frames "$MAX_FRAMES" \
    --num-comparisons "$NUM_COMPARISONS" \
    --device "$DEVICE"

echo ""
echo "[✓] Benchmark execution completed successfully!"
echo "    Executive Report: $OUTPUT_DIR/README.md"
echo "    Charts:           $OUTPUT_DIR/charts"
echo "    Comparisons:      $OUTPUT_DIR/blurred/10_Frame_Comparisons"

