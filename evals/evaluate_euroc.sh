#!/bin/bash
set -euo pipefail

# EuRoC MAV dataset batch evaluation for flash-recon SLAM
#
# Usage:
#   bash tools/evaluate_euroc.sh [--stride 2] [extra args...]
#
# Expects:
#   - EuRoC sequences in datasets/EuRoC/<sequence_name>/
#   - Ground truth in datasets/EuRoC/groundtruth/<sequence_name>.txt

EUROC_PATH="${EUROC_PATH:-datasets/EuRoC}"
WEIGHTS="${WEIGHTS:-neural/weights/droid.pth}"
DEPTH_WEIGHTS="${DEPTH_WEIGHTS:-neural/weights/depth_anything_v2_vits.pth}"

evalset=(
    MH_01_easy
    MH_02_easy
    MH_03_medium
    MH_04_difficult
    MH_05_difficult
    V1_01_easy
    V1_02_medium
    V1_03_difficult
    V2_01_easy
    V2_02_medium
    V2_03_difficult
)

RESULTS_FILE=$(mktemp)
trap 'rm -f "$RESULTS_FILE"' EXIT

echo "========================================"
echo "  EuRoC Batch Evaluation (flash-recon)"
echo "========================================"
echo ""

resolve_datapath() {
    local seq="$1"
    # Try flat layout first: datasets/EuRoC/MH_01_easy/mav0/
    if [ -d "${EUROC_PATH}/${seq}/mav0" ]; then
        echo "${EUROC_PATH}/${seq}"
        return
    fi
    # Try grouped layout: datasets/EuRoC/machine_hall/MH_01_easy/mav0/
    local subdir
    case "$seq" in
        MH*) subdir="machine_hall" ;;
        V1*) subdir="vicon_room1" ;;
        V2*) subdir="vicon_room2" ;;
    esac
    if [ -d "${EUROC_PATH}/${subdir}/${seq}/mav0" ]; then
        echo "${EUROC_PATH}/${subdir}/${seq}"
        return
    fi
    # Return flat path (will fail with a clear error)
    echo "${EUROC_PATH}/${seq}"
}

for seq in "${evalset[@]}"; do
    datapath=$(resolve_datapath "$seq")
    gt="${EUROC_PATH}/groundtruth/${seq}.txt"

    if [ ! -d "$datapath/mav0" ]; then
        echo "SKIP ${seq} (not found: ${datapath}/mav0/)"
        echo "${seq} SKIP" >> "$RESULTS_FILE"
        continue
    fi

    echo "--- ${seq} ---"
    # Run evaluation and capture the RESULT line
    output=$(python -m evals.eval_euroc \
        --datapath "$datapath" \
        --gt "$gt" \
        --weights "$WEIGHTS" \
        --depth_weights "$DEPTH_WEIGHTS" \
        "$@" 2>&1) || true

    echo "$output"

    # Extract RESULT line
    result_line=$(echo "$output" | grep "^RESULT" || true)
    if [ -n "$result_line" ]; then
        echo "$result_line" >> "$RESULTS_FILE"
    else
        echo "${seq} FAIL" >> "$RESULTS_FILE"
    fi

    echo ""
done

# Print summary table
echo "========================================"
echo "  Summary"
echo "========================================"
printf "%-20s %s\n" "Sequence" "ATE RMSE (m)"
printf "%-20s %s\n" "--------" "------------"

total=0
count=0
while IFS= read -r line; do
    if echo "$line" | grep -q "^RESULT"; then
        seq=$(echo "$line" | awk '{print $2}')
        ate=$(echo "$line" | awk '{print $3}')
        printf "%-20s %s\n" "$seq" "$ate"
        total=$(echo "$total + $ate" | bc)
        count=$((count + 1))
    elif echo "$line" | grep -q "SKIP"; then
        seq=$(echo "$line" | awk '{print $1}')
        printf "%-20s %s\n" "$seq" "SKIP"
    else
        seq=$(echo "$line" | awk '{print $1}')
        printf "%-20s %s\n" "$seq" "FAIL"
    fi
done < "$RESULTS_FILE"

if [ "$count" -gt 0 ]; then
    avg=$(echo "scale=6; $total / $count" | bc)
    printf "%-20s %s\n" "--------" "------------"
    printf "%-20s %s (%d sequences)\n" "Average" "$avg" "$count"
fi
