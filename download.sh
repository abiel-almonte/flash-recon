#!/bin/bash
set -euo pipefail

# Download datasets and model weights for flash-recon.
#
# Usage:
#   bash download.sh                # download everything
#   bash download.sh weights        # download model weights only
#   bash download.sh tum            # download TUM RGB-D sequences
#   bash download.sh euroc          # download all EuRoC sequences
#   bash download.sh euroc MH       # download EuRoC Machine Hall only
#   bash download.sh euroc V1       # download EuRoC Vicon Room 1 only
#   bash download.sh euroc V2       # download EuRoC Vicon Room 2 only
#
# Layout:
#   neural/weights/droid.pth
#   neural/weights/depth_anything_v2_vits.pth
#   datasets/TUM/fr1_desk/          (rgb/, depth/, groundtruth.txt)
#   datasets/TUM/fr1_room/
#   datasets/TUM/fr1_360/
#   datasets/EuRoC/machine_hall/    (MH_01_easy/ ... MH_05_difficult/)
#   datasets/EuRoC/vicon_room1/     (V1_01_easy/ ... V1_03_difficult/)
#   datasets/EuRoC/vicon_room2/     (V2_01_easy/ ... V2_03_difficult/)
#   datasets/EuRoC/groundtruth/     (MH_01_easy.txt ... V2_03_difficult.txt)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TARGET="${1:-all}"

# ============================================================
# Model weights
# ============================================================
download_weights() {
    echo "=== Downloading model weights ==="
    mkdir -p neural/weights

    if [ ! -f neural/weights/droid.pth ]; then
        echo "  Downloading DROID network weights..."
        gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh -O neural/weights/droid.pth
    else
        echo "  neural/weights/droid.pth already exists, skipping"
    fi

    if [ ! -f neural/weights/depth_anything_v2_vits.pth ]; then
        echo "  Downloading DepthAnythingV2 weights..."
        wget -q --show-progress \
            https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth \
            -O neural/weights/depth_anything_v2_vits.pth
    else
        echo "  neural/weights/depth_anything_v2_vits.pth already exists, skipping"
    fi

    echo ""
}

# ============================================================
# TUM RGB-D
# ============================================================
TUM_DIR="datasets/TUM"
TUM_BASE_URL="https://cvg.cit.tum.de/rgbd/dataset/freiburg1"

download_tum_sequence() {
    local name="$1"        # e.g. fr1_desk
    local archive="$2"     # e.g. rgbd_dataset_freiburg1_desk
    local dest="${TUM_DIR}/${name}"

    if [ -d "$dest" ]; then
        echo "  ${dest} already exists, skipping"
        return
    fi

    echo "  Downloading ${name}..."
    wget -q --show-progress "${TUM_BASE_URL}/${archive}.tgz" -O "/tmp/${archive}.tgz"
    mkdir -p "${TUM_DIR}"
    tar -xzf "/tmp/${archive}.tgz" -C "${TUM_DIR}"
    mv "${TUM_DIR}/${archive}" "${dest}"
    rm "/tmp/${archive}.tgz"
    echo "  Done: ${dest}"
}

download_tum() {
    echo "=== Downloading TUM RGB-D sequences ==="
    download_tum_sequence "fr1_desk" "rgbd_dataset_freiburg1_desk"
    download_tum_sequence "fr1_room" "rgbd_dataset_freiburg1_room"
    download_tum_sequence "fr1_360"  "rgbd_dataset_freiburg1_360"
    echo ""
}

# ============================================================
# EuRoC MAV
# ============================================================
EUROC_DIR="datasets/EuRoC"
EUROC_BASE_URL="https://www.research-collection.ethz.ch/bitstreams"

# Grouped archive bitstream IDs from ETH Research Collection
MH_ID="7b2419c1-62b5-4714-b7f8-485e5fe3e5fe"   # Machine Hall (~12 GB)
V1_ID="02ecda9a-298f-498b-970c-b7c44334d880"     # Vicon Room 1 (~5.8 GB)
V2_ID="ea12bc01-3677-4b4c-853d-87c7870b8c44"     # Vicon Room 2 (~5.7 GB)

download_euroc_group() {
    local name="$1"
    local bitstream_id="$2"
    local dest="${EUROC_DIR}/${name}"

    if [ -d "$dest" ]; then
        echo "  ${dest} already exists, skipping"
        return
    fi

    local url="${EUROC_BASE_URL}/${bitstream_id}/download"
    local zipfile="/tmp/${name}.zip"

    echo "  Downloading ${name}..."
    wget -q --show-progress "${url}" -O "${zipfile}"

    echo "  Extracting into ${EUROC_DIR}/..."
    mkdir -p "${EUROC_DIR}"
    unzip -qo "${zipfile}" -d "${EUROC_DIR}"
    rm "${zipfile}"
    echo "  Done: ${dest}"
}

download_euroc() {
    local filter="${1:-all}"
    echo "=== Downloading EuRoC MAV sequences ==="

    if [[ "$filter" == "all" || "$filter" == "MH" ]]; then
        download_euroc_group "machine_hall" "$MH_ID"
    fi
    if [[ "$filter" == "all" || "$filter" == "V1" ]]; then
        download_euroc_group "vicon_room1" "$V1_ID"
    fi
    if [[ "$filter" == "all" || "$filter" == "V2" ]]; then
        download_euroc_group "vicon_room2" "$V2_ID"
    fi

    echo ""
}

case "$TARGET" in
    all)
        download_weights
        download_tum
        download_euroc "all"
        ;;
    weights)
        download_weights
        ;;
    tum)
        download_tum
        ;;
    euroc)
        download_euroc "${2:-all}"
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: bash download.sh [all|weights|tum|euroc [MH|V1|V2]]"
        exit 1
        ;;
esac

echo "Done."
