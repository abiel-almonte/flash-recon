#!/bin/bash

# droid model
gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh -O /workspace/neural/weights/droid.pth

#depth model
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth -P /workspace/neural/weights/

# datasets
mkdir -p /workspace/datasets/
wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
tar -xvzf rgbd_dataset_freiburg1_desk.tgz -C datasets/
mv /workspace/datasets/rgbd_dataset_freiburg1_desk /workspace/datasets/desk
rm rgbd_dataset_freiburg1_desk.tgz

wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz
tar -xvzf rgbd_dataset_freiburg1_room.tgz -C datasets/
mv /workspace/datasets/rgbd_dataset_freiburg1_room /workspace/datasets/room
rm rgbd_dataset_freiburg1_room.tgz