#!/bin/sh
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_densepose.py --config-file ./configs/e2e_keypoint_rcnn_R_50_FPN_1x.yaml SOLVER.IMS_PER_BATCH 8
