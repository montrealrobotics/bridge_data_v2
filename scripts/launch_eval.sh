#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh && conda activate mini_grp

POLICY=miniGRP.pth
TRAINING_CFG=grp_config.yaml
SERVER_IP=192.168.1.123
GOAL_TYPE=gc
IMAGE_SIZE=256

exec python /home/administrator/playground/bridge_data_v2/experiments/eval_grp.py --im_size $IMAGE_SIZE --policy $POLICY --goal_type $GOAL_TYPE --blocking --ip $SERVER_IP --training_config $TRAINING_CFG

