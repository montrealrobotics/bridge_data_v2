#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh && conda activate mini_grp

CHECKPOINT_CFG_PATH=$CHECKPOINT_DIR/dgcbc_128/gcbc_128_config.json
CHECKPOINT_WEIGHTS_PATH=$CHECKPOINT_DIR/dgcbc_128/checkpoint_2000000
POLICY=miniGRP.pth
TRAINING_CFG=grp_config.yaml
SERVER_IP=192.168.1.123
GOAL_TYPE=gc
IMAGE_SIZE=256

exec python /home/administrator/playground/bridge_data_v2/experiments/eval_grp.py --im_size $IMAGE_SIZE --policy $POLICY --goal_type $GOAL_TYPE --blocking --ip $SERVER_IP --checkpoint_weights_path $CHECKPOINT_WEIGHTS_PATH --checkpoint_config_path $CHECKPOINT_CFG_PATH --training_config $TRAINING_CFG

