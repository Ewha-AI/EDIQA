#!/bin/bash

# Uncomment and set the following variables for corresponding dataset
# ################### MAYO ##################
DATA_TYPE="mayo"
WORK_DIR="test_mayo"
LABEL_DIR='/media/wonkyong/hdd2/data/D2IQA/mayo/mayo57np.csv'
IMG_PATH='/media/wonkyong/268c4ce6-d8a7-453b-9607-b1a121f48743/data/nimg_train/*/*/*.tiff'
# ################## PHANTOM ################
# DATA_TYPE="phantom"
# WORK_DIR="test_phantom"
# LABEL_DIR='/media/wonkyong/hdd2/data/D2IQA/phantom/phantom_label_0426.csv'
# IMG_PATH='/media/wonkyong/268c4ce6-d8a7-453b-9607-b1a121f48743/data/phantom_train_test_new/train/phantom/ge/chest/*/*.tiff'
# ################## MRI ####################
# DATA_TYPE="mri"
# WORK_DIR="test_mri"
# LABEL_DIR='/media/wonkyong/hdd2/data/D2IQA/d2iqa_lbset_ABIDEII_GU_1124_6_40.csv'
# IMG_PATH='/media/wonkyong/268c4ce6-d8a7-453b-9607-b1a121f48743/data/ABIDEII_GU_1/ABIDEII-GU_1/*/*.tiff'


python train.py \
    --work_dirs $WORK_DIR \
    --label_dir $LABEL_DIR \
    --img_path "$IMG_PATH" \
    --transfer detection \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-5 \
    --model_type ediqa \
    --data_type $DATA_TYPE