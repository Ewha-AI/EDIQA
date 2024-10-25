#!/bin/bash

# Uncomment and set the following variables for corresponding dataset
# ################### MAYO(Rads) ##################
DATA_TYPE="mayo"
WORK_DIR="./work_dirs/mayo/best_loss.pth"
LABEL_DIR='/media/wonkyong/hdd2/data/D2IQA/mayo/mayo_test.csv'
IMG_PATH='/media/wonkyong/268c4ce6-d8a7-453b-9607-b1a121f48743/data/nimg-test-3channel/*/*/*.tiff'
OUTPUT_PATH='results/mayo_rads.csv'
GT_PATH='/media/wonkyong/hdd2/data/D2IQA/mayo/mayo_test_5.csv.csv'
################## MAYO(D2IQA) ##################
# DATA_TYPE="mayo"
# WORK_DIR="./work_dirs/mayo/best_loss.pth"
# LABEL_DIR='/media/wonkyong/hdd2/data/D2IQA/mayo_d2iqa_test.csv'
# IMG_PATH='/media/wonkyong/268c4ce6-d8a7-453b-9607-b1a121f48743/data/nimg/nimg_3channel_2/*/*/*.tiff'
# OUTPUT_PATH='results/mayo_d2iqa.csv'
# GT_PATH='/media/wonkyong/hdd2/data/D2IQA/mayo/mayo57np.csv'
# ################## PHANTOM(D2IQA) ##################
# DATA_TYPE="phantom"
# WORK_DIR="work_dirs/phantom/best_loss.pth"
# LABEL_DIR='/media/wonkyong/hdd2/data/D2IQA/phantom/phantom_test_label_0427.csv'
# IMG_PATH='/media/wonkyong/268c4ce6-d8a7-453b-9607-b1a121f48743/data/phantom_train_test_new/test/phantom/ge/chest/*/*.tiff'
# OUTPUT_PATH='results/phantom_d2iqa.csv'
# GT_PATH='/media/wonkyong/hdd2/data/D2IQA/phantom/phantom.csv'
# ################## MRI(D2IQA) ##################
# DATA_TYPE="mri"
# WORK_DIR="work_dirs/mri/best_loss.pth"
# # LABEL_DIR='/media/wonkyong/hdd2/data/D2IQA/mri_d2iqa_test.csv'
# LABEL_DIR='/media/wonkyong/hdd2/data/EDIQA/mri/rads_mean.csv'
# # IMG_PATH='/media/wonkyong/268c4ce6-d8a7-453b-9607-b1a121f48743/data/ABIDEII_GU_1/ABIDEII-GU_1/*/*.tiff'
# IMG_PATH='/media/wonkyong/268c4ce6-d8a7-453b-9607-b1a121f48743/data/ABIDEII_GU_1/ABIDEII-GU_3ch/*/*.tiff'
# # OUTPUT_PATH='results/mri_d2iqa.csv'
# OUTPUT_PATH='results/mri_rads.csv'
# # GT_PATH='/media/wonkyong/hdd2/data/D2IQA/d2iqa_lbset_ABIDEII_GU_1124_6_40.csv'
# GT_PATH='/media/wonkyong/hdd2/data/EDIQA/mri/rads_mean.csv'


python test.py \
    --work_dir $WORK_DIR \
    --output_path $OUTPUT_PATH \
    --data_type $DATA_TYPE \
    --label_dir $LABEL_DIR \
    --img_path "$IMG_PATH"

python calculate_corr.py \
    --pred_path $OUTPUT_PATH \
    --gt_path $GT_PATH \
    --data_type $DATA_TYPE