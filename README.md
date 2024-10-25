# Efficient Deep-Detector Image Quality Assessment Based on Knowledge Distillation

This repo contains the supported code and configuration files to reproduce image quality assessment results of [Efficient Deep-Detector Image Quality Assessment Based on Knowledge Distillation](doi.org/10.1109/TIM.2023.3346519). 

## Updates

***10/25/2024*** Initial commits

## Models and pretrained backbone weights

| Model | Swin transformer backbone | Convnext backbone |
| :---: | :---: | :---: |
| [EDIQA](https://drive.google.com/file/d/1eQfFzyNbu7W0z6E9iYucMo13t7Px6Td4/view?usp=sharing) | [model](https://drive.google.com/file/d/1wK6km6t5nXO4rG_jaUg0Oyh5U2fLfPk9/view?usp=drive_link)/[config]() | [model](https://drive.google.com/file/d/1tmdTb5-dRsUVwKsKDitpGmZMZec1wp5I/view?usp=drive_link)/[config]() |

**Notes**:

- To train the model using pretrained backbone weights, download the pretrained models and place them in `work_dirs/detection_models/mayo`.
- The pretrained models were trained using [MMDetection](https://github.com/open-mmlab/mmdetection).

## Usage

### Inference
```
bash test.sh
```

**Arguments**:

- `WORK_DIR`: model weight path for inference
- `LABEL_DIR`: path to the csv file containing a list of image names in the `img` column
- `IMG_PATH`: path to test images
- `OUTPUT_PATH`: path to save the results file
- `GT_PATH`: path to the ground truth csv file for correlation calculation. Should conatin two columns, `img` and `mean`, with image names and ground truth IQA scores.

### Training

To train EDIQA with pre-trained detection weights, download pretrained backbone models and run:
```
bash train.sh
```

**Arguments**:
- `WORK_DIR`: directory to save the trained model
- `LABEL_DIR`: path to the csv file containing image names and IQA scores in the `img` and `mean` columns
- `IMG_PATH`: path to training images
- `transfer`: type of pretrained models. To train with imagenet weightsm set `--transfer` to `imagenet`. To train from scratch, set `--transfer` to `none`.

### Gererating D2IQA detection scores for EDIQA training

To generate D2IQA scores for EDIQA training, refer to this repo and create csv files with D2IQA scores for each image.<br>
Then, run `make_label.py` to generate the final annotation file for EDIQA, adjusting `detect_path` and `save_dir` as needed.