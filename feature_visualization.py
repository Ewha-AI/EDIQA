import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from utils import seed_everything, get_distribution, CosineAnnealingWarmUpRestarts, distribution_to_score, calculate_plcc
from models.build import build_model
from dataset import MayoDataset, MayoPatchDataset, MayoRandomPatchDataset, MayoPatchDataset2, MayoRandomPatchDataset2
from glob import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from einops import rearrange
import argparse
import cv2


parser = argparse.ArgumentParser(description='TIQA')

parser.add_argument('--work_dirs', type=str, required=True, help='work directory name to save model')
parser.add_argument('--transfer', type=str, default=None, help='whether to trasfer weights|Options: detection, imagenet, resume, none')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--gid', type=int, default=0, help='GPU ids')
parser.add_argument('--model_type', type=str, default='multi_swin', help='model type')
args = parser.parse_args()

def img_tile(img_list):
    return cv2.vconcat([cv2.hconcat(img) for img in img_list])


# Training settings
batch_size = args.batch_size
device = 'cuda:{}'.format(args.gid)

# load data
test_label_dir = '../../data/nimg-test-3channel/mayo_test.csv'
test_list = sorted(glob('../../data/nimg-test-3channel/*/*/*.tiff')) # L506 & L067

# load datasets
test_data = MayoRandomPatchDataset2(test_list, test_label_dir, transform='val', norm=False)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=False, num_workers=4)

# set model
model = build_model(args.model_type)
model = torch.nn.DataParallel(model).cuda()

# transfer weights
checkpoint = torch.load('./work_dirs/{}/best_plcc.pth'.format(args.work_dirs))
model.load_state_dict(checkpoint, strict=True)

# test
model.eval()
test_outputs, gt = [], []
with torch.no_grad():
    for data, mean in tqdm(test_loader, desc='test'):
        mean = mean.cuda()
        mean = rearrange(mean, 'b -> b 1')

        test_score = torch.zeros(size=mean.shape).cuda()
        for img in data:
            img = img.cuda()
            test_output = model(img)
            test_score += test_output
        test_score = test_score / len(data)
        test_outputs.append(test_score)
        gt.append(mean)

        # test_plcc = calculate_plcc(test_score, mean)
        # epoch_test_plcc += test_plcc / len(test_loader)
    test_plcc = calculate_plcc(test_output, gt[0])
    print(test_plcc)
    # exit()

print(
    f"Test PLCC : {test_plcc:.4f}.."
)

# visualize feature map
# print(model.module[0].swin_transformer)
# print(model.module[0].convnext)
# load data
test_label_dir = '../../data/nimg-test-3channel/mayo_test.csv'
test_list = sorted(glob('../../data/nimg-test-3channel/*/*/*.tiff'))[0] # L506 & L067
test_list = [test_list]

# load datasets
test_data = MayoRandomPatchDataset2(test_list, test_label_dir, transform='val', norm=False)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=False)

total_swin_outputs, total_conv_outputs = [], []
swin_model = model.module[0].swin_transformer
conv_model = model.module[0].convnext
for data, _ in test_loader:
    swin_outputs, conv_outputs = [], []
    for img in data:
        img = img.cuda()
        swin_output = swin_model(img)[-1]
        swin_output = swin_output.squeeze(0).view(7, 7, -1)
        conv_output = conv_model(img)[-1]
        conv_output = conv_output.squeeze(0).view(7, 7, -1)
        swin_outputs.append(swin_output)
        conv_outputs.append(conv_output)
    swin = img_tile([[swin_outputs[0], swin_outputs[1]] , [swin_outputs[2], swin_outputs[3]]])
    conv = img_tile([conv_outputs[:2], conv_outputs[:4]])
    total_swin_outputs.append(swin)
    print(swin.shape)
    exit()
    total_conv_outputs.append(conv)

# mkdir
save_dir = './work_dirs/{}/featuremap_visualize/'.format(args.work_dirs)
os.makedirs(save_dir, exists_ok=True)

#concat 24x32 / 27x27
# img_tile
img_tile_red = img_tile()
    