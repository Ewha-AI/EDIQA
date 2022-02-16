import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from utils import calculate_plcc
from models.SwinT_modified2 import SwinTransformer
from dataset import MayoDataset
from glob import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from einops import rearrange
import argparse


parser = argparse.ArgumentParser(description='TIQA')

parser.add_argument('--work_dirs', type=str, required=True, help='work directory name to save model')
parser.add_argument('--model_type', type=str, default='best_plcc', help='model type | options: best_plcc or lastest')
parser.add_argument('--norm', action='store_true', help='whether to normalize image')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--gid', type=int, default=0, help='GPU ids')
args = parser.parse_args()

# Test settings
batch_size = args.batch_size
device = 'cuda:{}'.format(args.gid)
# seed = 42
# seed_everything(seed)

# load data
label_dir = '../../data/nimg-test-3channel/mayo_test.csv'
test_list = sorted(glob('../../data/nimg-test-3channel/*/*/*.tiff')) # L506 & L067

# load datasets
test_data = MayoDataset(test_list, label_dir, transform='val', norm=args.norm)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=False)

# set model
model = SwinTransformer(feature_num=4, mlp_head=1)
model = model.to(device)

# transfer weights
checkpoint = torch.load('./work_dirs/{}/{}.pth'.format(args.work_dirs, args.model_type), map_location=device)
model.load_state_dict(checkpoint, strict=False)

# test
model.eval()
test_output, gt = [], []
with torch.no_grad():
    for data, mean in tqdm(test_loader, desc='test'):
        data = data.to(device)
        mean = mean.to(device)
        mean = rearrange(mean, 'b -> b 1')

        output = model(data)
        test_output.append(output)
        gt.append(mean)

    test_plcc = calculate_plcc(test_output[0], gt[0])
    # epoch_test_plcc += val_plcc / len(val_loader)
    # epoch_val_loss += val_loss / len(val_loader)

print(
    f"Test PLCC : {test_plcc:.4f}.."
)