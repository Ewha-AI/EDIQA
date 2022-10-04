import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from utils import calculate_plcc
from models.SwinT_modified2 import SwinTransformer
from dataset import MayoRandomPatchDataset2, MayoDataset, PhantomTIQAData, ABIDEIITIQAData, MayoRandomPatchDataset
from glob import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from einops import rearrange
import argparse

from models.build import build_model


parser = argparse.ArgumentParser(description='TIQA')

parser.add_argument('--work_dirs', type=str, required=True, help='work directory name to save model')
parser.add_argument('--model_type', type=str, default='swin_conv_concat', help='model type | options: best_plcc or lastest')
parser.add_argument('--norm', action='store_true', help='whether to normalize image')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--gid', type=int, default=0, help='GPU ids')
parser.add_argument('--data_type', type=str, default='mayo', help='data type (mayo|phantom|mri)')
args = parser.parse_args()

# Test settings
batch_size = args.batch_size
device = 'cuda:{}'.format(args.gid)

# set model
# model = SwinTransformer(feature_num=4, mlp_head=1)
model = build_model(args.model_type)
model = model.cuda()

# transfer weights
checkpoint = torch.load('./work_dirs/{}/best_loss.pth'.format(args.work_dirs)) #, map_location=device)

for key in list(checkpoint.keys()):
    checkpoint[key.replace('module.', '')] = checkpoint.pop(key)

model.load_state_dict(checkpoint, strict=True)


# load data (rads)
if args.data_type == 'mayo':
    label_dir = '../../data/nimg-test-3channel/mayo_test.csv'
    test_list = sorted(glob('../../data/nimg-test-3channel/*/*/*.tiff')) # L506 & L067
    # test_data = MayoDataset(test_list, label_dir, transform='val', norm=args.norm)
    test_data = MayoRandomPatchDataset(test_list, label_dir, transform='val', norm=args.norm)
    test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=False)

    # test rads
    with open('results/{}_rads_{}.csv'.format(args.work_dirs, args.data_type), "a") as log:
        model.eval()
        test_output, gt = [], []
        with torch.no_grad():
            for data, mean, imgname in tqdm(test_loader, desc='test'):
                # data = data.cuda()
                for i in range(len(data)):
                    data[i] = data[i].cuda()
                mean = mean.cuda()
                mean = rearrange(mean, 'b -> b 1')

                output = model(data).cuda()
                test_output.append(output)
                gt.append(mean)

                log.write('{},{}\n'.format(imgname[0], float(output[0])))



# load data (d2iqa)

if args.data_type == 'mayo':
    label_dir = '../../data/nimg_3ch/mayo57np.csv'

    temp_test_list = []
    for pid in ['L506', 'L067']:
        temp_test_list.append(glob('../../data/nimg_3ch/{}/*/*.tiff'.format(pid)))

    test_list = []
    for i in range(len(temp_test_list)):
        test_list += temp_test_list[i]

    # test_data = MayoDataset(test_list, label_dir, transform='val', norm=args.norm)
    test_data = MayoRandomPatchDataset(test_list, label_dir, transform='val', norm=args.norm)
    test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=False)

elif args.data_type == 'phantom':
    label_dir = '../../data/csv/phantom_test_label.csv'
    test_list = sorted(glob('../../data/phantom_train_test_new/test/phantom/ge/chest/*/*.tiff'))

    test_data = PhantomTIQAData(test_list, label_dir, transform='test', norm=args.norm)
    test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=False)

elif args.data_type == 'mri':
    label_dir = '../../data/ABIDEII-GU_3ch/ABIDEII.csv'
    test_ids = ['28746', '28750', '28754', '28764', '28768', '28780', '28789', '28792', 
            '28796', '28810', '28830', '28847']
    temp_test_list, test_list = [], []
    for pid in test_ids:
        print(pid)
        temp_test_list.append(glob('../../data/ABIDEII-GU_3ch/{}/*.tiff'.format(pid)))
    for i in range(len(temp_test_list)):
        test_list += temp_test_list[i]

    test_data = ABIDEIITIQAData(test_list, label_dir, transform='test', norm=args.norm)
    test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=False)

# test d2iqa
with open('results/{}_d2iqa_{}.csv'.format(args.work_dirs, args.data_type), "a") as log:
    model.eval()
    test_output, gt = [], []
    with torch.no_grad():
        for data, mean, imgname in tqdm(test_loader, desc='test'):
            # print(imgname)
            # data = data.cuda()
            for i in range(len(data)):
                data[i] = data[i].cuda()
            mean = mean.cuda()
            mean = rearrange(mean, 'b -> b 1')

            output = model(data).cuda()
            test_output.append(output)
            gt.append(mean)

            log.write('{},{}\n'.format(imgname[0], float(output[0])))