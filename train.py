import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from utils import seed_everything, get_distribution, CosineAnnealingWarmUpRestarts, distribution_to_score, calculate_plcc
from models.ViT import ViT
from dataset import MayoDataset
from glob import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import random

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from collections import OrderedDict

from utils import seed_everything, get_distribution, distribution_to_score, calculate_plcc, CosineAnnealingWarmUpRestarts
from models.build import build_model
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
parser.add_argument('--transfer', type=str, default=None, help='whether to trasfer weights|Options: detection, imagenet, resume, none')
parser.add_argument('--scheduler', type=str, default=None, help='whether to trasfer weights|Options: step, cosince, none')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
parser.add_argument('--step', type=int, default=10, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='learning rate')
parser.add_argument('--val_pid', type=str, default='L333', help='validation pid')
parser.add_argument('--gid', type=int, default=0, help='GPU ids')
parser.add_argument('--model_type', type=str, default='multi_swin', help='model type')
# parser.add_argument('--feature_num', type=int, default=-1, help='feature vector numbers|Options: -1, 4, 12')
# parser.add_argument('--mlp_head', type=int, default=None, help='mlp head')
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
args = parser.parse_args()


# Training settings
batch_size = args.batch_size
# epochs = 500
epochs = 100
lr = args.lr
gamma = args.gamma
step = args.step
seed = 42 #42

seed_everything(seed)
device = 'cuda:{}'.format(args.gid)

# load data
total_pid = sorted(os.listdir('../../data/nimg_3ch'))
total_pid = [pid for pid in total_pid if pid[0]=='L']
val_pid = [args.val_pid]
# train_pid = [pid for pid in total_pid if pid not in val_pid]
test_pid = ['L067', 'L506']
train_pid = ['L096', 'L291', 'L310', 'L109', 'L143', 'L192', 'L286']
val_pid = ['L179', 'L186', 'L187', 'L125']

label_dir = '../../data/nimg_3ch/mayo57np.csv'
# test_label_dir = '../../data/nimg-test-3channel/mayo_test.csv'

temp_train_list = []
for pid in train_pid:
    temp_train_list.append(glob('../../data/nimg_3ch/{}/*/*.tiff'.format(pid)))
temp_val_list = []
for pid in val_pid:
    temp_val_list.append(glob('../../data/nimg_3ch/{}/*/*.tiff'.format(pid)))
temp_test_list = []
for pid in test_pid:
    temp_test_list.append(glob('../../data/nimg_3ch/{}/*/*.tiff'.format(pid)))


train_list, val_list, test_list = [], [], []
for i in range(len(temp_train_list)):
    train_list += temp_train_list[i]
for i in range(len(temp_val_list)):
    val_list += temp_val_list[i]
for i in range(len(temp_test_list)):
    test_list += temp_test_list[i]
train_list = sorted(train_list)
# random select train_list
# random.shuffle(train_list)
# train_list = sorted(train_list[:20000])

val_list = sorted(val_list)
test_list = sorted(test_list)

print('================================')
print('    Dataset')
print('--------------------------------')
print('    Train: ', len(train_list))
print('    Validation: ', len(val_list))
print('    Test: ', len(test_list))
print('================================')


# load datasets
train_data = MayoDataset(train_list, label_dir, transform='train', norm=False)
train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True, num_workers=4)
val_data = MayoDataset(val_list, label_dir, transform='val', norm=False)
val_loader = DataLoader(dataset = val_data, batch_size=batch_size, shuffle=False, num_workers=4)
test_data = MayoDataset(test_list, label_dir, transform='val', norm=False)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=False, num_workers=4)

# set model
# model = SwinTransformer(feature_num=args.feature_num, mlp_head = args.mlp_head)
model = build_model(args.model_type)
# model = SwinTransformer(num_classes=1)
# model = model.to(device)
model = torch.nn.DataParallel(model).cuda()

# training setting
# loss function
criterion = nn.L1Loss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr) #, weight_decay=0.1)
if args.scheduler == 'step':
    scheduler = StepLR(optimizer, step_size=step, gamma=gamma)
elif args.scheduler == 'cosine':
    optimizer = optim.Adam(model.parameters(), lr=0) #, weight_decay=0.1)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=100, T_mult=1, eta_max=lr,  T_up=10, gamma=0.5)
elif args.scheduler == 'plateau':
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

best_loss = 100
best_plcc = 0
best_epoch = 0
start_epoch = 0
# # Training settings
# batch_size = 2
# epochs = 500
# lr = 1e-6
# gamma = 0.7
# seed = 42

# seed_everything(seed)
# device = 'cuda'

# # load data
# train_pid = ['L096', 'L291', 'L310']
# val_pid = ['L333']

# train_label_dir = '../../data/nimg-train/mayo_train.csv'
# val_label_dir = '../../data/nimg-train/mayo_val.csv'

# temp_train_list = []
# for pid in train_pid:
#     temp_train_list.append(glob('../../data/nimg-train/{}/*/*.tiff'.format(pid)))
# temp_val_list = []
# for pid in val_pid:
#     temp_val_list.append(glob('../../data/nimg-train/{}/*/*.tiff'.format(pid)))

# train_list, val_list = [], []
# for i in range(len(temp_train_list)):
#     train_list += temp_train_list[i]
# for i in range(len(temp_val_list)):
#     val_list += temp_val_list[i]
# train_list = sorted(train_list)
# val_list = sorted(val_list)
# # test_list = sorted(glob('../../data/nimg/*/*/*.tiff')) # L506 & L067

# # load datasets
# train_data = MayoDataset(train_list, train_label_dir, transform='train')
# train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
# val_data = MayoDataset(val_list, val_label_dir, transform='val')
# val_loader = DataLoader(dataset = val_data, batch_size=batch_size, shuffle=False)

# # set model
# model = ViT(
#         image_size = 224,
#         patch_size = 32,
#         num_classes = 5,
#         dim = 32,
#         depth = 2,
#         heads = 8,
#         mlp_dim = 64,
#         dropout = 0.1,
#         emb_dropout = 0.1
#         )
# model = model.to(device)

# # training
# # loss function
# criterion = nn.CrossEntropyLoss()
# # optimizer
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
# # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=150, T_mult=1, eta_max=lr,  T_up=10, gamma=0.5)

# best_plcc = 0
# best_epoch = 0

for epoch in range(epochs):
    epoch_loss = 0
    epoch_plcc = 0

    for data, label, mean in tqdm(train_loader, desc='train'):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = distribution_to_score(list(range(1,11)), output)
        print(preds)
        print(output)
        print(mean)
        exit()
        gt = mean
        plcc = calculate_plcc(preds, gt)
        epoch_plcc += plcc / len(train_loader)
        epoch_loss += loss / len(train_loader)
        # acc = (output.argmax(dim=1) == label).float().mean()
        # epoch_accuracy += acc / len(train_loader)

    with torch.no_grad():
        epoch_val_plcc = 0
        epoch_val_loss = 0
        for data, label, mean in tqdm(val_loader, desc='validation'):
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            preds = distribution_to_score(list(range(1,11)), val_output)
            gt = mean
            val_plcc = calculate_plcc(preds, gt)
            epoch_val_plcc += val_plcc / len(val_loader)
            epoch_val_loss += val_loss / len(val_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - plcc : {epoch_plcc:.4f} - val_loss : {epoch_val_loss:.4f} - val_plcc : {epoch_val_plcc:.4f}"
    )
    # print(
    #     f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - plcc : {epoch_plcc:.4f}"
    # )

    work_dir = './work_dirs/temp3/'
    # save logs
    if os.path.isfile(work_dir+'logs.txt') == False:
        log_file = open(work_dir+'logs.txt', "w")
    with open(work_dir+'logs.txt', "a") as log:
        log_line = f"\"Epoch\": {epoch+1}, \"loss\": {epoch_loss:.4f}, \"plcc\": {epoch_plcc:.4f}, \"val_loss\": {epoch_val_loss:.4f}, \"val_plcc\": {epoch_val_plcc:.4f}"
        log_line = '{'+log_line+'}\n'
        log.write(log_line)

    # save best plcc model
    if epoch_val_plcc >= best_plcc:
        best_plcc = epoch_val_plcc
        torch.save(model.state_dict(), work_dir+'best_plcc.pth'.format(epoch+1))
        best_epoch = epoch+1
    
    print('Best PLCC epoch so far: ', best_epoch)