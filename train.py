import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from utils import seed_everything, get_distribution, CosineAnnealingWarmUpRestarts, distribution_to_score, calculate_plcc
from ViT import ViT
from dataset import MayoDataset
from glob import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

# Training settings
batch_size = 2
epochs = 500
lr = 1e-6
gamma = 0.7
seed = 42

seed_everything(seed)
device = 'cuda'

# load data
train_pid = ['L096', 'L291', 'L310']
val_pid = ['L333']

train_label_dir = '../../data/nimg-train/mayo_train.csv'
val_label_dir = '../../data/nimg-train/mayo_val.csv'

temp_train_list = []
for pid in train_pid:
    temp_train_list.append(glob('../../data/nimg-train/{}/*/*.tiff'.format(pid)))
temp_val_list = []
for pid in val_pid:
    temp_val_list.append(glob('../../data/nimg-train/{}/*/*.tiff'.format(pid)))

train_list, val_list = [], []
for i in range(len(temp_train_list)):
    train_list += temp_train_list[i]
for i in range(len(temp_val_list)):
    val_list += temp_val_list[i]
train_list = sorted(train_list)
val_list = sorted(val_list)
# test_list = sorted(glob('../../data/nimg/*/*/*.tiff')) # L506 & L067

# load datasets
train_data = MayoDataset(train_list, train_label_dir, transform='train')
train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
val_data = MayoDataset(val_list, val_label_dir, transform='val')
val_loader = DataLoader(dataset = val_data, batch_size=batch_size, shuffle=False)

# set model
model = ViT(
        image_size = 512,
        patch_size = 32,
        num_classes = 10,
        dim = 32,
        depth = 2,
        heads = 8,
        mlp_dim = 64,
        dropout = 0.1,
        emb_dropout = 0.1
        )
model = model.to(device)

# training
# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
# scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=150, T_mult=1, eta_max=lr,  T_up=10, gamma=0.5)

best_plcc = 0
best_epoch = 0

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