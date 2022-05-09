import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils import seed_everything, get_distribution, CosineAnnealingWarmUpRestarts, distribution_to_score, calculate_plcc
from models.swin_transformer_aggregation_per_stage import SwinTransformer
# from models.swin_transformer import SwinTransformer
from models.ViT import ViT
from dataset import PIPALDataset
from glob import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from einops import rearrange
import argparse


parser = argparse.ArgumentParser(description='TIQA')

parser.add_argument('--work_dirs', type=str, required=True, help='work directory name to save model')
parser.add_argument('--norm', action='store_true', help='whether to normalize image')
parser.add_argument('--transfer', type=str, default=None, help='whether to trasfer weights|Options: detection, imagenet, resume, none')
parser.add_argument('--scheduler', type=str, default=None, help='whether to trasfer weights|Options: step, cosince, none')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--val_pid', type=str, default='L333', help='validation pid')
parser.add_argument('--gid', type=int, default=0, help='GPU ids')
parser.add_argument('--feature_num', type=int, default=-1, help='feature vector numbers|Options: -1, 4, 12')
parser.add_argument('--mlp_head', type=int, default=None, help='mlp head')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
args = parser.parse_args()

# Training settings
batch_size = args.batch_size
epochs = 500
lr = args.lr
gamma = 0.7
seed = 42

seed_everything(seed)
device = 'cuda:{}'.format(args.gid)

# load data
train_label_dir = '../../data/pipal/train/dmos.csv'
val_label_dir = '../../data/pipal/test/dmos.csv'

train_list = sorted(glob('../../data/pipal/train/dst_imgs/*.bmp'))
val_list = sorted(glob('../../data/pipal/test/dst_imgs/*.bmp'))
# test_list = sorted(glob('../../data/nimg/*/*/*.tiff')) # L506 & L067

# transform image
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# load datasets
train_data = PIPALDataset(train_list, train_label_dir, transform=data_transforms['train'], norm=args.norm)
train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True, num_workers=12)
val_data = PIPALDataset(val_list, val_label_dir, transform=data_transforms['val'], norm=args.norm)
val_loader = DataLoader(dataset = val_data, batch_size=batch_size, shuffle=False, num_workers=12)

# set model
# model = SwinTransformer(feature_num=args.feature_num, mlp_head = args.mlp_head)
model = SwinTransformer(num_classes=1)
# model = ViT(
#         image_size = 512,
#         patch_size = 32,
#         num_classes=10,
#         dim = 32,
#         depth = 2, # encoder depth
#         heads = 8,
#         mlp_dim = 64,
#         dropout = 0.1,
#         emb_dropout = 0.1
#     )
model = model.to(device)

# transfer weights
if args.transfer == 'detection':
    checkpoint = torch.load('work_dirs/cascade_mask_rcnn_detDataset_1_3/epoch_100.pth', map_location='cuda:{}'.format(args.gid))

    del checkpoint['meta']
    del checkpoint['optimizer']

    layers = [c for c in checkpoint['state_dict'].keys() if 'backbone' not in c]
    for l in layers:
        del checkpoint['state_dict'][l]

    for key in list(checkpoint['state_dict'].keys()):
        checkpoint['state_dict'][key.replace('backbone.', '')] = checkpoint['state_dict'].pop(key)

    model.load_state_dict(checkpoint['state_dict'], strict=False)
elif args.transfer == 'imagenet':
    checkpoint = torch.load('./work_dirs/swin_tiny_patch4_window7_224.pth', map_location='cuda:{}'.format(args.gid))
    
    del checkpoint['model']['head.bias']
    del checkpoint['model']['head.weight']   

    # del checkpoint['model']['norm.weight']
    # del checkpoint['model']['norm.bias']

    for key in ['norm.weight', 'norm.bias']:
        checkpoint['model'][key.replace('norm.', 'norm3.')] = checkpoint['model'].pop(key)

    model.load_state_dict(checkpoint['model'], strict=False)

elif args.transfer == 'resume':
    checkpoint = torch.load('./work_dirs/{}/latest.pth'.format(args.work_dirs))
    model.load_state_dict(checkpoint, strict=True)
    # lr = 1.680700e-07

# training
# loss function
criterion = nn.MSELoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
if args.scheduler == 'step':
    scheduler = StepLR(optimizer, step_size=50, gamma=gamma)
elif args.scheduler == 'cosine':
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=100, T_mult=1, eta_max=lr,  T_up=10, gamma=0.7)

best_plcc = 0
best_epoch = 0

for epoch in range(epochs):
    epoch_loss = 0
    epoch_plcc = 0

    model.train()
    for data, mean in tqdm(train_loader, desc='train'):
        data = data.to(device)
        # label = label.to(device)
        mean = mean.to(device).double()
        
        output = model(data).double().to(device)
        # output, xs = model(data)
        # output = output.double().to(device)
        mean = rearrange(mean, 'b -> b 1')
        loss = criterion(output, mean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        gt = mean
        plcc = calculate_plcc(output, gt)
        # kld = kl_divergence(output, gt)
        # print(kld)
        # exit()
        epoch_plcc += plcc / len(train_loader)
        epoch_loss += loss / len(train_loader)
        # acc = (output.argmax(dim=1) == label).float().mean()
        # epoch_accuracy += acc / len(train_loader)

    model.eval()
    with torch.no_grad():
        epoch_val_plcc = 0
        epoch_val_loss = 0
        for data, mean in tqdm(val_loader, desc='validation'):
            data = data.to(device)
            # label = label.to(device)
            mean = mean.to(device)
            mean = rearrange(mean, 'b -> b 1')

            val_output = model(data)
            # val_output, xs = model(data)
            val_output = val_output.double().to(device)

            val_loss = criterion(val_output, mean)

            # for i in range(4):
            #     x = xs[i]
            #     print(F.kl_div(F.log_softmax(x[i], 0), F.softmax(x[i], 0), reduction='none').mean())
            #     print(F.kl_div(F.log_softmax(x[0][i], 0), F.softmax(x[i], 0), reduction='none').mean())
            #     print(F.kl_div(F.log_softmax(x[i], 0), F.softmax(x[i], 0), reduction='none').mean())
            #     print(F.kl_div(F.log_softmax(x[i], 0), F.softmax(x[i], 0), reduction='none').mean())

            gt = mean
            val_plcc = calculate_plcc(val_output, gt)
            epoch_val_plcc += val_plcc / len(val_loader)
            epoch_val_loss += val_loss / len(val_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - plcc : {epoch_plcc:.4f} - val_loss : {epoch_val_loss:.4f} - val_plcc : {epoch_val_plcc:.4f}, lr: {lr:e}" #scheduler.get_lr()[0]
    )
    # print(
    #     f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - plcc : {epoch_plcc:.4f}"
    # )

    # create work dirs
    work_dir = './work_dirs/{}/'.format(args.work_dirs)
    if os.path.isdir(work_dir) != True:
        os.mkdir(work_dir)
    # save
    if os.path.isfile(work_dir+'logs.txt') == False:
        log_file = open(work_dir+'logs.txt', "w")
    with open(work_dir+'logs.txt', "a") as log:
        log_line = f"\"Epoch\": {epoch+1}, \"loss\": {epoch_loss:.4f}, \"plcc\": {epoch_plcc:.4f}, \"val_loss\": {epoch_val_loss:.4f}, \"val_plcc\": {epoch_val_plcc:.4f}, \"lr\": {lr:e}"
        log_line = '{'+log_line+'}\n'
        log.write(log_line)

    # save best plcc model
    if epoch_val_plcc >= best_plcc:
        best_plcc = epoch_val_plcc
        torch.save(model.state_dict(), work_dir+'best_plcc.pth'.format(epoch+1))
        best_epoch = epoch+1
        
    # save lastest model
    torch.save(model.state_dict(), work_dir+'latest.pth')
    
    print('Best PLCC epoch so far: ', best_epoch)
    # print('lr: ', scheduler.get_lr())

    if args.scheduler=='cosine' and epoch >= 100:
        scheduler.step()
    elif args.scheduler=='step':
        scheduler.step()
    else:
        pass