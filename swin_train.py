import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

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

from models.swin_transformer import SwinTransformer

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def rank_loss(d, y, num_images, eps=1e-6, norm_num=True): # prediction, target
    loss = torch.zeros(1).cuda() #, device=device)
    if num_images < 2:
        return loss

    dp = torch.abs(d)
    combinations = torch.combinations(torch.arange(num_images), 2)
    combinations_count = max(1, len(combinations))

    for i, j in combinations:
        rl = torch.clamp_min(-(y[i] - y[j]) * (d[i] - d[j]) / (torch.abs(y[i] - y[j]) + eps), min=0)
        loss += rl / max(dp[i], dp[j])  # normalize by maximum value
    if norm_num:
        loss = loss / combinations_count  # mean

    return loss



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
epochs = 300
lr = args.lr
gamma = args.gamma
step = args.step
seed = 42 #42

seed_everything(seed)
device = 'cuda:{}'.format(args.gid)

# load data
total_pid = sorted(os.listdir('../../data/nimg-train-3channel'))
total_pid = [pid for pid in total_pid if pid[0]=='L']
val_pid = [args.val_pid]
train_pid = [pid for pid in total_pid if pid not in val_pid]
test_pid = ['L067', 'L506']
# train_pid = ['L096', 'L291', 'L310', 'L109', 'L143', 'L192', 'L286']
# val_pid = ['L333']

label_dir = '../../data/nimg-train-3channel/mayo25yp_total.csv'
test_label_dir = '../../data/nimg-test-3channel/mayo_test.csv'

temp_train_list = []
for pid in train_pid:
    temp_train_list.append(glob('../../data/nimg-train-3channel/{}/*/*.tiff'.format(pid)))
temp_val_list = []
for pid in val_pid:
    temp_val_list.append(glob('../../data/nimg-train-3channel/{}/*/*.tiff'.format(pid)))
temp_test_list = []
for pid in test_pid:
    temp_test_list.append(glob('../../data/nimg-train-3channel/{}/*/*.tiff'.format(pid)))


train_list, val_list, test_list = [], [], []
for i in range(len(temp_train_list)):
    train_list += temp_train_list[i]
for i in range(len(temp_val_list)):
    val_list += temp_val_list[i]
for i in range(len(temp_test_list)):
    test_list += temp_test_list[i]
train_list = sorted(train_list)
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

best_plcc = 0
best_epoch = 0
start_epoch = 0

# transfer weights
if args.transfer == 'detection':
    checkpoint = torch.load('../Swin-Transformer-Object-Detection/work_dirs/cascade_mask_rcnn_detDataset_1_3/epoch_100.pth', map_location='cuda:{}'.format(args.gid))

    del checkpoint['meta']
    del checkpoint['optimizer']

    #  swin transformer
    layers = [c for c in checkpoint['state_dict'].keys() if 'backbone' not in c]
    # fpn
    # layers = [c for c in layers if 'neck' not in c]

    for l in layers:
        del checkpoint['state_dict'][l]

    for key in list(checkpoint['state_dict'].keys()):
        if 'backbone' in key:
            checkpoint['state_dict'][key.replace('backbone.', 'module.0.')] = checkpoint['state_dict'].pop(key)
        elif 'neck' in key:
            checkpoint['state_dict'][key.replace('neck.', 'module.1.')] = checkpoint['state_dict'].pop(key)
    
    for key in list(checkpoint['state_dict'].keys()):
        if '.conv.' in key:
            checkpoint['state_dict'][key.replace('.conv.', '.')] = checkpoint['state_dict'].pop(key)

    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    # for name, param in model[0].named_parameters():
    #     print(name, param.requires_grad)
    # exit()

elif args.transfer == 'imagenet':
    checkpoint = torch.load('/data1/wonkyong/workspace/TmIQA/work_dirs/swin_tiny_patch4_window7_224.pth')
    
    del checkpoint['model']['head.bias']
    del checkpoint['model']['head.weight']   

    # del checkpoint['model']['norm.weight']
    # del checkpoint['model']['norm.bias']

    for key in ['norm.weight', 'norm.bias']:
        checkpoint['model'][key.replace('norm.', 'norm3.')] = checkpoint['model'].pop(key)

    model.load_state_dict(checkpoint['model'], strict=False)

elif args.transfer == 'swin_conv_detection':
    print('detection weights')
    swin_checkpoint = torch.load('../Swin-Transformer-Object-Detection/work_dirs/cascade_mask_rcnn_detDataset_1_3/epoch_100.pth', map_location='cuda:{}'.format(args.gid))
    conv_checkpoint = torch.load('/data1/wonkyong/workspace/Swin-Transformer-Object-Detection/work_dirs/cascade_mask_rcnn_convnext/epoch_100.pth', map_location='cuda:{}'.format(args.gid))

    del swin_checkpoint['meta']
    del swin_checkpoint['optimizer']
    del conv_checkpoint['meta']
    del conv_checkpoint['optimizer']

    s_layers = [c for c in swin_checkpoint['state_dict'].keys() if 'backbone' not in c]
    c_layers = [c for c in conv_checkpoint['state_dict'].keys() if 'backbone' not in c]

    for l in s_layers:
        del swin_checkpoint['state_dict'][l]
    for l in c_layers:
        del conv_checkpoint['state_dict'][l]

    print(conv_checkpoint.keys())

    for key in list(swin_checkpoint['state_dict'].keys()):
        if 'backbone' in key:
            swin_checkpoint['state_dict'][key.replace('backbone.', 'module.0.swin_transformer.')] = swin_checkpoint['state_dict'].pop(key)

    for key in list(conv_checkpoint['state_dict'].keys()):
        if 'backbone' in key:
            conv_checkpoint['state_dict'][key.replace('backbone.', 'module.0.convnext.')] = conv_checkpoint['state_dict'].pop(key)

    
    swin_checkpoint['state_dict'].update(conv_checkpoint['state_dict'])
    model.load_state_dict(swin_checkpoint['state_dict'], strict=False)

elif args.transfer == 'resume':
    checkpoint = torch.load('./work_dirs/{}/latest.pth'.format(args.work_dirs))
    model.load_state_dict(checkpoint, strict=True)
    # best_epoch = pd.read_json('./work_dirs/{}/logs.txt'.format(args.work_dirs), lines=True)
    # print(best_epoch)
    # exit()
    best_plcc = 75.0894 # 21_2
    start_epoch = 250

for epoch in range(start_epoch, epochs):
    epoch_loss = 0
    epoch_plcc = 0

    model.train()
    for data, mean in tqdm(train_loader, desc='train'):
        # data = data.to(device)
        # mean = mean.to(device).double()
        data = data.cuda()
        mean = mean.cuda()
        
        output = model(data).double()
        mean = rearrange(mean, 'b -> b 1')
        loss = criterion(output, mean)# + rank_loss(output, mean, len(output))
        # print(loss)
        # print(torch.is_tensor(loss))
        # print(int(loss[0]))
        # exit()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        gt = mean
        plcc = calculate_plcc(output, gt)
        epoch_plcc += plcc / len(train_loader)
        epoch_loss += loss / len(train_loader)
        # acc = (output.argmax(dim=1) == label).float().mean()
        # epoch_accuracy += acc / len(train_loader)

    # validation
    model.eval()
    with torch.no_grad():
        epoch_val_plcc = 0
        epoch_val_loss = 0
        for data, mean in tqdm(val_loader, desc='validation'):
            # data = data.to(device)
            # mean = mean.to(device)
            data = data.cuda()
            mean = mean.cuda()
            mean = rearrange(mean, 'b -> b 1')

            val_output = model(data)
            val_loss = criterion(val_output, mean)# + rank_loss(output, mean, len(output))

            gt = mean
            val_plcc = calculate_plcc(val_output, gt)
            epoch_val_plcc += val_plcc / len(val_loader)
            epoch_val_loss += val_loss / len(val_loader)

    # test
    model.eval()
    with torch.no_grad():
        epoch_test_plcc = 0
        for data, mean in tqdm(test_loader, desc='test'):
            # data = data.to(device)
            # mean = mean.to(device)
            data = data.cuda()
            mean = mean.cuda()
            mean = rearrange(mean, 'b -> b 1')

            test_output = model(data)
            gt = mean

            test_plcc = calculate_plcc(test_output, gt)
            epoch_test_plcc += test_plcc / len(test_loader)

    # create work dirs
    work_dir = './work_dirs/{}/'.format(args.work_dirs)
    if os.path.isdir(work_dir) != True:
        os.mkdir(work_dir)

    # save best plcc model
    if epoch_val_plcc >= best_plcc:
        best_plcc = epoch_val_plcc
        torch.save(model.state_dict(), work_dir+'best_plcc.pth'.format(epoch+1))
        best_epoch = epoch+1
        
    # save lastest model
    torch.save(model.state_dict(), work_dir+'latest.pth')

    # convert loss to int
    # if torch.is_tensor(loss):
    #     epoch_loss = float(epoch_loss[0])
    #     epoch_val_loss = float(epoch_val_loss[0])

    try:
        print(
            f"Epoch : {epoch+1} - train_loss : {epoch_loss:.4f} - train_plcc : {epoch_plcc:.4f} - val_loss : {epoch_val_loss:.4f} - val_plcc : {epoch_val_plcc:.4f} - test_plcc : {epoch_test_plcc:.4f} - lr: {scheduler.get_last_lr()[0]:e} - best_epoch: {best_epoch}"
        )

        # save log
        if os.path.isfile(work_dir+'logs.txt') == False:
            log_file = open(work_dir+'logs.txt', "w")
        with open(work_dir+'logs.txt', "a") as log:
            log_line = f"\"Epoch\": {epoch+1}, \"train_loss\": {epoch_loss:.4f}, \"train_plcc\": {epoch_plcc:.4f}, \"val_loss\": {epoch_val_loss:.4f}, \"val_plcc\": {epoch_val_plcc:.4f}, \"test_plcc\": {epoch_test_plcc:.4f}, \"lr\": {scheduler.get_last_lr()[0]:e}, \"best_epoch\": {best_epoch}"
            log_line = '{'+log_line+'}\n'
            log.write(log_line)
    except:
        print(
        f"Epoch : {epoch+1} - train_loss : {epoch_loss:.4f} - train_plcc : {epoch_plcc:.4f} - val_loss : {epoch_val_loss:.4f} - val_plcc : {epoch_val_plcc:.4f} - test_plcc : {epoch_test_plcc:.4f} - lr: {optimizer.param_groups[0]['lr']:e} - best_epoch: {best_epoch}"
        )


        # save log
        if os.path.isfile(work_dir+'logs.txt') == False:
            log_file = open(work_dir+'logs.txt', "w")
        with open(work_dir+'logs.txt', "a") as log:
            log_line = f"\"Epoch\": {epoch+1}, \"train_loss\": {epoch_loss:.4f}, \"train_plcc\": {epoch_plcc:.4f}, \"val_loss\": {epoch_val_loss:.4f}, \"val_plcc\": {epoch_val_plcc:.4f}, \"test_plcc\": {epoch_test_plcc:.4f}, \"lr\": {optimizer.param_groups[0]['lr']:e}, \"best_epoch\": {best_epoch}"
            log_line = '{'+log_line+'}\n'
            log.write(log_line)
    
    # print('lr: ', scheduler.get_lr())

    if args.scheduler=='cosine': #and epoch >= 100:
        scheduler.step()
    elif args.scheduler == 'step':
        scheduler.step()