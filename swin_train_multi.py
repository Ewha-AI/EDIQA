import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import random
from scipy.stats import spearmanr, pearsonr

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from collections import OrderedDict

from utils import seed_everything, get_distribution, distribution_to_score, calculate_plcc, CosineAnnealingWarmUpRestarts
from models.build import build_model
from dataset import MayoDataset, MayoRandomPatchDataset2, MayoRandomPatchDataset
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
parser.add_argument('--epochs', type=int, default=100, help='max epoch')
parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for adam optimizer')
parser.add_argument('--T_max', type=int, default=50, help='T_max for cosine2 scheduler')
parser.add_argument('--step', type=int, default=10, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='learning rate')
parser.add_argument('--gid', type=int, default=0, help='GPU ids')
parser.add_argument('--model_type', type=str, default='multi_swin', help='model type')
parser.add_argument('--criterion', type=str, default='mae', help='Loss function to use')
parser.add_argument('--val_pid', type=str, nargs="*", default=['L179', 'L186', 'L187', 'L125'], help='patient ids for validation set')
args = parser.parse_args()


# Training settings
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
gamma = args.gamma
step = args.step
device = 'cuda:{}'.format(args.gid)

# set seed
seed = 42
seed_everything(seed)

# load data
total_pid = ['L096', 'L291', 'L310', 'L109', 'L143', 'L192', 'L286']
# total_pid = sorted(os.listdir('../../data/nimg_3ch'))
total_pid = [pid for pid in total_pid if pid[0]=='L']
test_pid = ['L067', 'L506']
val_pid = args.val_pid
train_pid = [pid for pid in total_pid if pid not in val_pid]
train_pid = [pid for pid in train_pid if pid not in test_pid]

# check if all pids are loaded
assert len(set(train_pid + val_pid + test_pid)) == 10 #57

# load label
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

# create work dirs
work_dir = './work_dirs/{}/'.format(args.work_dirs)
if os.path.isdir(work_dir) != True:
    os.mkdir(work_dir)

# save train setup
with open(work_dir+'setup.txt', "a") as log:
    log_line = 'args:{}\ntrain_pid:{}\nval_pid:{}\ntest_pid:{}\ntrain_imgs:{}\nval_imgs:{}\ntest_imgs:{}'.format(args, train_pid, val_pid, test_pid,
                                                                                                                 len(train_list), len(val_list), len(test_list))
    log.write(log_line)


# load datasets
train_data = MayoDataset(train_list, label_dir, transform='train', norm=False)
train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True, num_workers=4)
val_data = MayoDataset(val_list, label_dir, transform='val', norm=False)
val_loader = DataLoader(dataset = val_data, batch_size=batch_size, shuffle=False, num_workers=4)
test_data = MayoDataset(test_list, label_dir, transform='val', norm=False)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=False, num_workers=4)

# set model
# model = SwinTransformer(feature_num=args.feature_num, mlp_head = args.mlp_head)
# model = SwinTransformer(num_classes=1)
# model = model.to(device)
model = build_model(args.model_type)
model = torch.nn.DataParallel(model).cuda()

# training setting
if args.criterion == 'mae':
    criterion = nn.L1Loss()
elif args.criterion == 'mse':
    criterion = nn.MSELoss()
else:
    raise ValueError('Criterion can only be mae or mse.')


optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
# optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
if args.scheduler == 'step':
    scheduler = StepLR(optimizer, step_size=step, gamma=gamma)
elif args.scheduler == 'cosine':
    optimizer = optim.Adam(model.parameters(), lr=0, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=100, T_mult=1, eta_max=lr,  T_up=10, gamma=0.5)
elif args.scheduler == 'cosine2':
    # optimizer = optim.Adam(model.parameters(), lr=0) #, weight_decay=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=0)
elif args.scheduler == 'plateau':
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

best_loss = 100
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
    swin_checkpoint = torch.load('/data1/wonkyong/workspace/TmIQA/work_dirs/swin_tiny_patch4_window7_224.pth')
    conv_checkpoint = torch.load('/data1/wonkyong/workspace/TmIQA/work_dirs/convnext_tiny_22k_224.pth')
    
    del swin_checkpoint['model']['head.bias']
    del swin_checkpoint['model']['head.weight'] 
    del conv_checkpoint['model']['head.bias']
    del conv_checkpoint['model']['head.weight']   

    del swin_checkpoint['model']['norm.weight']
    del swin_checkpoint['model']['norm.bias']
    del conv_checkpoint['model']['norm.weight']
    del conv_checkpoint['model']['norm.bias']

    # for key in ['norm.weight', 'norm.bias']:
    #     swin_checkpoint['model'][key.replace('norm.', 'norm3.')] = swin_checkpoint['model'].pop(key)

    swin_new = OrderedDict()
    conv_new = OrderedDict()

    for key, value in swin_checkpoint['model'].items():
        swin_new['module.0.swin_transformer.' + key] = value
    for key, value in conv_checkpoint['model'].items():
        conv_new['module.0.convnext.' + key] = value

    swin_new.update(conv_new)

    model.load_state_dict(swin_new, strict=False)

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

elif args.transfer == 'swin_detection_conv_imgnet':
    print('detection weights')
    swin_checkpoint = torch.load('../Swin-Transformer-Object-Detection/work_dirs/cascade_mask_rcnn_detDataset_1_3/epoch_100.pth', map_location='cuda:{}'.format(args.gid))

    del swin_checkpoint['meta']
    del swin_checkpoint['optimizer']

    s_layers = [c for c in swin_checkpoint['state_dict'].keys() if 'backbone' not in c]

    for l in s_layers:
        del swin_checkpoint['state_dict'][l]

    conv_checkpoint = torch.load('/data1/wonkyong/workspace/TmIQA/work_dirs/convnext_tiny_22k_224.pth')
    
    del conv_checkpoint['model']['head.bias']
    del conv_checkpoint['model']['head.weight']   
    del conv_checkpoint['model']['norm.weight']
    del conv_checkpoint['model']['norm.bias']

    # for key in ['norm.weight', 'norm.bias']:
    #     swin_checkpoint['model'][key.replace('norm.', 'norm3.')] = swin_checkpoint['model'].pop(key)

    conv_new = OrderedDict()
    for key, value in conv_checkpoint['model'].items():
        conv_new['module.0.convnext.' + key] = value

    for key in list(swin_checkpoint['state_dict'].keys()):
        if 'backbone' in key:
            swin_checkpoint['state_dict'][key.replace('backbone.', 'module.0.swin_transformer.')] = swin_checkpoint['state_dict'].pop(key)

    swin_checkpoint['state_dict'].update(conv_new)
    model.load_state_dict(swin_checkpoint['state_dict'], strict=False)

elif args.transfer == 'resume':
    checkpoint = torch.load('./work_dirs/{}/latest.pth'.format(args.work_dirs))
    model.load_state_dict(checkpoint, strict=True)
    # best_epoch = pd.read_json('./work_dirs/{}/logs.txt'.format(args.work_dirs), lines=True)
    # print(best_epoch)
    # exit()
    best_plcc = 75.0894 # 21_2
    start_epoch = 250

else:
    print('Train from scratch')


for epoch in range(start_epoch, epochs):
    epoch_loss = 0

    model.train()
    trainloader = tqdm(train_loader, desc='train')
    total_output, total_mean = [], []
    for i, (data, mean) in enumerate(trainloader):
        # for i in range(len(data)):
        #     data[i] = data[i].cuda()
        data = data.cuda()
        mean = mean.cuda()
        
        output = model(data).double()
        # mean = rearrange(mean, 'b -> b 1')
        output = rearrange(output, 'b 1 -> b')

        total_output = total_output + output.detach().cpu().tolist()
        total_mean = total_mean + mean.detach().cpu().tolist()

        loss = criterion(output, mean)
        epoch_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plcc = pearsonr(total_output, total_mean)[0]
    srocc = spearmanr(total_output, total_mean)[0]
    epoch_loss = epoch_loss / len(train_loader)

    # print PLCC and loss every epoch
    trainloader.set_description("Loss %.04f \t\t PLCC %.04f \t\t SROCC %.04f \t\t Step %d/%d" % (epoch_loss, plcc, srocc, i, len(train_loader)))


    # validation
    model.eval()
    with torch.no_grad():
        total_val_output, total_mean = [], []
        # epoch_val_plcc = 0
        epoch_val_loss = 0
        for data, mean in tqdm(val_loader, desc='validation'):
            for i in range(len(data)):
                data[i] = data[i].cuda()
            # data = data.cuda()
            mean = mean.cuda()
            # mean = rearrange(mean, 'b -> b 1')

            # val_output = 0
            # for im in data:
            #     val_output += model(im.cuda())
            # val_output = val_output / len(data)
            val_output = model(data)
            # print(val_output)
            # exit()
            val_output = rearrange(val_output, 'b 1 -> b')
            val_loss = criterion(val_output, mean)
            epoch_val_loss += val_loss

            # epoch_val_loss += val_loss / len(val_loader)

            total_val_output += val_output.detach().cpu().tolist()
            total_mean += mean.detach().cpu().tolist()

        val_plcc = pearsonr(total_val_output, total_mean)[0]
        val_srocc = spearmanr(total_val_output, total_mean)[0]
        # epoch_val_plcc += val_plcc / len(val_loader)
        epoch_val_loss = epoch_val_loss / len(val_loader)

    # test
    model.eval()
    with torch.no_grad():
        # epoch_test_plcc = 0
        total_test_output, total_mean = [], []
        for data, mean in tqdm(test_loader, desc='test'):
            # for i in range(len(data)):
            #     data[i] = data[i].cuda()
            data = data.cuda()
            mean = mean.cuda()
            # mean = rearrange(mean, 'b -> b 1')

            # test_output = model(data)
            # test_output = 0
            # for im in data:
            #     test_output += model(im.cuda())
            # test_output = test_output / len(data)
            test_output = model(data)
            test_output = rearrange(test_output, 'b 1 -> b')

            # test_plcc = calculate_plcc(test_output, mean)
            # epoch_test_plcc += test_plcc / len(test_loader)

            total_test_output += test_output.detach().cpu().tolist()
            total_mean += mean.detach().cpu().tolist()

        test_plcc = pearsonr(total_test_output, total_mean)[0]
        test_srocc = spearmanr(total_test_output, total_mean)[0]


    # save best srocc model
    if val_srocc >= best_plcc:
        best_plcc = val_srocc
        torch.save(model.state_dict(), work_dir+'best_srocc.pth'.format(epoch+1))
        # best_epoch = epoch+1

    # save best loss model
    if epoch_val_loss <= best_loss:
        best_loss = epoch_val_loss
        torch.save(model.state_dict(), work_dir+'best_loss.pth'.format(epoch+1))
        best_epoch = epoch+1
        
    # save lastest model
    torch.save(model.state_dict(), work_dir+'latest.pth')

    # save logs
    try:
        print(
            f"Epoch : {epoch+1} - train_loss : {epoch_loss:.4f} - train_plcc : {plcc:.4f} - train_srocc : {srocc:.4f} - val_loss : {epoch_val_loss:.4f} - val_plcc : {val_plcc:.4f} - val_srocc : {val_srocc:.4f} - test_plcc : {test_plcc:.4f} - test_srocc : {test_srocc:.4f} - lr: {scheduler.get_last_lr()[0]:e} - best_epoch: {best_epoch}"
        )

        # save log
        if os.path.isfile(work_dir+'logs.txt') == False:
            log_file = open(work_dir+'logs.txt', "w")
        with open(work_dir+'logs.txt', "a") as log:
            log_line = f"\"Epoch\": {epoch+1}, \"train_loss\": {epoch_loss:.4f}, \"train_plcc\": {plcc:.4f}, \"train_srocc\": {srocc:.4f}, \"val_loss\": {epoch_val_loss:.4f}, \"val_plcc\": {val_plcc:.4f}, \"val_srocc\": {val_srocc:.4f}, \"test_plcc\": {test_plcc:.4f}, \"test_srocc\": {test_srocc:.4f}, \"lr\": {scheduler.get_last_lr()[0]:e}, \"best_epoch\": {best_epoch}"
            log_line = '{'+log_line+'}\n'
            log.write(log_line)
    except:
        print(
        f"Epoch : {epoch+1} - train_loss : {epoch_loss:.4f} - train_plcc : {plcc:.4f} - train_srocc : {srocc:.4f} - val_loss : {epoch_val_loss:.4f} - val_plcc : {val_plcc:.4f} - val_srocc : {val_srocc:.4f} - test_plcc : {test_plcc:.4f} - test_srocc : {test_srocc:.4f} - lr: {optimizer.param_groups[0]['lr']:e} - best_epoch: {best_epoch}"
        )


        # save log
        if os.path.isfile(work_dir+'logs.txt') == False:
            log_file = open(work_dir+'logs.txt', "w")
        with open(work_dir+'logs.txt', "a") as log:
            log_line = f"\"Epoch\": {epoch+1}, \"train_loss\": {epoch_loss:.4f}, \"train_plcc\": {plcc:.4f}, \"train_srocc\": {srocc:.4f}, \"val_loss\": {epoch_val_loss:.4f}, \"val_plcc\": {val_plcc:.4f}, \"val_srocc\": {val_srocc:.4f}, \"test_plcc\": {test_plcc:.4f}, \"test_srocc\": {test_srocc:.4f}, \"lr\": {optimizer.param_groups[0]['lr']:e}, \"best_epoch\": {best_epoch}"
            log_line = '{'+log_line+'}\n'
            log.write(log_line)
    

    # scheduler step
    # print('lr: ', scheduler.get_lr())
    if args.scheduler == None:
        pass
    elif args.scheduler == 'plateau':
        scheduler.step(epoch_val_loss)
    else:
        scheduler.step()