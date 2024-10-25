import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import requests

from torch.utils.data import DataLoader
from collections import OrderedDict

from models.build import build_model
from dataset import EDIQAData
from glob import glob
from tqdm import tqdm
import numpy as np
import os
from einops import rearrange
import argparse
import json
import pandas as pd

from scipy.stats import pearsonr, spearmanr


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def download_model(url, save_path, gdown=False):
    print(f"Downloading model from {url}...")
    response = requests.get(url)
    if not gdown:
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Model saved to {save_path}")
        else:
            print(f"Failed to download model. Status code: {response.status_code}")
    else:
        gdown.download(url, save_path, fuzzy=True)
        print(f"Model saved to {save_path}")

def load_model(model_type, data_type, transfer):
    # set model
    model = build_model(model_type, data_type)
    model = torch.nn.DataParallel(model).cuda()

    # transfer weights
    if transfer == 'imagenet':
        print('load imagenet weights')

        if not os.path.isdir('work_dirs/detection_weights'):
            os.makedirs('work_dirs/detection_weights')
        if not os.path.isdir('work_dirs/detection_weights/imagenet'):
            os.makedirs('work_dirs/detection_weights/imagenet')
            download_model(url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
                        save_path='work_dirs/detection_weights/imagenet/swin_tiny_patch4_window7_224.pth')
            download_model(url='https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth',
                        save_path='work_dirs/detection_weights/imagenet/convnext_tiny_22k_224.pth')
        
        swin_checkpoint = torch.load('work_dirs/detection_weights/imagenet/swin_tiny_patch4_window7_224.pth')
        conv_checkpoint = torch.load('work_dirs/detection_weights/imagenet/convnext_tiny_22k_224.pth')
        
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

    elif transfer == 'detection':
        print('detection weights')

        # TODO: add downloading process
        # if not os.path.isdir('work_dirs/detection_weights_'):
        #     os.makedirs('work_dirs/detection_weights_')
        # if not os.path.isdir('work_dirs/detection_weights_/{}'.format(data_type)):
        #     os.makedirs('work_dirs/detection_weights_/{}'.format(data_type))
        #     download_model(url='https://drive.google.com/file/d/1wK6km6t5nXO4rG_jaUg0Oyh5U2fLfPk9/view?usp=drive_link', # 'https://drive.google.com/uc?id=1wK6km6t5nXO4rG_jaUg0Oyh5U2fLfPk9',
        #                 save_path='work_dirs/detection_weights_/{}/swin.pth'.format(data_type))
        #     download_model(url= 'https://drive.google.com/file/d/1tmdTb5-dRsUVwKsKDitpGmZMZec1wp5I/view?usp=drive_link', # 'https://drive.google.com/uc?id=1tmdTb5-dRsUVwKsKDitpGmZMZec1wp5I',
        #                 save_path='work_dirs/detection_weights_/{}/conv.pth'.format(data_type))

        swin_checkpoint = torch.load('work_dirs/detection_weights/{}/swin.pth'.format(data_type), map_location='cpu') 
        conv_checkpoint = torch.load('work_dirs/detection_weights/{}/conv.pth'.format(data_type), map_location='cpu')

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

    else:
        print('Train from scratch')

    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TIQA')
    parser.add_argument('--work_dirs', type=str, required=True, help='work directory name to save model')
    parser.add_argument('--label_dir', type=str, default='/media/wonkyong/hdd2/data/D2IQA/mayo/mayo57np.csv', help='path to data label file')
    parser.add_argument('--img_path', type=str, default='/media/wonkyong/268c4ce6-d8a7-453b-9607-b1a121f48743/data/nimg_train/*/*/*.tiff')
    parser.add_argument('--transfer', type=str, default=None, help='whether to trasfer weights|Options: detection, coco, none')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size') #
    parser.add_argument('--epochs', type=int, default=100, help='max epoch')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='weight decay for adam optimizer')
    parser.add_argument('--T_max', type=int, default=50, help='T_max for cosine scheduler') #
    parser.add_argument('--model_type', type=str, default='ediqa', help='model type')
    parser.add_argument('--data_type', type=str, default='mayo', help='data type [mayo|ncc|phantom|mri]')
    args = parser.parse_args()

    # create work dirs
    work_dir = './work_dirs/{}/'.format(args.work_dirs)
    if os.path.isdir(work_dir) != True:
        os.mkdir(work_dir)

    # Training settings
    seed = 42
    seed_everything(seed)

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    criterion = nn.MSELoss()
    model = load_model(model_type=args.model_type, data_type=args.data_type, transfer=args.transfer)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=0) #, weight_decay=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=0)

    if args.data_type == 'mayo':
        # load data
        total_pid = ['L096', 'L291', 'L310', 'L109', 'L143', 'L192', 'L286'] # Total 13342
        # total_pid = sorted(os.listdir('../../data/nimg_3ch'))
        # total_pid = [pid for pid in total_pid if pid[0]=='L']
        val_pid = ['L333']
        test_pid = ['L067', 'L506']
        train_pid = [pid for pid in total_pid if pid not in val_pid]
        train_pid = [pid for pid in train_pid if pid not in test_pid]

        # check if all pids are loaded
        # assert len(set(train_pid + val_pid + test_pid)) == 10 #57

        # load label
        # label_dir = '/media/wonkyong/hdd2/data/D2IQA/mayo/mayo57np.csv'
        # test_label_dir = '../../data/nimg-test-3channel/mayo_test.csv'

        train_list = sorted(glob(args.img_path))
        train_list = [im for im in train_list if im.split('/')[-3] in train_pid]
        val_list = sorted(glob(args.img_path))
        val_list = [im for im in val_list if im.split('/')[-3] in val_pid]

    elif args.data_type == 'phantom':
        data_list = pd.read_csv(args.label_dir)['img']
        data_list = set([im.split('_')[2] for im in data_list])
        data = sorted(glob(args.img_path))
        data = [im for im in data if im.split('/')[-1].split('_')[-1].split('.')[0] in data_list]
        random.shuffle(data)
        train_list = sorted(data[:int(len(data)*0.8)])
        val_list = sorted(data[int(len(data)*0.8):])

    elif args.data_type == 'mri':
        # /media/wonkyong/268c4ce6-d8a7-453b-9607-b1a121f48743/data/ABIDEII_GU_1/ABIDEII-GU_3ch
        total_pid = pd.read_csv(args.label_dir)['img']
        total_pid = sorted(set([pid.split('_')[2] for pid in total_pid]))
        val_pid = ['28788', '28827', '28767', '28820', '28748', '28818', '28760', '28744',  
                   '28807', '28838', '28749', '28811', '28776', '28774', '28782', '28769', 
                   '28752', '28790', '28813']
        test_pid = ['28746', '28750', '28754', '28764', '28768', '28780', '28789', '28792', 
                    '28796', '28810', '28830', '28847']
        train_pid = [pid for pid in total_pid if pid not in val_pid]
        train_pid = [pid for pid in train_pid if pid not in test_pid]
        assert len(total_pid) == len(train_pid) + len(val_pid) + len(test_pid)

        data = sorted(glob(args.img_path))
        train_list = [im for im in data if im.split('/')[-2] in train_pid]
        val_list = [im for im in data if im.split('/')[-2] in val_pid]

    else:
        raise NotImplementedError(f"Unkown data_type: {args.data_type}")

    print('================================')
    print('    Dataset')
    print('--------------------------------')
    print('    Train: ', len(train_list))
    print('    Validation: ', len(val_list))
    print('================================')

    # load datasets
    train_data = EDIQAData(args.data_type, train_list, args.label_dir, mode='train')
    train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_data = EDIQAData(args.data_type, val_list, args.label_dir, mode='test')
    val_loader = DataLoader(dataset = val_data, batch_size=batch_size, shuffle=False, num_workers=4)

    best_loss = np.inf
    start_epoch = 0
    log = {}

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        epoch_plcc = 0

        model.train()
        trainloader = tqdm(train_loader, desc='train')
        total_output, total_mean = [], []

        for data, mean in trainloader:
            data = data.cuda()
            mean = mean.cuda()
            
            output = model(data).double()
            output = rearrange(output, 'b 1 -> b')

            total_output = total_output + output.detach().cpu().tolist()
            total_mean = total_mean + mean.detach().cpu().tolist()

            loss = criterion(output, mean)
            epoch_loss += loss
            # epoch_plcc += pearsonr(output, mean)[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainloader.set_postfix(loss=loss.item())

        plcc = pearsonr(total_output, total_mean)[0]
        srocc = spearmanr(total_output, total_mean)[0]
        # plcc = epoch_plcc / len(train_loader)
        epoch_loss = epoch_loss / len(train_loader)

        # print PLCC and loss every epoch
        # trainloader.set_description("Loss %.04f \t\t PLCC %.04f \t\t SROCC %.04f \t\t Step %d/%d" % (epoch_loss, plcc, srocc, i, len(train_loader)))

        # validation
        model.eval()
        validloader = tqdm(val_loader, desc='val')
        total_val_output, total_mean = [], []

        with torch.no_grad():
            epoch_val_loss = 0
            for data, mean, _ in validloader:
                data = data.cuda()
                mean = mean.cuda()

                val_output = model(data).double()
                val_output = rearrange(val_output, 'b 1 -> b')

                val_loss = criterion(val_output, mean)
                epoch_val_loss += val_loss
                # epoch_val_plcc += calculate_plcc(val_output, mean)

                validloader.set_postfix(loss=loss.item())

                total_val_output += val_output.detach().cpu().tolist()
                total_mean += mean.detach().cpu().tolist()

            val_plcc = pearsonr(total_val_output, total_mean)[0]
            val_srocc = spearmanr(total_val_output, total_mean)[0]
            epoch_val_loss = epoch_val_loss / len(val_loader)

        # save best loss model
        if epoch_val_loss <= best_loss:
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), work_dir+'best_loss.pth')
        # save lastest model
        torch.save(model.state_dict(), work_dir+'latest.pth')

        # log arguments
        log['epoch'] = epoch + 1
        log['train_loss'] = epoch_loss.item()
        log['train_plcc'] = plcc
        log['train_srocc'] = srocc
        log['val_loss'] = epoch_val_loss.item()
        log['val_plcc'] = val_plcc
        log['val_srocc'] = val_srocc
        print(log)
        with open(os.path.join(work_dir, 'log.jsonl'), 'a') as log_file:
            json.dump(log, log_file)
            log_file.write('\n')
        

        # scheduler step
        # print('lr: ', scheduler.get_lr())
        scheduler.step()