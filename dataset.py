import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from utils import get_distribution
import numpy as np
import imageio
import cv2
import random
from einops import rearrange
import torchvision
import torchvision.transforms as transforms

class MayoDataset(Dataset):
    def __init__(self, file_list, csv_dir, transform='train', norm=False):
        self.file_list = file_list
        self.transform = transform
        self.labels = pd.read_csv(csv_dir)
        self.labels.index = self.labels['img']
        self.norm = norm

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        # read image
        img_path = self.file_list[idx]
        img = imageio.imread(img_path)

        # get label
        pid, lv, fid = img_path.split('/')[-3], img_path.split('/')[-2], img_path.split('/')[-1][:-5]
        imgname = '{}_{}_{}'.format(pid, lv, fid)
        mean = self.labels.loc[imgname].values[1]

        # transform
        transformer = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [1, 1, 1])
        ])

        # transform image
        if self.transform is 'train':
            img = cv2.resize(img, (224, 224))
            if random.random() >= 0.5:
                img = np.flip(img, 1)
            img = torch.from_numpy(img.copy())
            img = rearrange(img, 'h w c -> c h w')
        else:
            img = cv2.resize(img, (224, 224))
            img = torch.from_numpy(img.copy())
            img = rearrange(img, 'h w c -> c h w')

        if self.norm == True:
            img = transformer(img)

        return img, mean



class MayoDistDataset(Dataset):
    def __init__(self, file_list, csv_dir, transform='train', norm=False):
        self.file_list = file_list
        self.transform = transform
        self.csv_dir = csv_dir
        self.mos_scale = list(range(1,11))
        self.norm = norm

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        # read image
        img_path = self.file_list[idx]
        img = imageio.imread(img_path)
        # convert image channel
        img = np.stack((img,)*3, axis=-1)

        # get label
        label_df = pd.read_csv(self.csv_dir)
        label_df.index = label_df['img']
        pid, lv, fid = img_path.split('/')[-3], img_path.split('/')[-2], img_path.split('/')[-1][:-5]
        imgname = '{}_{}_{}'.format(pid, lv, fid)
        mean = label_df.loc[imgname].values[1]
        # std = label_df.loc[imgname].values[2]
        # label = get_distribution(self.mos_scale, mean, std)
        # label = torch.from_numpy(np.asarray(label))

        # transformer
        transformer = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [1, 1, 1])
        ])

        # transform image
        if self.transform is 'train':
            img = cv2.resize(img, (224, 224))
            if random.random() >= 0.5:
                img = np.flip(img, 1)
            img = torch.from_numpy(img.copy())
            img = rearrange(img, 'h w c -> c h w')
        else:
            img = cv2.resize(img, (224, 224))
            img = torch.from_numpy(img.copy())
            img = rearrange(img, 'h w c -> c h w')

        if self.norm == True:
            img = transformer(img)

        # return img, label, mean
        return img, mean


class PIPALDataset(Dataset):
    def __init__(self, file_list, csv_dir, transform=None, norm=False):
        self.file_list = file_list
        self.transform = transform
        self.labels = pd.read_csv(csv_dir)
        self.labels.index = self.labels['dst_img']
        self.norm = norm

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        # read image
        img_path = self.file_list[idx]
        img = Image.open(img_path)

        # get label
        imgname = img_path.split('/')[-1]
        mean = self.labels.loc[imgname].values[-1]
        dist = self.labels.loc[imgname].values[0]

        # transform image
        if self.transform != None:
            img = self.transform(img)

        return img, mean