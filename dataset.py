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
        # print(idx)
        img_path = self.file_list[idx]
        # print(img_path)
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
        if self.transform == 'train':
            img = cv2.resize(img, (224, 224))
            # center crop
            # img = img[144: 144+224, 144: 144+224]
            # random crop
            # x = random.randint(100, 189)
            # y = random.randint(100, 189)
            # x = random.randint(0, 288)
            # y = random.randint(0, 288)
            # img = img[x: x+224, y: y+224]
            # imageio.imsave('random_crop_test_{}_{}.png'.format(x, y), img)
            if random.random() >= 0.5:
                img = np.flip(img, 1)
            if random.random() >= 0.5:
                img = np.flip(img, 0)
            img = torch.from_numpy(img.copy())
            img = rearrange(img, 'h w c -> c h w')
        else:
            img = cv2.resize(img, (224, 224))
            img = torch.from_numpy(img.copy())
            img = rearrange(img, 'h w c -> c h w')
            # center crop
            # img = img[144: 144+224, 144: 144+224]
            # img = torch.from_numpy(img.copy())
            # img = rearrange(img, 'h w c -> c h w')
            # imgs = []
            # for i in range(5):
            #     x = random.randint(0, 288)
            #     y = random.randint(0, 288)
            #     patch = img[x: x+224, y: y+224]
            #     patch = torch.from_numpy(patch.copy())
            #     patch = rearrange(patch, 'h w c -> c h w')
            #     imgs.append(patch)
            # img = tuple(imgs)

        if self.norm == True:
            img = transformer(img)

        return img, mean, imgname


class MayoRandomPatchDataset(Dataset):
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
        # print(idx)
        img_path = self.file_list[idx]
        # print(img_path)
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
        if self.transform == 'train':
            # random horizontal flip
            if random.random() >= 0.5:
                img = np.flip(img, 1)
            # random crop image
            x = random.randint(0, 288)
            y = random.randint(0, 288)
            img = img[x: x+224, y: y+224]
            img = torch.from_numpy(img.copy())
            img = rearrange(img, 'h w c -> c h w')
        else:
            patches = []
            # random crop image
            for i in range(5):
                patch = img.copy()
                x = random.randint(0, 288)
                y = random.randint(0, 288)
                patch = patch[x: x+224, y: y+224]
                patch = torch.from_numpy(patch.copy())
                patch = rearrange(patch, 'h w c -> c h w')
                patches.append(patch)
            img = tuple(patches)

        if self.norm == True:
            img = transformer(img)

        return img, mean

class MayoRandomPatchDataset2(Dataset):
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
        # print(idx)
        img_path = self.file_list[idx]
        # print(img_path)
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
            # random crop image
            x = random.randint(0, 288)
            y = random.randint(0, 288)
            img = img[x: x+224, y: y+224]
            # # random horizontal flip
            if random.random() >= 0.5:
                img = np.flip(img, 0)
            # # random vertical flip
            if random.random() >= 0.5:
                img = np.flip(img, 1)
            img = torch.from_numpy(img.copy())
            img = rearrange(img, 'h w c -> c h w')
        else:
            # crop image to 224 * 2 x 224 * 2
            h, _, _ = img.shape
            center = h // 2
            img = img[center-224:center+224, center-224:center+224]
            # split image to patches
            patch1 = img[:224, :224]
            patch2 = img[:224, 224:]
            patch3 = img[224:, :224]
            patch4 = img[224:, 224:]
            patch = [patch1, patch2, patch3, patch4]
            # points = [0, 144, 288]
            # patch = []
            # for x in points:
            #     for y in points:
            #         patch.append(img[x:x+224, y:y+224])
            for i in range(4):
                patch[i] = torch.from_numpy(patch[i].copy())
                patch[i] = rearrange(patch[i], 'h w c -> c h w')
            # img = torch.from_numpy(img.copy())
            # img = rearrange(img, 'h w c -> c h w')
            # random select patch (temp, FIX)
            # img = patch[random.randint(0,4)]
            img = tuple(patch)

        if self.norm == True:
            img = transformer(img)

        return img, mean


class MayoPatchDataset(Dataset):
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
        # print(idx)
        img_path = self.file_list[idx]
        # print(img_path)
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
            # crop image to 224 * 2 x 224 * 2
            # h, _, _ = img.shape
            # center = h // 2
            # img = img[center-224:center+224, center-224:center+224]
            x = random.randint(0, 63)
            y = random.randint(0, 63)
            img = img[x:x+448, y:y+448]
            # random horizontal flip
            if random.random() >= 0.5:
                img = np.flip(img, 1)
            # split image to patches
            patch1 = img[:224, :224]
            patch2 = img[:224, 224:]
            patch3 = img[224:, :224]
            patch4 = img[224:, 224:]
            patch = [patch1, patch2, patch3, patch4]
            for i in range(4):
                patch[i] = torch.from_numpy(patch[i].copy())
                patch[i] = rearrange(patch[i], 'h w c -> c h w')
            # img = torch.from_numpy(img.copy())
            # img = rearrange(img, 'h w c -> c h w')
            # random select patch (temp, FIX)
            # img = patch[random.randint(0,3)]
            img = tuple(patch)
        else:
            # crop image to 224 * 2 x 224 * 2
            h, _, _ = img.shape
            center = h // 2
            img = img[center-224:center+224, center-224:center+224]
            # split image to patches
            patch1 = img[:224, :224]
            patch2 = img[:224, 224:]
            patch3 = img[224:, :224]
            patch4 = img[224:, 224:]
            patch = [patch1, patch2, patch3, patch4]
            for i in range(4):
                patch[i] = torch.from_numpy(patch[i].copy())
                patch[i] = rearrange(patch[i], 'h w c -> c h w')
            # img = torch.from_numpy(img.copy())
            # img = rearrange(img, 'h w c -> c h w')
            # random select patch (temp, FIX)
            # img = patch[random.randint(0,4)]
            img = tuple(patch)

        if self.norm == True:
            img = transformer(img)

        return img, mean


class MayoPatchDataset2(Dataset):
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
        # print(idx)
        img_path = self.file_list[idx]
        # print(img_path)
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
            # random horizontal flip
            if random.random() >= 0.5:
                img = np.flip(img, 1)
            # random crop image
            x = random.randint(0, 288)
            y = random.randint(0, 288)
            img = img[x: x+224, y: y+224]
            img = torch.from_numpy(img.copy())
            img = rearrange(img, 'h w c -> c h w')
            # split image to patches
            # h, _, _ = img.shape
            # center = h//2
            # patches = []
            # patches.append(img[:224, :224])
            # patches.append(img[512-224:, :224])
            # patches.append(img[:224, 512-224:])
            # patches.append(img[512-224:, 512-224:])
            # patches.append(img[center-112:center+112, center-112:center+112])
            # img = tuple(patches)
        else:
            # split image to patches
            h, _, _ = img.shape
            center = h//2
            patches = []
            patches.append(img[:224, :224])
            patches.append(img[512-224:, :224])
            patches.append(img[:224, 512-224:])
            patches.append(img[512-224:, 512-224:])
            patches.append(img[center-112:center+112, center-112:center+112])
            for i in range(5):
                patches[i] = torch.from_numpy(patches[i].copy())
                patches[i] = rearrange(patches[i], 'h w c -> c h w')
            img = tuple(patches)

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