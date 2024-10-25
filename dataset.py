import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import imageio
import random
from einops import rearrange


class EDIQAData(Dataset):
    """Dataloader for EDIQA."""

    def __init__(self, data_type, file_list, csv_dir, mode='train'):
        self.file_list = file_list
        self.mode = mode
        self.data_type = data_type
        self.labels = pd.read_csv(csv_dir) # encoding="UTF-7"
        self.labels.index = self.labels['img']

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        # read image
        img_path = self.file_list[idx]
        img = imageio.imread(img_path)

        # get label
        if self.data_type == 'mayo':
            pid, lv, fid = img_path.split('/')[-3], img_path.split('/')[-2], img_path.split('/')[-1].split('.')[0]
            imgname = '{}_{}_{}.0'.format(pid, lv, fid)
            mean = self.labels.loc[imgname]['mean']
        elif self.data_type == 'mri':
            imgname = 'ABIDEII-GU_1_{}_session_1_anat_1_{}'.format(img_path.split('/')[-2], img_path.split('/')[-1].split('.')[0])
            mean = self.labels.loc[imgname].values[1]
        elif self.data_type == 'phantom':
            imgname = img_path.split('/')[-1]
            _, _, level1, level2, fid = imgname.split('_')
            imgname = 'ge_chest_{}_{}_{}'.format(fid.split('.')[0], level1, level2)
            mean = self.labels.loc[imgname]['mean']

        # NCC
        # mean = self.labels[self.labels['path']=='/media/wonkyong/268c4ce6-d8a7-453b-9607-b1a121f48743/data/MayoProjection/Patient_Noise/Scaled_Norm_Recon/{}/{}/{}.tiff'.format(pid, lv, fid)]['score']
        # mean = mean.values[0]

        # image augmentation
        if self.mode == 'train':
            if random.random() >= 0.5: 
                img = np.flip(img, 1)
            if random.random() >= 0.5:
                img = np.flip(img, 0)

        if np.max(img) > 1:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))

        if len(img.shape) == 2:
            img = np.stack((img,)*3, axis=-1)
        img = torch.from_numpy(img.copy())
        img = rearrange(img, 'h w c -> c h w')

        if self.mode == 'train':
            return img, mean
        return img, mean, imgname