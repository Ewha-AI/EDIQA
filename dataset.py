import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import pandas as pd
from utils import get_distribution
import numpy as np
import imageio
import cv2
import random
from einops import rearrange
import torchvision
import torchvision.transforms as transforms
import nibabel as nib
import SimpleITK as sitk

def createMIP(np_img, slices_num = 15):
    ''' create the mip image from original image, slice_num is the number of 
    slices for maximum intensity projection
    https://github.com/ljpadam/maximum_intensity_projection/blob/master/maximum_intensity_projection.py'''
    img_shape = np_img.shape
    np_mip = np.zeros(img_shape)
    for i in range(img_shape[0]):
        start = max(0, i-slices_num)
        np_mip[i,:,:] = np.amax(np_img[start:i+1],0)
    return np_mip


class ABIDEData(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
    
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
        
    def __getitem__(self, idx):
        imgs = []
        img_path = self.file_list[idx]
        # print(img_path)
        nii_img = nib.load(img_path).get_fdata()
        # print(img.shape)
        # get center slice
        h, w, c = nii_img.shape
        for i in range(1):
            # img = nii_img[:,:,int(c/2)-15+i]

            #find out center slice
            slice_idx = []
            for ch in range(c):
                intst = np.mean(nii_img[:,:,ch])
                if intst != 0:
                    slice_idx.append(ch)
            mid = int(np.median(slice_idx))

            # img = nii_img[:,:,int(c/2)]
            img = nii_img[:,:,mid]
            # pad and crop image
            size = max(h, w)
            pimg = np.zeros((size, size))
            if h != w:
                try:
                    pimg[int(size/2)-int(h/2):int(size/2)+int(h/2)+1, int(size/2)-int(w/2):int(size/2)+int(w/2)] = img
                except:
                    pimg[int(size/2)-int(h/2):int(size/2)+int(h/2), int(size/2)-int(w/2):int(size/2)+int(w/2)+1] = img
                img = pimg[int(size/2)-112:int(size/2)+112,int(size/2)-112:int(size/2)+112]
                # imageio.imsave('ptest.png', img)
                # print(img.shape)
                # exit()
            # normalize image
            else:
                img = img[int(size/2)-112:int(size/2)+112,int(size/2)-112:int(size/2)+112]
            # imageio.imsave('test_{}.nii'.format(idx), img)
            # print(img_path,int(c/2), np.max(img), np.min(img))
            img = (img-np.min(img))/(np.max(img)-np.min(img))
            # except:
                # print(img_path)

            # sitk_img = sitk.ReadImage(img_path)
            # np_img = sitk.GetArrayFromImage(sitk_img)
            # img = createMIP(np_img, slices_num = 256)
            # sitk_mip = sitk.GetImageFromArray(img)
            # sitk_mip.SetOrigin(sitk_img.GetOrigin())
            # sitk_mip.SetSpacing(sitk_img.GetSpacing())
            # sitk_mip.SetDirection(sitk_img.GetDirection())
            # writer = sitk.ImageFileWriter()
            # writer.SetFileName('test.nii')
            # writer.Execute(sitk_mip)
            # imageio.imsave('test.png', img[:,:,150])
            # exit()
            # imageio.imsave('test_{}.png'.format(idx), img)

            # img = cv2.resize(img, (224, 224))
            img = np.dstack([img, img, img])
            # imageio.imsave('ptest.IMA', img)
            # temp = nib.Nifti1Image(img, None)
            # nib.save(temp, 'test_{}.nii'.format(idx))
            # exit()
            imageio.imsave('imgs/{}_{}_{}_{}_{}.png'.format(img_path.split('/')[-5], img_path.split('/')[-4], img_path.split('/')[-3], img_path.split('/')[-2], i), img)

            img = torch.from_numpy(img.copy())
            img = rearrange(img, 'h w c -> c h w')

            imgs.append(img)


        level = img_path.split('/')[-2]
        img_name = '{}_{}_{}_{}'.format(img_path.split('/')[-5], img_path.split('/')[-4], img_path.split('/')[-3], img_path.split('/')[-2])
        # img_name = img_path.split('/')[-1]
        # print(img_name)
        # exit()

        return imgs, img_name

class ABIDEIITIQAData(Dataset):
    def __init__(self, file_list, csv_dir, transform='train', norm=False):
        self.file_list = file_list
        self.transform = transform
        self.labels = pd.read_csv(csv_dir, encoding="UTF-7")
        self.labels.index = self.labels['title']
        self.norm = norm

    def __len__(self):
        self.filelength = len(self.file_list)
        # self.filelength = len(self.labels)
        return self.filelength

    def __getitem__(self, idx):
        # read image
        img_path = self.file_list[idx]
        imgname = 'ABIDEII-GU_1_{}_session_1_anat_1_{}'.format(img_path.split('/')[-2], img_path.split('/')[-1].split('.')[0])
        img = imageio.imread(img_path)

        # get label
        mean = self.labels.loc[imgname].values[1]

        # transform
        transformer = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [1, 1, 1])
        ])

        # transform image
        if self.transform == 'train':
            # read image
            # img_path = '../../data/phantom_train_test_new/train/phantom/ge/chest/{}/{}_{}_{}_{}.tiff'.format(lv, imgname.split('_')[0], imgname.split('_')[1], lv, imgname.split('_')[2])
            # img = imageio.imread(img_path)
            # img = cv2.resize(img, (224, 224))
            if random.random() >= 0.5:
                img = np.flip(img, 1)
            if random.random() >= 0.5:
                img = np.flip(img, 0)
            img = torch.from_numpy(img.copy())
            # img = np.dstack([img, img, img])
            img = rearrange(img, 'h w c -> c h w')
        else:
            # read image
            # if self.transform == 'val':
            #     img_path = '../../data/phantom_train_test_new/train/phantom/ge/chest/{}/{}_{}_{}_{}.tiff'.format(lv, imgname.split('_')[0], imgname.split('_')[1], lv, imgname.split('_')[2])
            # else:
            #     img_path = '../../data/phantom_train_test_new/test/phantom/ge/chest/{}/{}_{}_{}_{}.tiff'.format(lv, imgname.split('_')[0], imgname.split('_')[1], lv, imgname.split('_')[2])
            # img = imageio.imread(img_path)
            # img = cv2.resize(img, (224, 224))
            img = torch.from_numpy(img.copy())
            # img = np.dstack([img, img, img])
            img = rearrange(img, 'h w c -> c h w')

        if self.norm == True:
            # img = transformer(img)
            pass

        return img, mean, imgname


class PhantomData(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
    
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
        
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = imageio.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = np.dstack([img, img, img])
        img = torch.from_numpy(img.copy())
        img = rearrange(img, 'h w c -> c h w')

        level = img_path.split('/')[-2]
        # img_name = '{}_{}_{}_{}'.format(img_path.split('/')[-4], img_path.split('/')[-3], img_path.split('/')[-2], img_path.split('/')[-1])
        img_name = img_path.split('/')[-1]

        return img, level, img_name


class PhantomTIQAData(Dataset):
    def __init__(self, file_list, csv_dir, transform='train', norm=False):
        self.file_list = file_list
        self.transform = transform
        self.labels = pd.read_csv(csv_dir)
        self.labels.index = self.labels['title']
        self.norm = norm

    def __len__(self):
        # self.filelength = len(self.file_list)
        self.filelength = len(self.labels)
        return self.filelength

    def __getitem__(self, idx):
        # read image
        # img_path = self.file_list[idx]
        imgname = self.labels.index[idx]
        # img = imageio.imread(img_path)

        # get label
        lv = '{}_{}'.format(imgname.split('_')[-2], imgname.split('_')[-1])
        # imgname = '{}_{}_{}_{}'.format(fid.split('_')[0], fid.split('_')[1], fid.split('_')[-1], lv)
        mean = self.labels.loc[imgname].values[1]

        # transform
        transformer = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [1, 1, 1])
        ])

        # transform image
        if self.transform == 'train':
            # read image
            img_path = '../../data/phantom_train_test_new/train/phantom/ge/chest/{}/{}_{}_{}_{}.tiff'.format(lv, imgname.split('_')[0], imgname.split('_')[1], lv, imgname.split('_')[2])
            img = imageio.imread(img_path)
            # img = cv2.resize(img, (224, 224))
            if random.random() >= 0.5:
                img = np.flip(img, 1)
            if random.random() >= 0.5:
                img = np.flip(img, 0)
            img = torch.from_numpy(img.copy())
            img = np.dstack([img, img, img])
            img = rearrange(img, 'h w c -> c h w')
        else:
            # read image
            if self.transform == 'val':
                img_path = '../../data/phantom_train_test_new/train/phantom/ge/chest/{}/{}_{}_{}_{}.tiff'.format(lv, imgname.split('_')[0], imgname.split('_')[1], lv, imgname.split('_')[2])
            else:
                img_path = '../../data/phantom_train_test_new/test/phantom/ge/chest/{}/{}_{}_{}_{}.tiff'.format(lv, imgname.split('_')[0], imgname.split('_')[1], lv, imgname.split('_')[2])
            img = imageio.imread(img_path)
            # img = cv2.resize(img, (224, 224))
            img = torch.from_numpy(img.copy())
            img = np.dstack([img, img, img])
            img = rearrange(img, 'h w c -> c h w')

        if self.norm == True:
            # img = transformer(img)
            pass

        return img, mean, imgname



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
            # img = cv2.resize(img, (224, 224))
            # center crop
            # img = img[144: 144+224, 144: 144+224]
            # random crop
            # x = random.randint(100, 189)
            # y = random.randint(100, 189)
            # for cahdc
            # x = random.randint(0, 212)
            # y = random.randint(0, 212)
            # img = img[x: x+300, y: y+300]
            # for maniqa
            # x = random.randint(0, 512-224)
            # y = random.randint(0, 512-224)
            # img = img[x: x+224, y: y+224]
            if random.random() >= 0.5:
                img = np.flip(img, 1)
            if random.random() >= 0.5:
                img = np.flip(img, 0)
            img = torch.from_numpy(img.copy())
            img = rearrange(img, 'h w c -> c h w')
        else:
            # img = cv2.resize(img, (224, 224))
            # img = img[int(512/2)-150: int(512/2)+150, int(512/2)-150: int(512/2)+150] # cahdc
            # img = img[int(512/2)-112: int(512/2)+112, int(512/2)-112: int(512/2)+112] # maniqa
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
            # img = transformer(img)
            pass

        return img, mean, imgname


class MayoRandomPatchDataset(Dataset): # wadiqam
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
            if random.random() >= 0.5:
                img = np.flip(img, 0)
            # random crop image
            # x = random.randint(0, 480)
            # y = random.randint(0, 480)
            # img = img[x: x+32, y: y+32]
            # img = torch.from_numpy(img.copy())
            # img = rearrange(img, 'h w c -> c h w')
        # else:
        patches = []
        # random crop image
        for i in range(32):
            patch = img.copy()
            x = random.randint(0, 480)
            y = random.randint(0, 480)
            patch = patch[x: x+32, y: y+32]
            patch = torch.from_numpy(patch.copy())
            patch = rearrange(patch, 'h w c -> c h w')
            patches.append(patch)
        img = tuple(patches)

        # if self.norm == True:
        #     img = transformer(img)

        return img, mean, imgname

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

        return img, mean, imgname


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
