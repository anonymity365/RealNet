# import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice, shuffle
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
import os
import torchvision
from glob import glob
from operator import itemgetter
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True

class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")

def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)

rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, rz_interp=['bilinear']):
    interp = sample_discrete(rz_interp)
    return TF.resize(img, (256, 256), interpolation=rz_dict[interp])

def loadpathslist(root,flag):
    classes = os.listdir(root)
    paths = []
    if not '0_real' in classes:
        for class_name in classes:
            imgpaths = os.listdir(root+'/'+class_name +'/'+flag+'/')
            for imgpath in imgpaths:
                paths.append(root+'/'+class_name +'/'+flag+'/'+imgpath)
        return paths
    else:
        imgpaths = os.listdir(root+'/'+flag+'/')
        for imgpath in imgpaths:
            paths.append(root+'/'+flag+'/'+imgpath)
        return paths

def processing(img, isTrain, isVal):
    if isTrain:
        crop_func = transforms.RandomCrop((256, 256))
        flip_func = transforms.RandomHorizontalFlip()
        rz_func = transforms.Lambda(lambda img: custom_resize(img))
    else:
        crop_func = transforms.CenterCrop((256,256))
        flip_func = transforms.Lambda(lambda img: img)
        rz_func = transforms.Lambda(lambda img: custom_resize(img))

    trans = transforms.Compose([
                rz_func,
                # transforms.Lambda(lambda img: data_augment(img) if isTrain else img),
                crop_func,
                flip_func,
                transforms.ToTensor()
                ])
    return trans(img)

class read_data():
    def __init__(self, dataroot, isTrain=True, isVal=False, dir='0_real'):
        self.root = dataroot
        self.isTrain=isTrain
        self.isVal=isVal
        if self.isTrain:
            real_img_list = loadpathslist(self.root, dir)
            real_label_list = [0 for _ in range(len(real_img_list))]
            self.img = real_img_list
            self.label = real_label_list
        else:
            real_img_list = loadpathslist(self.root, '0_real')
            real_label_list = [0 for _ in range(len(real_img_list))]
            fake_img_list = loadpathslist(self.root, '1_fake')
            fake_label_list = [1 for _ in range(len(fake_img_list))]
            self.img = real_img_list+fake_img_list
            self.label = real_label_list+fake_label_list

    def __getitem__(self, index):
        img, target = Image.open(self.img[index]).convert('RGB'), self.label[index]
        imgname = self.img[index]
        img = processing(img, self.isTrain, self.isVal)

        return img, target

    def __len__(self):
        return len(self.label)

def get_dataloader(opt):
    train_data_path = opt.train_data_path
    val_data_path = opt.val_data_path
    dataset_train = read_data(train_data_path, isTrain=True, isVal=False)
    dataset_val = read_data(val_data_path, isTrain=False, isVal=True)
    data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                              batch_size=opt.batchsize,
                                              shuffle=True,
                                              num_workers=int(0))
    data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=int(0))
    print(len(dataset_train), len(dataset_val))
    return data_loader_train, data_loader_val