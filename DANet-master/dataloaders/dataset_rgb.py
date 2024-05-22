import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils_isp.image_utils import is_png_file, load_img
from utils_isp.GaussianBlur import get_gaussian_kernel
import torchvision.transforms.functional as TF
from random import random, choice
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

def loadpathslist(root,flag):
    classes = os.listdir(root)
    paths = []
    if not '1_fake' in classes:
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
rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, rz_interp=['bilinear']):
    interp = sample_discrete(rz_interp)
    return TF.resize(img, (256, 256), interpolation=rz_dict[interp])

def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)
class read_data_new():
    def __init__(self, dataroot, sample=1,isTrain=True, isVal=False):
        self.root = dataroot
        self.isTrain=isTrain
        self.isVal=isVal
        real_img_list = loadpathslist(self.root,'0_real')[::sample]
        real_label_list = [0. for _ in range(len(real_img_list))]
        self.img = real_img_list
        self.label = real_label_list
        if not self.isVal:
            fake_img_list = loadpathslist(self.root,'1_fake')[::sample]
            fake_label_list = [1. for _ in range(len(fake_img_list))]
            self.img = real_img_list+fake_img_list
            self.label = real_label_list+fake_label_list


    def __getitem__(self, index):
        img = torch.from_numpy(np.float32(load_img(self.img[index])))
        img = img.permute(2, 0, 1)
        filename = self.img[index]
        return img, filename

    def __len__(self):
        return len(self.label)
##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, 'clean')))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'noisy')))


        self.clean_filenames = [os.path.join(rgb_dir, 'clean', x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, 'noisy', x) for x in noisy_files if is_png_file(x)]
        

        self.tar_size = len(self.clean_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
                
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        return clean, noisy, clean_filename, noisy_filename

##################################################################################################

class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTest, self).__init__()

        self.target_transform = target_transform

        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'noisy')))


        self.noisy_filenames = [os.path.join(rgb_dir, 'noisy', x) for x in noisy_files if is_png_file(x)]
        

        self.tar_size = len(self.noisy_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
                
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        noisy = noisy.permute(2,0,1)

        return noisy, noisy_filename


##################################################################################################

MAX_SIZE = 512    

def divisible_by(img, factor=16):
    h, w, _ = img.shape
    img = img[:int(np.floor(h/factor)*factor),:int(np.floor(w/factor)*factor),:]
    return img

class DataLoader_NoisyData(Dataset):
    def __init__(self, rgb_dir):
        super(DataLoader_NoisyData, self).__init__()

        rgb_files=sorted(os.listdir(rgb_dir))
        
        #print("number of images:", len(rgb_files))
        self.target_filenames = [os.path.join(rgb_dir, x) for x in rgb_files if is_png_file(x)]
        
        self.tar_size = len(self.target_filenames)  # get the size of target
        self.blur, self.pad = get_gaussian_kernel(kernel_size=5, sigma=1)   ### preprocessing to remove noise from the input rgb image

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size

        target = np.float32(load_img(self.target_filenames[tar_index]))
        
        target = divisible_by(target, 16)

        tar_filename = os.path.split(self.target_filenames[tar_index])[-1]


        target = torch.Tensor(target)
        target = target.permute(2,0,1)
        
        target = F.pad(target.unsqueeze(0), (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        target = self.blur(target).squeeze(0)

        padh = (MAX_SIZE - target.shape[1])//2
        padw = (MAX_SIZE - target.shape[2])//2
        target = F.pad(target.unsqueeze(0), (padw, padw, padh, padh), mode='constant').squeeze(0)

        return target, tar_filename, padh, padw

def loadpathslist(root,flag):
    classes = os.listdir(root)
    paths = []
    if not '1_fake' in classes:
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

class DataLoader_RPTC(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoader_RPTC, self).__init__()
        self.rgb_dir = rgb_dir
        self.target_transform = target_transform
        self.real_img_list = loadpathslist(self.rgb_dir, '1_fake')

        self.tar_size = len(self.real_img_list)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        image = torch.from_numpy(np.float32(load_img(self.real_img_list[tar_index])))

        image = image.permute(2, 0, 1)

        return image