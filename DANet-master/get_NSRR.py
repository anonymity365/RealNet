from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from networks import UNetD
from utils import PadUNet
from skimage import img_as_ubyte
import h5py
import scipy.io as sio
from pdb import set_trace as stx
from dataloaders.data_rgb import get_validation_data
import cv2

parser = argparse.ArgumentParser(description='Image Denoising using DANet')


parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', default=True, action='store_true', help='Save denoised images in result directory')
parser.add_argument('--model', type=str, default='DANet+',
                                          help="Model selection: DANet or DANet+, (default:DANet+)")
parser.add_argument('--dataroot', type=str, default=r'E:\data\AIGCDetect_test\test',
                                          help="root")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

net = UNetD(3, wf=32, depth=5).cuda()
# load the pretrained model
if args.model.lower() == 'danet':
    net.load_state_dict(torch.load('./model_states/DANet.pt', map_location='cpu')['D'])
else:
    net.load_state_dict(torch.load('./model_states/DANetPlus.pt', map_location='cpu'))


########################
vals = ['progan','biggan', 'cyclegan', 'stargan', 'gaugan',
        'stylegan2',
        'ADM','Glide','Midjourney','stable_diffusion_v_1_4','stable_diffusion_v_1_5','VQDM','wukong','DALLE2',
        'EGC', 'MaskGIT', 'VAR', 'VQVAE2', 'LFM'
        ]


for v_id, val in enumerate(vals):
    dataroot = '{}/{}'.format(args.dataroot, val)

    test_dataset = get_validation_data(dataroot)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    print(len(test_dataset))
    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            rgb_noisy = data_test[0].cuda()
            filenames = data_test[1]
            padunet = PadUNet(rgb_noisy, dep_U=5)
            inputs_pad = padunet.pad()
            outputs_pad = inputs_pad - net(inputs_pad)
            outputs = padunet.pad_inverse(outputs_pad)

            NSRR = rgb_noisy - outputs    #Noise-Sensitive Real Representation(NSRR)
            NSRR.clamp_(0.0, 1.0)

            rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
            NSRR = NSRR.permute(0, 2, 3, 1).cpu().detach().numpy()

            if args.save_images:
                for batch in range(len(rgb_noisy)):
                    denoised_img = img_as_ubyte(NSRR[batch])
                    new_path = filenames[batch].replace('AIGCDetect_test', 'NSRR')
                    if not os.path.exists(os.path.dirname(new_path)):
                        os.makedirs(os.path.dirname(new_path))
                    cv2.imwrite(new_path, denoised_img * 255)

            # cv2.cvtColor(denoised_img, cv2.COLOR_RGB2BGR) * 255