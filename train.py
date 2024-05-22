import random
import numpy as np
import logging
import os
import torch
import argparse
from data.datasets import get_dataloader
from network.trainer import Trainer

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default='./checkpoint')
    parser.add_argument("--train_name", type=str, default='RealNet')
    parser.add_argument("--train_data_path", type=str, default='/home/NSRR/train')
    parser.add_argument("--val_data_path", type=str, default='/home/NSRR/val')
    parser.add_argument("--w_adaptor", default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in_channels", type=int, default=64)
    parser.add_argument("--out_channels", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--std", type=float, default=0.008)
    parser.add_argument("--batchsize", type=int, default=8)
    args = parser.parse_args()
    return args

def main(opt):
    data_loader_train, data_loader_val = get_dataloader(opt)
    trainer = Trainer(opt)
    best_ap = trainer.train(data_loader_train, data_loader_val)



if __name__ == "__main__":
    opt = get_options()
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    main(opt)