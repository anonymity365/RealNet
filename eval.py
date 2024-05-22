import os
import csv
import random
from util.validate import eval
from network.discriminator import Discriminator
from data.datasets import read_data
import torch
import argparse

parser = argparse.ArgumentParser(description='RealNet eval')
parser.add_argument('--data_root', type=str, default='/home/NSRR/test')
parser.add_argument('--ckpt_path', type=str, default='weights/Discriminator/all_model_best.pth')
parser.add_argument('--outdir', type=str, default='./results')
parser.add_argument('--test_name', type=str, default='RealNet')
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



rows = [["{} model testing on...".format(args.test_name)],
        ['testset', 'ap', 'uncalibrated_acc', 'oracle_acc', 'uncalibrated_r_acc', 'uncalibrated_f_acc', 'oracle_r_acc',
         'oracle_f_acc', 'f1_scores', 'auc', 'best_threshold', 'stable_threshold']]

vals = ['progan','biggan', 'cyclegan', 'stargan', 'gaugan',
        'stylegan2',
        'ADM','Glide','Midjourney','stable_diffusion_v_1_4','stable_diffusion_v_1_5','VQDM','wukong','DALLE2',
        'EGC', 'MaskGIT', 'VAR', 'VQVAE2', 'LFM'
        ]


uncalibrated_threshold = 0
for v_id, val in enumerate(vals):
    model = Discriminator(64, 64)
    model.load(args.ckpt_path)
    model.to(device)
    model.eval()
    dataroot = '{}/{}'.format(args.data_root, val)
    testdataset = read_data(dataroot, isTrain=False, isVal=False)
    test_dataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=True, num_workers=int(0))
    if v_id == 0:
        metrics = eval(test_dataloader, model, device, v_id, uncalibrated_threshold)
        uncalibrated_threshold = metrics[-1]
    else:
        metrics = eval(test_dataloader, model, device, v_id, uncalibrated_threshold)

    rows.append(metrics)
    print("({}) ap {:.4f}, oracle_acc {:.4f}, f1_scores {:.4f}, auc {:.4f}".format(val, metrics[0], metrics[2], metrics[7], metrics[8]))

# save results
results_dir = args.results_dir
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
csv_name = results_dir + '/{}.csv'.format(args.test_name)
with open(csv_name, 'a+') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)

