import torch.fft as fft
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm


def fft2d(input):
    fft_out = fft.fftn(input, dim=(2, 3), norm='ortho')
    return fft_out

def fftshift2d(input):
    b, c, h, w = input.shape
    fs11 = input[:, :, -h // 2:h, -w // 2:w]
    fs12 = input[:, :, -h // 2:h, 0:w // 2]
    fs21 = input[:, :, 0:h // 2, -w // 2:w]
    fs22 = input[:, :, 0:h // 2, 0:w // 2]
    output = torch.cat([torch.cat([fs11, fs21], axis=2), torch.cat([fs12, fs22], axis=2)], axis=3)
    return output

def feature_transformer(x, device):
    x_fft = fft2d(x)
    x_amplitude = torch.abs(x_fft)
    x_amplitude = torch.pow(x_amplitude + 1e-8, 0.8)
    amplitude = fftshift2d(x_amplitude)
    amplitude_1 = 0.11 * amplitude[:, 0, :, :] + 0.59 * amplitude[:, 1, :, :] + 0.3 * amplitude[:, 2, :, :]
    amplitude = torch.unsqueeze(amplitude_1, dim=1).to(device)

    assert len(amplitude.shape) == 4 and amplitude.shape[2] == amplitude.shape[3] == 256   #feaure enhance
    A_1 = torch.mean(amplitude, dim=3, keepdim=True).to(device)
    A = amplitude - A_1
    A = torch.where(A > 0, amplitude**2, torch.zeros(A.shape).to(device))

    return torch.flatten(A[:, :, ::32, ::32], 1, -1)


def get_TRR(NSRR, device):      # compress the dimensions of NSRR
    IN = nn.InstanceNorm1d(1, affine=False)
    NSRR = NSRR.to(device)

    features = feature_transformer(NSRR, device)
    features = features.view(features.shape[0], 1, features.shape[-1])
    TRR = IN(features)
    return TRR.view(TRR.shape[0], TRR.shape[-1])



