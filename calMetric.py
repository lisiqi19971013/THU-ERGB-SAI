import glob
import numpy as np
import os
import cv2
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import argparse


def calpsnr(gt, pred):
    return peak_signal_noise_ratio(gt, pred)


def calssim(gt, pred):
    return structural_similarity(gt, pred, channel_axis=2, gaussian_weights=True)


parser = argparse.ArgumentParser()
parser.add_argument('--dataFolder', type=str, default='/lisiqi/ERGBSAI')
parser.add_argument('--opFolder', type=str, default='./output')
args = parser.parse_args()


with open(os.path.join(args.dataFolder, 'test.txt'), 'r') as f:
    list = f.readlines()

path = args.opFolder

psnr = []
ssim = []

psnr_dense, ssim_dense = [], []
psnr_sparse, ssim_sparse = [], []


imgList = glob.glob(os.path.join(path, '*_output.png'))
for i, f in enumerate(imgList):
    op = cv2.imread(f)
    gt = cv2.imread(f.replace('output.png', 'gt.png'))
    p = calpsnr(gt, op)
    s = calssim(gt, op)

    file = list[i]

    psnr.append(p)
    ssim.append(s)
    if 'outdoor/' in file or 'pf_' in file or 'fence' in file:
        psnr_sparse.append(p)
        ssim_sparse.append(s)
    else:
        psnr_dense.append(p)
        ssim_dense.append(s)

    print(i, p, s)


line = "sparse:%f,%f\tdense:%f,%f\tTotal:%f,%f" %\
       (sum(psnr_sparse)/len(psnr_sparse), sum(ssim_sparse)/len(ssim_sparse),
        sum(psnr_dense)/len(psnr_dense), sum(ssim_dense)/len(ssim_dense),
        sum(psnr)/len(psnr), sum(ssim)/len(ssim))
print(line)

with open(os.path.join(path, 'res.txt'), 'w') as f:
    f.writelines(line)