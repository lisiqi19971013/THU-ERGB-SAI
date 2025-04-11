import sys
import numpy as np
sys.path.append('..')
from utils import dataset
from model_final.model_full import EventFrameDeOcc
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
import cv2


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def saveImg(img, path):
    img[img > 1] = 1
    img[img < 0] = 0
    img = np.array(img[0].cpu().permute(1, 2, 0) * 255)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    import torch
    from torch import nn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='/lisiqi/ERGBSAI')
    parser.add_argument('--ckpt', type=str, default='./')
    parser.add_argument('--opFolder', type=str, default='./output')
    args = parser.parse_args()

    device = 'cuda'

    testFolder = dataset(args.folder, train=False)
    testLoader = DataLoaderX(testFolder, batch_size=1, shuffle=False, pin_memory=False, num_workers=0)

    model = EventFrameDeOcc(inChannels_event=36, inChannels_frame=51, norm='BN')
    model = torch.nn.DataParallel(model)
    model.cuda()

    print('==> loading existing model:', os.path.join(args.ckpt, 'ckpt.pth'))
    model_info = torch.load(os.path.join(args.ckpt, 'ckpt.pth'))
    model.load_state_dict(model_info['state_dict'])

    os.makedirs(args.opFolder, exist_ok=True)

    with torch.no_grad():
        model.eval()
        for i, (event_vox, img, mask, gt_img) in enumerate(testLoader):

            event_vox = event_vox.cuda()
            img = img.cuda().float()
            mask = mask.cuda().float()
            gt_img = gt_img.cuda().float()

            output = model(event_vox, img, mask)

            saveImg(output, os.path.join(args.opFolder, '%d_output.png'%i))
            saveImg(gt_img, os.path.join(args.opFolder, '%d_gt.png'%i))

            print(i)
