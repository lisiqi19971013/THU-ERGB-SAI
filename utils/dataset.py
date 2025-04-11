import glob
from PIL import Image
from torch.utils import data
import os
import torch
import numpy as np
from torchvision import transforms


class dataset(data.Dataset):
    def __init__(self, folder, train=True, nb_of_bin=36):
        self.file = os.path.join(folder, 'train.txt') if train else os.path.join(folder, 'test.txt')
        self.folder = []
        self.nb_of_bin = nb_of_bin
        with open(self.file, 'r') as f:
            for line in f.readlines():
                p1 = line.strip('\n')
                self.folder.append(os.path.join(folder, p1))
        self.train = train

    def __getitem__(self, idx):
        files = os.listdir(self.folder[idx])

        event = np.load(os.path.join(self.folder[idx], 'events.npy'))
        mid = int(glob.glob(os.path.join(self.folder[idx], 'frame_*_1.jpg'))[0].split('_')[-2])
        ts = np.load(os.path.join(self.folder[idx], 'ts.npy'))
        frames = glob.glob(os.path.join(self.folder[idx], 'frame_*.jpg'))
        frames.sort()

        id_start = mid-8
        id_end = mid+9
        imgs = torch.cat([transforms.ToTensor()(Image.open(frames[i])) for i in range(id_start,id_end)], dim=0)

        mask = []
        if 'image.jpg' in files:
            gt = transforms.ToTensor()(Image.open(os.path.join(self.folder[idx], 'image.jpg')))
            t_start = (ts[id_start] + ts[id_start - 1]) / 2
            t_end = (ts[id_end - 1] + ts[id_end]) / 2
            event = event[:, [0, 2, 3, 1]]
            event1 = event[(event[:, 0] >= t_start) & (event[:, 0] <= t_end), :].astype(np.int64)
            dim = (260, 346)
            event_vox = self.event2vox(event1.copy(), dim)
            for k in range(id_start - 1, id_end + 1):
                e1 = event[(event[:, 0] > (ts[k] + ts[k - 1]) / 2) & (event[:, 0] < (ts[k] + ts[k + 1]) / 2), :]
                ecm = self.event2ecm(e1, dim)
                m = (ecm != 0)
                mask.append(m)
            mask = 1 - np.concatenate(mask[1:-1], axis=0).astype(np.float32)
        else:
            gt = transforms.ToTensor()(Image.open(os.path.join(self.folder[idx], 'gt_kinect_cvt.png')))
            event1 = event[(event[:, 0] >= ts[id_start][0]) & (event[:, 0] <= ts[id_end-1][1]), :].astype(np.int64)
            dim = (260, 260)
            event_vox = self.event2vox(event1.copy(), dim)
            mask = []
            for k in range(id_start - 1, id_end + 1):
                e1 = event[(event[:, 0] > ts[k][0]) & (event[:, 0] < ts[k][1]), :]
                ecm = self.event2ecm(e1, dim)
                m = (ecm != 0)
                mask.append(m)
            mask = 1 - np.concatenate(mask[1:-1], axis=0).astype(np.float32)
        mask = torch.from_numpy(mask)
        img, mask, gt, event_vox = self.data_augmentation(imgs, mask, gt, event_vox)
        return event_vox, img, mask, gt

    def __len__(self):
        return len(self.folder)

    def data_augmentation(self, input_image, mask, gt_image, event_grid, crop_size=(256, 256)):
        if self.train:
            transforms_list = transforms.Compose([transforms.RandomCrop(crop_size), transforms.RandomHorizontalFlip()])
        else:
            transforms_list = transforms.Compose([transforms.CenterCrop(crop_size)])

        input_channel = input_image.shape[0]
        mask_channel = mask.shape[0]
        gt_channel = gt_image.shape[0]
        x = torch.cat([input_image, mask, gt_image, event_grid], dim=0)
        x1 = transforms_list(x)
        input_image1 = x1[:input_channel, ...]
        mask1 = x1[input_channel:input_channel + mask_channel, ...]
        gt_image1 = x1[input_channel + mask_channel:input_channel + mask_channel + gt_channel, ...]
        event_grid1 = x1[input_channel + mask_channel + gt_channel:, ...]
        return input_image1, mask1, gt_image1, event_grid1

    def event2vox(self, event, dim):
        event = torch.from_numpy(event).float()
        H, W = dim
        # x -> W, y-> H
        voxel_grid = torch.zeros(self.nb_of_bin, H, W, dtype=torch.float32, device='cpu')
        vox = voxel_grid.ravel()

        t, x, y, p = event.t()
        t = t.long()
        time_max = t.max()
        time_min = t.min()

        t = (t-time_min) * (self.nb_of_bin - 1) / (time_max-time_min)
        t = t.float()
        left_t, right_t = t.floor(), t.floor()+1
        left_x, right_x = x.float().floor(), x.float().floor()+1
        left_y, right_y = y.float().floor(), y.float().floor()+1

        for lim_x in [left_x, right_x]:
            for lim_y in [left_y, right_y]:
                for lim_t in [left_t, right_t]:
                    mask = (0 <= lim_x) & (0 <= lim_y) & (0 <= lim_t) & (lim_x <= W-1) & (lim_y <= H-1) & (lim_t <= self.nb_of_bin-1)
                    lin_idx = lim_x.long() + lim_y.long() * W + lim_t.long() * W * H
                    weight = p * (1-(lim_x-x).abs()) * (1-(lim_y-y).abs()) * (1-(lim_t-t).abs())
                    vox.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())

        return voxel_grid

    def event2ecm(self, event, dim):
        event = torch.from_numpy(event).float()
        H, W = dim

        voxel_grid = torch.zeros(1, H, W, dtype=torch.float32, device='cpu')
        vox = voxel_grid.ravel()

        t, x, y, p = event.t()
        left_x, right_x = x.float().floor(), x.float().floor()+1
        left_y, right_y = y.float().floor(), y.float().floor()+1

        for lim_x in [left_x, right_x]:
            for lim_y in [left_y, right_y]:
                    mask = (0 <= lim_x) & (0 <= lim_y) & (lim_x <= W-1) & (lim_y <= H-1)
                    lin_idx = lim_x.long() + lim_y.long() * W
                    weight = (1 - (lim_x - x).abs()) * (1 - (lim_y - y).abs())
                    vox.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())

        return voxel_grid




