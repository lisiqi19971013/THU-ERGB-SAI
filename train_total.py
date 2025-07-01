import sys
sys.path.append('..')
from utils import dataset
from model_final.model_full import EventFrameDeOcc
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
import numpy as np
import datetime


class Metric(object):
    def __init__(self):
        self.L1Loss_this_epoch = []
        self.Lpips_this_epoch = []
        self.FeaLoss_this_epoch = []
        self.total_this_epoch = []

        self.L1Loss_history = []
        self.Lpips_history = []
        self.FeaLoss_history = []
        self.total_history = []

    def update(self, L1Loss, Lpips, FeaLoss, total):
        self.L1Loss_this_epoch.append(L1Loss.item())
        self.Lpips_this_epoch.append(Lpips.item())
        self.FeaLoss_this_epoch.append(FeaLoss.item())
        self.total_this_epoch.append(total.item())

    def update_epoch(self):
        avg = self.get_average_epoch()
        self.L1Loss_history.append(avg[0])
        self.Lpips_history.append(avg[1])
        self.FeaLoss_history.append(avg[2])
        self.total_history.append(avg[3])
        self.new_epoch()

    def new_epoch(self):
        self.L1Loss_this_epoch = []
        self.Lpips_this_epoch = []
        self.FeaLoss_this_epoch = []
        self.total_this_epoch = []

    def get_average_epoch(self):
        return np.average(self.L1Loss_this_epoch), np.average(self.Lpips_this_epoch), np.average(self.FeaLoss_this_epoch), np.average(self.total_this_epoch)


class TimeRecorder(object):
    def __init__(self, total_epoch, iter_per_epoch):
        self.total_epoch = total_epoch
        self.iter_per_epoch = iter_per_epoch
        self.start_train_time = datetime.datetime.now()
        self.start_epoch_time = datetime.datetime.now()
        self.t_last = datetime.datetime.now()

    def get_iter_time(self, epoch, iter):
        dt = (datetime.datetime.now() - self.t_last).__str__()
        self.t_last = datetime.datetime.now()
        remain_time = self.cal_remain_time(epoch, iter, self.total_epoch, self.iter_per_epoch)
        end_time = (datetime.datetime.now() + datetime.timedelta(seconds=remain_time)).strftime("%Y-%m-%d %H:%S:%M")
        remain_time = datetime.timedelta(seconds=remain_time).__str__()
        return dt, remain_time, end_time

    def cal_remain_time(self, epoch, iter, total_epoch, iter_per_epoch):
        t_used = (datetime.datetime.now() - self.start_train_time).total_seconds()
        time_per_iter = t_used / (epoch * iter_per_epoch + iter + 1)
        remain_iter = total_epoch * iter_per_epoch - (epoch * iter_per_epoch + iter + 1)
        remain_time_second = time_per_iter * remain_iter
        return remain_time_second


def calpsnr(gt, pred):
    return peak_signal_noise_ratio(gt, pred)


def calssim(gt, pred):
    return structural_similarity(gt, pred, multichannel=True, gaussian_weights=True)


def showMessage(message, file):
    print(message)
    with open(file, 'a') as f:
        f.writelines(message + '\n')


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    import shutil
    import torch
    from torch import nn
    from torch.optim import lr_scheduler
    import torch.optim as optim
    from tensorboardX import SummaryWriter
    from utils import EarlyStopping
    from lpips import lpips

    folder = './ERGBSAI'   # Change this for dataset folder
    frame_ckpt = './log/2025-06-30/frame_net/checkpoint_max_psnr.pth'  # Change this for Frame ckpt
    event_ckpt = './log/2025-07-01/ef_net/checkpoint_max_psnr.pth'     # Change this for Event-Frame ckpt

    device = 'cuda'
    random_seed = 1996
    batch_size = 16
    lr = 2e-3
    alpha = 0.1
    epochs = 1000

    run_dir = './log/' + datetime.date.today().__str__() + '/total'
    print('rundir:', os.path.abspath(run_dir))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True

    trainFolder = dataset(folder, train=True)
    trainLoader = DataLoaderX(trainFolder, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0, drop_last=True)
    testFolder = dataset(folder, train=False)
    testLoader = DataLoaderX(testFolder, batch_size=1, shuffle=False, pin_memory=False, num_workers=0)

    train_total_iter = len(trainLoader)
    test_total_iter = len(testLoader)
    print(train_total_iter, test_total_iter)

    model = EventFrameDeOcc(inChannels_event=36, inChannels_frame=51, norm='BN')

    a = torch.load(frame_ckpt)
    model.FrameEncoder.MaskPredNet.load_state_dict(a['MaskPredNet'])
    model.FrameEncoder.Encoder.load_state_dict(a['Encoder'])
    model.FrameDecoder.load_state_dict(a['FrameDecoder'])

    a = torch.load(event_ckpt)
    model.EventReFocus.load_state_dict(a['EventReFocus'])
    model.EventEncoder.load_state_dict(a['EventEncoder'])
    model.EventFrameDecoder.load_state_dict(a['EventFrameDecoder'])

    loss_l1 = nn.L1Loss()
    loss_lpips = lpips.LPIPS(net='vgg', spatial=False).cuda()

    tb = SummaryWriter(run_dir)
    early_stopping = EarlyStopping(patience=1000, verbose=True)

    if os.path.exists(os.path.join(run_dir, 'checkpoint.pth.tar')):
        print('==> loading existing model:', os.path.join(run_dir, 'checkpoint.pth.tar'))
        model_info = torch.load(os.path.join(run_dir, 'checkpoint.pth.tar'))
        model.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(run_dir):
            os.makedirs(run_dir)
        cur_epoch = 0

    with open(os.path.join(run_dir, 'cfg.txt'), 'w') as f:
        f.writelines('lr %f, epoch %d, alpha %f\n' % (lr, epochs, alpha))
        f.writelines(model.__repr__())

    shutil.copy(os.path.abspath(__file__), os.path.join(run_dir, os.path.basename(__file__)))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    for param_group in optimizer.param_groups:
        print(param_group['lr'])

    trainMetirc = Metric()
    testMetirc = Metric()

    psnr_list = []
    ssim_list = []
    TR = TimeRecorder(epochs-cur_epoch, train_total_iter+test_total_iter)

    for epoch in range(cur_epoch, epochs):
        model.train()

        for i, (event_vox, img, mask, gt_img) in enumerate(trainLoader):
            event_vox = event_vox.cuda()
            img = img.cuda().float()
            mask = mask.cuda().float()
            gt_img = gt_img.cuda().float()

            output = model(event_vox, img, mask)

            Lpips = torch.sum(loss_lpips.forward(output, gt_img, normalize=True)) / batch_size
            L1Loss = loss_l1(output, gt_img)
            Loss = L1Loss + alpha * Lpips

            if Loss.item() >= 50:
                print('warning')
                with open(os.path.join(run_dir, 'log.txt'), 'a') as f:
                    f.writelines('warning\n')
                torch.cuda.empty_cache()
                continue

            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
            trainMetirc.update(L1Loss=L1Loss, Lpips=Lpips, FeaLoss=torch.tensor([0]), total=Loss)

            if i % int(train_total_iter/10) == 0:
                avg = trainMetirc.get_average_epoch()
                dt, remain_time, end_time = TR.get_iter_time(epoch=epoch - cur_epoch, iter=i)
                message = 'Train, Epoch: [%d]/[%d], Iter [%d]/[%d], L1 Loss:%f, Lpips:%f, CompareLoss:%f, Total Loss:%f' \
                          % (epoch, epochs, i, train_total_iter, avg[0], avg[1], avg[2], avg[3]) + \
                          ", Cost Time: " + dt + ", Remain Time: " + remain_time + ', End At: ' + end_time
                showMessage(message, os.path.join(run_dir, 'log.txt'))
        torch.cuda.empty_cache()

        scheduler.step()
        avg = trainMetirc.get_average_epoch()
        tb.add_scalar('Train_Loss/L1 Loss', avg[0], epoch)
        tb.add_scalar('Train_Loss/Lpips Loss', avg[1], epoch)
        tb.add_scalar('Train_Loss/Compare Loss', avg[2], epoch)
        tb.add_scalar('Train_Loss/Total Loss', avg[3], epoch)
        trainMetirc.update_epoch()

        message = '============Train %d done, loss:%f============' % (epoch, avg[3])
        showMessage(message, os.path.join(run_dir, 'log.txt'))

        with torch.no_grad():
            model.eval()
            psnr = 0
            ssim = 0
            count = 0
            for i, (event_vox, img, mask, gt_img) in enumerate(testLoader):
                event_vox = event_vox.cuda()
                img = img.cuda().float()
                mask = mask.cuda().float()
                gt_img = gt_img.cuda().float()

                output = model(event_vox, img, mask)

                L1Loss = loss_l1(output, gt_img)
                Lpips = torch.sum(loss_lpips.forward(output, gt_img, normalize=True))
                Loss = L1Loss + alpha * Lpips

                testMetirc.update(L1Loss=L1Loss, Lpips=Lpips, FeaLoss=torch.tensor([0]), total=Loss)

                output[output>1] = 1
                output[output<0] = 0
                output *= 255
                gt_img *= 255
                output = np.array(output[0].cpu().permute(1,2,0)).astype(np.uint8)
                gt_img = np.array(gt_img[0].cpu().permute(1,2,0)).astype(np.uint8)
                p = calpsnr(output, gt_img)
                s = calssim(output, gt_img)

                psnr += p
                ssim += s

                if count % 10 == 0:
                    avg = testMetirc.get_average_epoch()
                    message = 'Test, Epoch: [%d]/[%d], Iter [%d]/[%d], L1Loss:%f, Lpips:%f, TotalLoss:%f' \
                              % (epoch, epochs, count, test_total_iter, avg[0], avg[1], avg[3])
                    showMessage(message, os.path.join(run_dir, 'log.txt'))
                count += 1
            torch.cuda.empty_cache()

            psnr /= count
            ssim /= count

            ssim_list.append(ssim)
            psnr_list.append(psnr)

            avg = testMetirc.get_average_epoch()
            tb.add_scalar('Test_Loss/L1 Loss', avg[0], epoch)
            tb.add_scalar('Test_Loss/Lpips Loss', avg[1], epoch)
            tb.add_scalar('Test_Loss/Total Loss', avg[3], epoch)
            tb.add_scalar('Test_Loss/PSNR', psnr, epoch)
            tb.add_scalar('Test_Loss/SSIM', ssim, epoch)

            testMetirc.update_epoch()
            message = '============Epoch %d test done, loss:%f, PSNR:%f, SSIM:%f============' % (epoch, avg[3], psnr, ssim)
            showMessage(message, os.path.join(run_dir, 'log.txt'))

        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        tb.add_scalar('Lr', param_group['lr'], epoch)

        model_dict = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        early_stopping(avg[3], model_dict, epoch, run_dir)
        if ssim_list[-1] == max(ssim_list):
            torch.save(model_dict, os.path.join(run_dir, "checkpoint_max_ssim.pth"))
        if psnr_list[-1] == max(psnr_list):
            torch.save(model_dict, os.path.join(run_dir, "checkpoint_max_psnr.pth"))
        if early_stopping.early_stop:
            print('Stop!!!!')
            break