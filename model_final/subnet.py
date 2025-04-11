import torch
import torch.nn.functional as F
import datetime
import math
from torch import nn


def closest_larger_multiple_of_minimum_size(size, minimum_size):
    return int(math.ceil(size / minimum_size) * minimum_size)


class SizeAdapter(object):
    """Converts size of input to standard size.
    Practical deep network works only with input images
    which height and width are multiples of a minimum size.
    This class allows to pass to the network images of arbitrary
    size, by padding the input to the closest multiple
    and unpadding the network's output to the original size.
    """
    def __init__(self, minimum_size=64):
        self._minimum_size = minimum_size
        self._pixels_pad_to_width = None
        self._pixels_pad_to_height = None

    def _closest_larger_multiple_of_minimum_size(self, size):
        return closest_larger_multiple_of_minimum_size(size, self._minimum_size)

    def pad(self, network_input):
        """Returns "network_input" paded with zeros to the "standard" size.
        The "standard" size correspond to the height and width that
        are closest multiples of "minimum_size". The method pads
        height and width  and and saves padded values. These
        values are then used by "unpad_output" method.
        """
        height, width = network_input.size()[-2:]
        self._pixels_pad_to_height = (self._closest_larger_multiple_of_minimum_size(height) - height)
        self._pixels_pad_to_width = (self._closest_larger_multiple_of_minimum_size(width) - width)
        return nn.ZeroPad2d((self._pixels_pad_to_width, 0, self._pixels_pad_to_height, 0))(network_input)

    def unpad(self, network_output):
        """Returns "network_output" cropped to the original size.
        The cropping is performed using values save by the "pad_input"
        method.
        """
        # print(self._pixels_pad_to_height, self._pixels_pad_to_width)
        return network_output[..., self._pixels_pad_to_height:, self._pixels_pad_to_width:]


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


class Attention_block(nn.Module):
    def __init__(self, input_channel=64):
        super(Attention_block, self).__init__()
        self.fc1 = nn.Linear(input_channel, int(input_channel/8))
        self.fc2 = nn.Linear(int(input_channel/8), input_channel)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, infeature):
        b, c, h, w = infeature.shape
        max_f = F.max_pool2d(infeature, kernel_size=[h, w]).reshape(b, 1, c)
        avg_f = F.avg_pool2d(infeature, kernel_size=[h, w]).reshape(b, 1, c)

        cha_f = torch.cat([max_f, avg_f], dim=1)
        out1 = self.fc2(self.relu(self.fc1(cha_f)))
        channel_attention = self.sigmoid(out1[:, 0, :] + out1[:, 1, :]).reshape(b, c, 1, 1)
        feature_with_channel_attention = infeature * channel_attention
        return feature_with_channel_attention


class up(nn.Module):
    def __init__(self, inChannels, outChannels, norm='BN'):
        super(up, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inChannels, outChannels // 4, 1, 1, 0), nn.BatchNorm2d(outChannels // 4)) if norm else nn.Conv2d(inChannels, outChannels // 4, 1, 1, 0)
        self.conv3 = nn.Sequential(nn.Conv2d(inChannels, outChannels // 4, 3, 1, 1), nn.BatchNorm2d(outChannels // 4)) if norm else nn.Conv2d(inChannels, outChannels // 4, 3, 1, 1)
        self.conv5 = nn.Sequential(nn.Conv2d(inChannels, outChannels // 4, 5, 1, 2), nn.BatchNorm2d(outChannels // 4)) if norm else nn.Conv2d(inChannels, outChannels // 4, 5, 1, 2)
        self.conv7 = nn.Sequential(nn.Conv2d(inChannels, outChannels // 4, 7, 1, 3), nn.BatchNorm2d(outChannels // 4)) if norm else nn.Conv2d(inChannels, outChannels // 4, 7, 1, 3)
        self.conv = nn.Sequential(nn.Conv2d(outChannels, outChannels, 1, 1, 0), nn.BatchNorm2d(outChannels)) if norm else nn.Conv2d(outChannels, outChannels, 3, 1, 1)
        self.conv_out = nn.Sequential(nn.Conv2d(outChannels*3, outChannels, 3, 1, 1), nn.BatchNorm2d(outChannels)) if norm else nn.Conv2d(outChannels*3, outChannels, 3, 1, 1)

    def forward(self, x, skpCn1, skpCn2):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")

        x_1 = self.conv1(x)
        x_3 = self.conv3(x)
        x_5 = self.conv5(x)
        x_7 = self.conv7(x)
        x = F.leaky_relu(torch.cat((x_1, x_3, x_5, x_7), 1), negative_slope=0.1)
        x = F.leaky_relu(self.conv(x), negative_slope=0.1)
        x = torch.cat((x, skpCn1, skpCn2), 1)

        x = F.leaky_relu(self.conv_out(x), negative_slope=0.1)

        return x


class up1(nn.Module):
    def __init__(self, inChannels, outChannels, norm='BN'):
        super(up1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inChannels, outChannels // 4, 1, 1, 0), nn.BatchNorm2d(outChannels // 4)) if norm else nn.Conv2d(inChannels, outChannels // 4, 1, 1, 0)
        self.conv3 = nn.Sequential(nn.Conv2d(inChannels, outChannels // 4, 3, 1, 1), nn.BatchNorm2d(outChannels // 4)) if norm else nn.Conv2d(inChannels, outChannels // 4, 3, 1, 1)
        self.conv5 = nn.Sequential(nn.Conv2d(inChannels, outChannels // 4, 5, 1, 2), nn.BatchNorm2d(outChannels // 4)) if norm else nn.Conv2d(inChannels, outChannels // 4, 5, 1, 2)
        self.conv7 = nn.Sequential(nn.Conv2d(inChannels, outChannels // 4, 7, 1, 3), nn.BatchNorm2d(outChannels // 4)) if norm else nn.Conv2d(inChannels, outChannels // 4, 7, 1, 3)
        self.conv = nn.Sequential(nn.Conv2d(outChannels, outChannels, 1, 1, 0), nn.BatchNorm2d(outChannels)) if norm else nn.Conv2d(outChannels, outChannels, 3, 1, 1)
        self.conv_out = nn.Sequential(nn.Conv2d(outChannels*2, outChannels, 3, 1, 1), nn.BatchNorm2d(outChannels)) if norm else nn.Conv2d(outChannels*3, outChannels, 3, 1, 1)

    def forward(self, x, skpCn1):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")

        x_1 = self.conv1(x)
        x_3 = self.conv3(x)
        x_5 = self.conv5(x)
        x_7 = self.conv7(x)
        x = F.leaky_relu(torch.cat((x_1, x_3, x_5, x_7), 1), negative_slope=0.1)
        x = F.leaky_relu(self.conv(x), negative_slope=0.1)
        x = torch.cat((x, skpCn1), 1)

        x = F.leaky_relu(self.conv_out(x), negative_slope=0.1)

        return x


class down(nn.Module):
    def __init__(self, inChannels, outChannels, norm='BN'):
        super(down, self).__init__()
        self.relu1 = nn.LeakyReLU(0.1, True)
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=4, stride=2, padding=1)
        self.norm = norm
        if norm:
            self.bn = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        x = self.conv1(x)
        if self.norm:
            x = self.bn(x)
        x = self.relu1(x)
        return x


class up_light(nn.Module):
    def __init__(self, inChannels, outChannels, norm=False):
        super(up_light, self).__init__()
        bias = False if norm else True
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(outChannels)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(outChannels, track_running_stats=True)
        elif norm == False:
            print('No Normalization.')
        else:
            raise ValueError("Choose BN or IN or False.")

    def forward(self, x, skpCn):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.conv1(torch.cat((x, skpCn), 1))
        if self.norm:
            x = F.leaky_relu(self.bn1(x), negative_slope=0.1)
        else:
            x = F.leaky_relu(x, negative_slope=0.1)
        return x


class down_light(nn.Module):
    def __init__(self, inChannels, outChannels, norm='BN'):
        super(down_light, self).__init__()
        self.relu1 = nn.LeakyReLU(0.1, True)
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=4, stride=2, padding=1)
        self.norm = norm
        if norm:
            self.bn = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        x = self.conv1(x)
        if self.norm:
            x = self.bn(x)
        x = self.relu1(x)
        return x


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        self.multi_channel = True
        self.return_mask = True

        super(PartialConv2d, self).__init__(*args, **kwargs)

        self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            if self.weight_maskUpdater.type() != input.type():
                self.weight_maskUpdater = self.weight_maskUpdater.to(input)
            mask = mask_in
            self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                        padding=self.padding, dilation=self.dilation, groups=1)

            # for mixed precision training, change 1e-8 to 1e-6
            self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)

            self.update_mask = torch.clamp(self.update_mask, 0, 1)
            self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class PDown(nn.Module):
    def __init__(self, inChannels, outChannels, norm='BN'):
        conv_bias = False if norm else True
        super(PDown, self).__init__()
        self.relu1 = nn.LeakyReLU(0.1, True)
        self.conv = PartialConv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=4, stride=2, padding=1, bias=conv_bias)
        self.norm = norm
        if norm:
            self.bn = nn.BatchNorm2d(outChannels)

    def forward(self, x, mask):
        x, mask1 = self.conv(x, mask)
        if self.norm:
            x = self.bn(x)
        x = self.relu1(x)
        return x, mask1


class PUp(nn.Module):
    def __init__(self, inChannels, outChannels, norm='BN'):
        super(PUp, self).__init__()
        self.conv1 = PartialConv2d(in_channels=inChannels, out_channels=outChannels // 4, kernel_size=1, stride=1, padding=0)
        self.conv3 = PartialConv2d(in_channels=inChannels, out_channels=outChannels // 4, kernel_size=3, stride=1, padding=1)
        self.conv5 = PartialConv2d(in_channels=inChannels, out_channels=outChannels // 4, kernel_size=5, stride=1, padding=2)
        self.conv7 = PartialConv2d(in_channels=inChannels, out_channels=outChannels // 4, kernel_size=7, stride=1, padding=3)
        self.conv = nn.Sequential(nn.Conv2d(outChannels, outChannels, 1, 1, 0), nn.BatchNorm2d(outChannels)) if norm else nn.Conv2d(outChannels, outChannels, 3, 1, 1)
        self.conv_out = PartialConv2d(in_channels=outChannels*2, out_channels=outChannels, kernel_size=3, padding=1, stride=1)
        self.norm = norm
        if norm:
            self.bn1 = nn.BatchNorm2d(outChannels//4)
            self.bn2 = nn.BatchNorm2d(outChannels//4)
            self.bn3 = nn.BatchNorm2d(outChannels//4)
            self.bn4 = nn.BatchNorm2d(outChannels//4)
            self.bn5 = nn.BatchNorm2d(outChannels)

    def forward(self, x, mask, skpCn1, mask_skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')

        x_1, mask1 = self.conv1(x, mask)
        if self.norm:
            x_1 = self.bn1(x_1)
        x_3, mask3 = self.conv3(x, mask)
        if self.norm:
            x_3 = self.bn2(x_3)
        x_5, mask5 = self.conv5(x, mask)
        if self.norm:
            x_5 = self.bn3(x_5)
        x_7, mask7 = self.conv7(x, mask)
        if self.norm:
            x_7 = self.bn4(x_7)

        x = F.leaky_relu(torch.cat((x_1, x_3, x_5, x_7), 1), negative_slope=0.1)
        x = F.leaky_relu(self.conv(x), negative_slope=0.1)
        x = torch.cat((x, skpCn1), 1)
        mask = torch.cat([mask1, mask3, mask5, mask7, mask_skip], dim=1)
        x, opMask = self.conv_out(x, mask)
        if self.norm:
            x = self.bn5(x)
        x = F.leaky_relu(x, 0.1)
        return x, opMask


class UNetLight(nn.Module):
    def __init__(self, inChannels, outChannels, layers=[128,256,256,512,512], norm='BN'):
        super(UNetLight, self).__init__()
        self._size_adapter = SizeAdapter(32)
        self.conv1 = nn.Sequential(nn.Conv2d(inChannels, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, True))
        self.down1 = down_light(64, layers[0], norm)
        self.down2 = down_light(layers[0], layers[1], norm)
        self.down3 = down_light(layers[1], layers[2], norm)
        self.down4 = down_light(layers[2], layers[3], norm)
        self.down5 = down_light(layers[3], layers[4], norm)
        self.up1 = up_light(layers[4]+layers[3], layers[3], norm)
        self.up2 = up_light(layers[3]+layers[2], layers[2], norm)
        self.up3 = up_light(layers[2]+layers[1], layers[1], norm)
        self.up4 = up_light(layers[1]+layers[0], layers[0], norm)
        self.up5 = up_light(layers[0]+64, 64, norm)
        self.conv3 = nn.Conv2d(64, outChannels, 1)

    def forward(self, image):
        x = self._size_adapter.pad(image)  # 1280, 736
        s1 = self.conv1(x)
        s2 = self.down1(s1)  # 640, 368
        s3 = self.down2(s2)  # 320, 184
        s4 = self.down3(s3)  # 160, 92
        s5 = self.down4(s4)  # 80, 46
        x = self.down5(s5)  # 40, 23

        x = self.up1(x, s5)    #
        x = self.up2(x, s4)    #
        x = self.up3(x, s3)    #
        x = self.up4(x, s2)    #
        x = self.up5(x, s1)    #
        x = self.conv3(x)      #
        x = self._size_adapter.unpad(x)
        return x


class UNet(nn.Module):
    def __init__(self, inChannels, outChannels, layers=[128,128,256,256,512,512,512], norm='BN', att=False):
        super(UNet, self).__init__()
        self._size_adapter = SizeAdapter(32)
        self.conv1 = nn.Sequential(nn.Conv2d(inChannels, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, True))
        self.down1 = down(64, layers[0], norm)
        self.down2 = down(layers[0], layers[1], norm)
        self.down3 = down(layers[1], layers[2], norm)
        self.down4 = down(layers[2], layers[3], norm)
        self.down5 = down(layers[3], layers[4], norm)
        self.down6 = down(layers[4], layers[5], norm)
        self.down7 = down(layers[5], layers[6], norm)
        self.up7 = up1(layers[6], layers[5], norm)
        self.up6 = up1(layers[5], layers[4], norm)
        self.up5 = up1(layers[4], layers[3], norm)
        self.up4 = up1(layers[3], layers[2], norm)
        self.up3 = up1(layers[2], layers[1], norm)
        self.up2 = up1(layers[1], layers[0], norm)
        self.up1 = up1(layers[0], 64, norm)
        self.conv3 = nn.Conv2d(64, outChannels, kernel_size=3, stride=1, padding=1)
        self.att = att
        if att:
            self.att = nn.ModuleList([Attention_block(layers[0]), Attention_block(layers[1]), Attention_block(layers[2]),
                                      Attention_block(layers[3]), Attention_block(layers[4]), Attention_block(layers[5]), Attention_block(layers[6])])

    def forward(self, x):
        x = self._size_adapter.pad(x)  # 1280, 736
        s1 = self.conv1(x)
        s2 = self.att[0](self.down1(s1)) if self.att else self.down1(s1) # 128
        s3 = self.att[1](self.down2(s2)) if self.att else self.down2(s2) # 128
        s4 = self.att[2](self.down3(s3)) if self.att else self.down3(s3) # 128
        s5 = self.att[3](self.down4(s4)) if self.att else self.down4(s4) # 128
        s6 = self.att[4](self.down5(s5)) if self.att else self.down5(s5) # 128
        s7 = self.att[5](self.down6(s6)) if self.att else self.down6(s6) # 128
        x = self.att[6](self.down7(s7)) if self.att else self.down7(s7) # 128

        x = self.up7(x, s7)
        x = self.up6(x, s6)
        x = self.up5(x, s5)
        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.conv3(x)
        x = self._size_adapter.unpad(x)
        return x