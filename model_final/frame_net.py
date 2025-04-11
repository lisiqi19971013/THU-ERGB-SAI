import torch.nn as nn
from model_final import PartialConv2d, PUp, PDown, UNetLight, Attention_block
import torch.nn.functional as F
import torch


class MaskPredNet(nn.Module):
    def __init__(self, img_channel=3):
        super(MaskPredNet, self).__init__()
        self.pred = UNetLight(img_channel+1, 1)
        self.img_channel = img_channel

    def forward(self, img, mask_init):
        bs, C, H ,W = img.shape
        img = img.reshape([-1, self.img_channel, H, W])
        mask_init = mask_init.reshape([-1, 1, H, W])
        x = torch.cat([img, mask_init], dim=1)
        mask = F.sigmoid(self.pred(x))
        mask = torch.repeat_interleave(mask, self.img_channel, dim=1)
        mask = mask.reshape([bs, -1, H, W])
        return mask


class PConvEncoder(nn.Module):
    def __init__(self, inChannels, size_adapter, layers=[128,128,256,256,512,512,512], norm='BN', att=True):
        super().__init__()
        self._size_adapter = size_adapter
        self.conv1 = PartialConv2d(in_channels=inChannels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        if norm:
            self.bn = nn.BatchNorm2d(64)
        self.norm = norm
        self.down1 = PDown(64, layers[0], norm)
        self.down2 = PDown(layers[0], layers[1], norm)
        self.down3 = PDown(layers[1], layers[2], norm)
        self.down4 = PDown(layers[2], layers[3], norm)
        self.down5 = PDown(layers[3], layers[4], norm)
        self.down6 = PDown(layers[4], layers[5], norm)
        self.down7 = PDown(layers[5], layers[6], norm)
        self.att = att
        if att:
            self.att = nn.ModuleList([Attention_block(layers[0]), Attention_block(layers[1]), Attention_block(layers[2]),
                                      Attention_block(layers[3]), Attention_block(layers[4]), Attention_block(layers[5]), Attention_block(layers[6])])

    def forward(self, input, mask):
        output = []
        mask_list = []
        x = self._size_adapter.pad(input)  # 1280, 736
        mask = self._size_adapter.pad(mask)  # 1280, 736

        s1, mask1 = self.conv1(x, mask)
        s1 = F.leaky_relu(self.bn(s1), 0.1) if self.norm else F.leaky_relu(s1, 0.1)
        output.append(s1)
        mask_list.append(mask1)

        s2, mask2 = self.down1(s1, mask1)  # 128
        s2 = self.att[0](s2) if self.att else s2
        output.append(s2)
        mask_list.append(mask2)

        s3, mask3 = self.down2(s2, mask2)  # 64
        s3 = self.att[1](s3) if self.att else s3
        output.append(s3)
        mask_list.append(mask3)

        s4, mask4 = self.down3(s3, mask3)  # 32
        s4 = self.att[2](s4) if self.att else s4
        output.append(s4)
        mask_list.append(mask4)

        s5, mask5 = self.down4(s4, mask4)  # 16
        s5 = self.att[3](s5) if self.att else s5
        output.append(s5)
        mask_list.append(mask5)

        s6, mask6 = self.down5(s5, mask5)  # 8
        s6 = self.att[4](s6) if self.att else s6
        output.append(s6)
        mask_list.append(mask6)

        s7, mask7 = self.down6(s6, mask6)  # 8
        s7 = self.att[5](s7) if self.att else s7
        output.append(s7)
        mask_list.append(mask7)

        x, mask8 = self.down7(s7, mask7)  # 8
        x = self.att[6](x) if self.att else x
        output.append(x)
        mask_list.append(mask8)

        return output, mask_list


class PConvDecoder(nn.Module):
    def __init__(self, outChannels, size_adapter, layers=[128,128,256,256,512,512,512], norm='BN'):
        super(PConvDecoder, self).__init__()
        self._size_adapter = size_adapter
        self.up7 = PUp(layers[6], layers[5], norm)
        self.up6 = PUp(layers[5], layers[4], norm)
        self.up5 = PUp(layers[4], layers[3], norm)
        self.up4 = PUp(layers[3], layers[2], norm)
        self.up3 = PUp(layers[2], layers[1], norm)
        self.up2 = PUp(layers[1], layers[0], norm)
        self.up1 = PUp(layers[0], 64, norm)
        self.conv3 = PartialConv2d(in_channels=64, out_channels=outChannels, kernel_size=3, stride=1, padding=1)

    def forward(self, input1, mask1):
        [s1, s2, s3, s4, s5, s6, s7, x] = input1
        [m1, m2, m3, m4, m5, m6, m7, m] = mask1
        x, mask7 = self.up7(x, m, s7, m7)
        x, mask6 = self.up6(x, mask7, s6, m6)
        x, mask5 = self.up5(x, mask6, s5, m5)
        x, mask4 = self.up4(x, mask5, s4, m4)
        x, mask3 = self.up3(x, mask4, s3, m3)
        x, mask2 = self.up2(x, mask3, s2, m2)
        x, mask1 = self.up1(x, mask2, s1, m1)

        x, opMask = self.conv3(x, mask1)
        x = self._size_adapter.unpad(x)
        opMask = self._size_adapter.unpad(opMask)
        return x, opMask


class FrameEncoder(nn.Module):
    def __init__(self, inChannels, size_adapter=None, layers=[128, 128, 256, 256, 512, 512, 512], norm='BN', img_channel=3):
        super().__init__()
        self.size_adapter = size_adapter
        self.MaskPredNet = MaskPredNet(img_channel)
        self.Encoder = PConvEncoder(inChannels, self.size_adapter, layers, norm=norm)

    def forward(self, img, initMask):
        predMask = self.MaskPredNet(img, initMask)
        features, outputMask = self.Encoder(img, predMask)
        return features, outputMask, predMask