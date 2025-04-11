import torch.nn as nn
from model_final import up, down, Attention_block
import torch


class EventEncoder(nn.Module):
    def __init__(self, inChannels, size_adapter, layers=[128,128,256,256,512,512,512], norm='BN', att=True):
        super().__init__()
        self._size_adapter = size_adapter
        self.conv1 = nn.Sequential(nn.Conv2d(inChannels, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, True))
        self.down1 = down(64, layers[0], norm)
        self.down2 = down(layers[0], layers[1], norm)
        self.down3 = down(layers[1], layers[2], norm)
        self.down4 = down(layers[2], layers[3], norm)
        self.down5 = down(layers[3], layers[4], norm)
        self.down6 = down(layers[4], layers[5], norm)
        self.down7 = down(layers[5], layers[6], norm)
        self.att = att
        if att:
            self.att = nn.ModuleList([Attention_block(layers[0]), Attention_block(layers[1]), Attention_block(layers[2]),
                                      Attention_block(layers[3]), Attention_block(layers[4]), Attention_block(layers[5]), Attention_block(layers[6])])

    def forward(self, input):
        output = []
        x = self._size_adapter.pad(input)  # 1280, 736
        s1 = self.conv1(x)
        output.append(s1)

        s2 = self.down1(s1)  # 128
        s2 = self.att[0](s2) if self.att else s2
        output.append(s2)

        s3 = self.down2(s2)  # 64
        s3 = self.att[1](s3) if self.att else s3
        output.append(s3)

        s4 = self.down3(s3)  # 32
        s4 = self.att[2](s4) if self.att else s4
        output.append(s4)

        s5 = self.down4(s4)  # 16
        s5 = self.att[3](s5) if self.att else s5
        output.append(s5)

        s6 = self.down5(s5)  # 8
        s6 = self.att[4](s6) if self.att else s6
        output.append(s6)

        s7 = self.down6(s6)  # 8
        s7 = self.att[5](s7) if self.att else s7
        output.append(s7)

        x = self.down7(s7)  # 8
        x = self.att[6](x) if self.att else x
        output.append(x)
        return output


class EventFrameDecoder(nn.Module):
    def __init__(self, outChannels, size_adapter, layers=[128,128,256,256,512,512,512], norm='BN'):
        super(EventFrameDecoder, self).__init__()
        self._size_adapter = size_adapter
        self.up7 = up(layers[6], layers[5], norm)
        self.up6 = up(layers[5], layers[4], norm)
        self.up5 = up(layers[4], layers[3], norm)
        self.up4 = up(layers[3], layers[2], norm)
        self.up3 = up(layers[2], layers[1], norm)
        self.up2 = up(layers[1], layers[0], norm)
        self.up1 = up(layers[0], 64, norm)
        self.conv3 = nn.Conv2d(64, outChannels, kernel_size=3, stride=1, padding=1)

    def forward(self, input1, input2):
        [s1, s2, s3, s4, s5, s6, s7, s] = input1
        [t1, t2, t3, t4, t5, t6, t7, t] = input2
        x = torch.mul(s, t)
        x = self.up7(x, s7, t7)
        x = self.up6(x, s6, t6)
        x = self.up5(x, s5, t5)
        x = self.up4(x, s4, t4)
        x = self.up3(x, s3, t3)
        x = self.up2(x, s2, t2)
        x = self.up1(x, s1, t1)

        x = self.conv3(x)
        x = self._size_adapter.unpad(x)
        return x