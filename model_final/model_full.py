import torch
from torch import nn
from model_final import SizeAdapter, UNet, UNetLight
from model_final.frame_net import FrameEncoder, PConvDecoder
from model_final.event_net import EventEncoder, EventFrameDecoder


class EventFrameDeOcc(nn.Module):
    def __init__(self, inChannels_frame, inChannels_event, norm='BN', img_channel=3):
        super(EventFrameDeOcc, self).__init__()
        self.size_adapter = SizeAdapter(32)
        layers = [128, 128, 256, 256, 512, 512, 512]
        self.EventReFocus = UNetLight(inChannels_event, inChannels_event, norm=norm)
        self.FrameEncoder = FrameEncoder(inChannels_frame, self.size_adapter, layers, norm=norm, img_channel=img_channel)
        self.EventEncoder = EventEncoder(inChannels_event, self.size_adapter, layers, norm=norm)
        self.EventFrameDecoder = EventFrameDecoder(img_channel, self.size_adapter, layers, norm=norm)
        self.FrameDecoder = PConvDecoder(img_channel, self.size_adapter, layers, norm=norm)
        self.Fusion = UNet(img_channel*2, img_channel)

    def forward(self, event_vox, img, initMask):
        image_features, outputMask, predMask = self.FrameEncoder(img, initMask)
        event_with_refocus = self.EventReFocus(event_vox)
        event_features = self.EventEncoder(event_with_refocus)
        output1 = self.EventFrameDecoder(image_features, event_features)
        output2, _ = self.FrameDecoder(image_features, outputMask)
        output = self.Fusion(torch.cat([output1, output2], dim=1))
        return output