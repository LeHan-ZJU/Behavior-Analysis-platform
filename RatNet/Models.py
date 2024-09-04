import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


class Trans_Conv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.Up_conv = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.Up_conv(x)


class RatNet_Resnet2(nn.Module):
    def __init__(self, SubResnet, n_channels, nof_joints):
        super(RatNet_Resnet2, self).__init__()
        self.SubResnet = SubResnet
        self.n_classes = nof_joints
        self.n_channels = n_channels
        self.UpConv2 = Trans_Conv(2048, 256, 256)            
        self.UpConv3 = Trans_Conv(256, 64, 256)           
        self.UpConv4 = Trans_Conv(256, 64, 256)          
        self.outConv = nn.Conv2d(256, self.n_classes, kernel_size=(1, 1), stride=(1, 1))
        nn.init.normal_(self.outConv.weight, std=0.001)   
        nn.init.constant_(self.outConv.bias, 0)

    def forward(self, img):
        f2 = self.SubResnet(img)
        f3 = self.UpConv2(f2[0])
        f4 = self.UpConv3(f3)
        f4 = self.UpConv3(f4)
        out = self.outConv(f4)
        return out
