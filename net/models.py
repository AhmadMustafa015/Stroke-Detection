import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch
import pretrainedmodels
import math
from efficientnet_pytorch import EfficientNet

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], 1)

def l2_norm(input, axis=1):
    norm = torch.norm(input,2, axis, True)
    output = torch.div(input, norm)
    return output

class se_resnext50_32x4d(nn.Module):
    def __init__(self):
        super(se_resnext50_32x4d, self).__init__()
        self.model_ft = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, 3, bias=True))

    def forward(self, x):
        x = self.model_ft(x)
        return x

class se_resnext101_32x4d(nn.Module):
    def __init__(self):
        super(se_resnext101_32x4d, self).__init__()
        self.model_ft = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, 3, bias=True))

    def forward(self, x):
        x = self.model_ft(x)
        return x

class DenseNet169_change_avg(nn.Module):
    def __init__(self):
        super(DenseNet169_change_avg, self).__init__()
        self.densenet169 = torchvision.models.densenet169(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1664, 3)
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):
        x = self.densenet169(x)      
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)

        return x

class DenseNet121_change_avg(nn.Module):
    def __init__(self):
        super(DenseNet121_change_avg, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1024, 3)
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):
        x = self.densenet121(x)      
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.mlp(x)
        
        return x
class unet_brain_seg(nn.Module):
    def __init__(self):
        super(unet_brain_seg, self).__init__()
        self.unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=2, init_features=32,pretrained=False)
        #self.num_ftrs = self.unet.conv.in_channels
        #self.kernelsize = self.unet.conv.kernel_size
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        #self.unet.conv = nn.Conv2d(self.num_ftrs, 2, kernel_size=self.kernelsize)

    def forward(self, x):
        x = self.unet(x)
        return x

