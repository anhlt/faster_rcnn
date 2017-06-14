import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models.vgg import vgg16 as _vgg16


class Conv2d(nn.Module):
    """docstring for Conv2d"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 relu=True,
                 same_padding=False,
                 bn=False):

        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(
            out_channels, eps=0.001,
            momentum=0,
            affine=True
        ) if bn else None

        self.relu = nn.ReLU(inplace=True) if relu else None

        def forward(self, x):
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            if self.relu is not None:
                x = self.relu(x)


class FC(nn.Module):
    """docstring for FC"""

    def __init__(self,
                 in_features,
                 out_features,
                 relu=True
                 ):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def vgg16():
    full_model = _vgg16(pretrained=True)
    feature_model = full_model.features
    for name, module in feature_model.named_children():
        for param in module.parameters():
            param.requires_grad = False

    return feature_model

def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
    v = Variable(torch.from_numpy(x).type(dtype))
    if is_cuda:
        v = v.cuda()
    return v