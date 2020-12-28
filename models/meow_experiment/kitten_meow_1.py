import torch.nn as nn
import torch
from torchvision import models
from models.deform_conv_v2 import DeformConv2d, TorchVisionBasicDeformConv2d
import collections
import torch.nn.functional as F

class M3(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    the improve version

    we change 5x5 7x7 9x9 with 3x3
    Keep the tail
    """
    def __init__(self, load_weights=False):
        super(M3, self).__init__()
        self.model_note = "We replace 5x5 7x7 9x9 with 3x3, no batchnorm yet, tail dilated"
        # self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        # self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        # self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)

        # ideal from crowd counting using DMCNN
        self.front_cnn_1 = nn.Conv2d(3, 20, 3, padding=1)
        self.front_cnn_2 = nn.Conv2d(20, 16, 3, padding=1)
        self.front_cnn_3 = nn.Conv2d(16, 14, 3, padding=1)
        self.front_cnn_4 = nn.Conv2d(14, 10, 3, padding=1)

        self.c0 = nn.Conv2d(40, 40, 3, padding=1)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(40, 60, 3, padding=1)

        # ideal from CSRNet
        self.c2 = nn.Conv2d(60, 40, 3, padding=2, dilation=2)
        self.c3 = nn.Conv2d(40, 20, 3, padding=2, dilation=2)
        self.c4 = nn.Conv2d(20, 10, 3, padding=2, dilation=2)
        self.output = nn.Conv2d(10, 1, 1)

    def forward(self,x):
        #x_red = self.max_pooling(F.relu(self.red_cnn(x), inplace=True))
        #x_green = self.max_pooling(F.relu(self.green_cnn(x), inplace=True))
        #x_blue = self.max_pooling(F.relu(self.blue_cnn(x), inplace=True))

        x_red = F.relu(self.front_cnn_1(x), inplace=True)
        x_red = F.relu(self.front_cnn_2(x_red), inplace=True)
        x_red = F.relu(self.front_cnn_3(x_red), inplace=True)
        x_red = F.relu(self.front_cnn_4(x_red), inplace=True)
        x_red = self.max_pooling(x_red)

        x_green = F.relu(self.front_cnn_1(x), inplace=True)
        x_green = F.relu(self.front_cnn_2(x_green), inplace=True)
        x_green = F.relu(self.front_cnn_3(x_green), inplace=True)
        x_green = self.max_pooling(x_green)

        x_blue = F.relu(self.front_cnn_1(x), inplace=True)
        x_blue = F.relu(self.front_cnn_2(x_blue), inplace=True)
        x_blue = self.max_pooling(x_blue)

        x = torch.cat((x_red, x_green, x_blue), 1)
        x = F.relu(self.c0(x), inplace=True)

        x = F.relu(self.c1(x), inplace=True)

        x = F.relu(self.c2(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c3(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c4(x), inplace=True)

        x = self.output(x)
        return x


class M1(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    the improve version

    we change 5x5 7x7 9x9 with 3x3
    Keep the tail
    """
    def __init__(self, load_weights=False):
        super(M1, self).__init__()
        self.model_note = "We replace 5x5 7x7 9x9 with 3x3, no batchnorm yet, keep tail, no dilated"
        # self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        # self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        # self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)

        # ideal from crowd counting using DMCNN
        self.front_cnn_1 = nn.Conv2d(3, 20, 3, padding=1)
        self.front_cnn_2 = nn.Conv2d(20, 16, 3, padding=1)
        self.front_cnn_3 = nn.Conv2d(16, 14, 3, padding=1)
        self.front_cnn_4 = nn.Conv2d(14, 10, 3, padding=1)

        self.c0 = nn.Conv2d(40, 40, 3, padding=1)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(40, 60, 3, padding=1)

        # ideal from CSRNet
        self.c2 = nn.Conv2d(60, 40, 3, padding=1)
        self.c3 = nn.Conv2d(40, 20, 3, padding=1)
        self.c4 = nn.Conv2d(20, 10, 3, padding=1)
        self.output = nn.Conv2d(10, 1, 1)

    def forward(self,x):
        #x_red = self.max_pooling(F.relu(self.red_cnn(x), inplace=True))
        #x_green = self.max_pooling(F.relu(self.green_cnn(x), inplace=True))
        #x_blue = self.max_pooling(F.relu(self.blue_cnn(x), inplace=True))

        x_red = F.relu(self.front_cnn_1(x), inplace=True)
        x_red = F.relu(self.front_cnn_2(x_red), inplace=True)
        x_red = F.relu(self.front_cnn_3(x_red), inplace=True)
        x_red = F.relu(self.front_cnn_4(x_red), inplace=True)
        x_red = self.max_pooling(x_red)

        x_green = F.relu(self.front_cnn_1(x), inplace=True)
        x_green = F.relu(self.front_cnn_2(x_green), inplace=True)
        x_green = F.relu(self.front_cnn_3(x_green), inplace=True)
        x_green = self.max_pooling(x_green)

        x_blue = F.relu(self.front_cnn_1(x), inplace=True)
        x_blue = F.relu(self.front_cnn_2(x_blue), inplace=True)
        x_blue = self.max_pooling(x_blue)

        x = torch.cat((x_red, x_green, x_blue), 1)
        x = F.relu(self.c0(x), inplace=True)

        x = F.relu(self.c1(x), inplace=True)

        x = F.relu(self.c2(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c3(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c4(x), inplace=True)

        x = self.output(x)
        return x


class M2(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    """
    def __init__(self, load_weights=False):
        super(M2, self).__init__()
        self.model_note = "No batchnorm, keep head, but dilated tail"
        self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)
        self.c0 = nn.Conv2d(40, 40, 3, padding=1)

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(40, 60, 3, padding=2, dilation=2)
        self.c2 = nn.Conv2d(60, 40, 3, padding=2, dilation=2)
        self.c3 = nn.Conv2d(40, 20, 3, padding=2, dilation=2)
        self.c4 = nn.Conv2d(20, 10, 3, padding=2, dilation=2)
        self.output = nn.Conv2d(10, 1, 1)

    def forward(self,x):
        x_red = self.max_pooling(F.relu(self.red_cnn(x), inplace=True))
        x_green = self.max_pooling(F.relu(self.green_cnn(x), inplace=True))
        x_blue = self.max_pooling(F.relu(self.blue_cnn(x), inplace=True))

        x = torch.cat((x_red, x_green, x_blue), 1)
        x = F.relu(self.c0(x), inplace=True)

        x = F.relu(self.c1(x), inplace=True)

        x = F.relu(self.c2(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c3(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c4(x), inplace=True)

        x = self.output(x)
        return x


class M4(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    the improve version

    we change 5x5 7x7 9x9 with 3x3

    change tail to dilated max 60
    """
    def __init__(self, load_weights=False):
        super(M4, self).__init__()
        self.model_note = "We replace 5x5 7x7 9x9 with 3x3, no batchnorm yet, change tail to dilated max 60 with dilated 2"
        # self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        # self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        # self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)

        # ideal from crowd counting using DMCNN
        self.front_cnn_1 = nn.Conv2d(3, 20, 3, padding=1)
        self.front_cnn_2 = nn.Conv2d(20, 16, 3, padding=1)
        self.front_cnn_3 = nn.Conv2d(16, 14, 3, padding=1)
        self.front_cnn_4 = nn.Conv2d(14, 10, 3, padding=1)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c0 = nn.Conv2d(40, 60, 3, padding=2, dilation=2)
        self.c1 = nn.Conv2d(60, 60, 3, padding=2, dilation=2)
        self.c2 = nn.Conv2d(60, 60, 3, padding=2, dilation=2)
        self.c3 = nn.Conv2d(60, 30, 3, padding=2, dilation=2)
        self.c4 = nn.Conv2d(30, 15, 3, padding=2, dilation=2)
        self.c5 = nn.Conv2d(15, 10, 3, padding=2, dilation=2)
        self.output = nn.Conv2d(10, 1, 1)

    def forward(self,x):
        #x_red = self.max_pooling(F.relu(self.red_cnn(x), inplace=True))
        #x_green = self.max_pooling(F.relu(self.green_cnn(x), inplace=True))
        #x_blue = self.max_pooling(F.relu(self.blue_cnn(x), inplace=True))

        x_red = F.relu(self.front_cnn_1(x), inplace=True)
        x_red = F.relu(self.front_cnn_2(x_red), inplace=True)
        x_red = F.relu(self.front_cnn_3(x_red), inplace=True)
        x_red = F.relu(self.front_cnn_4(x_red), inplace=True)
        x_red = self.max_pooling(x_red)

        x_green = F.relu(self.front_cnn_1(x), inplace=True)
        x_green = F.relu(self.front_cnn_2(x_green), inplace=True)
        x_green = F.relu(self.front_cnn_3(x_green), inplace=True)
        x_green = self.max_pooling(x_green)

        x_blue = F.relu(self.front_cnn_1(x), inplace=True)
        x_blue = F.relu(self.front_cnn_2(x_blue), inplace=True)
        x_blue = self.max_pooling(x_blue)

        x = torch.cat((x_red, x_green, x_blue), 1)
        x = F.relu(self.c0(x), inplace=True)

        x = F.relu(self.c1(x), inplace=True)

        x = F.relu(self.c2(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c3(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c4(x), inplace=True)

        x = F.relu(self.c5(x), inplace=True)

        x = self.output(x)
        return x


class H1_Bigtail3(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    the improve version

    we change 5x5 7x7 9x9 with 3x3
    Keep the tail
    """
    def __init__(self, load_weights=False):
        super(H1_Bigtail3, self).__init__()
        self.model_note = "headh1, all 20, with bigtail3 dilated"

        # ideal from crowd counting using DMCNN
        self.front_cnn_1 = nn.Conv2d(3, 10, 3, padding=1)
        self.front_cnn_2 = nn.Conv2d(10, 20, 3, padding=1)
        self.front_cnn_3 = nn.Conv2d(20, 20, 3, padding=1)
        self.front_cnn_4 = nn.Conv2d(20, 20, 3, padding=1)

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c0 = nn.Conv2d(60, 60, 3, padding=2, dilation=2)
        self.c1 = nn.Conv2d(60, 60, 3, padding=2, dilation=2)
        self.c2 = nn.Conv2d(60, 60, 3, padding=2, dilation=2)
        self.c3 = nn.Conv2d(60, 30, 3, padding=2, dilation=2)
        self.c4 = nn.Conv2d(30, 15, 3, padding=2, dilation=2)
        self.c5 = nn.Conv2d(15, 10, 3, padding=2, dilation=2)

        self.output = nn.Conv2d(10, 1, 1)

    def forward(self, x):
        x_red = F.relu(self.front_cnn_1(x), inplace=True)
        x_red = F.relu(self.front_cnn_2(x_red), inplace=True)
        x_red = F.relu(self.front_cnn_3(x_red), inplace=True)
        x_red = F.relu(self.front_cnn_4(x_red), inplace=True)
        x_red = self.max_pooling(x_red)

        x_green = F.relu(self.front_cnn_1(x), inplace=True)
        x_green = F.relu(self.front_cnn_2(x_green), inplace=True)
        x_green = F.relu(self.front_cnn_3(x_green), inplace=True)
        x_green = self.max_pooling(x_green)

        x_blue = F.relu(self.front_cnn_1(x), inplace=True)
        x_blue = F.relu(self.front_cnn_2(x_blue), inplace=True)
        x_blue = self.max_pooling(x_blue)

        # x = self.max_pooling(x)
        x = torch.cat((x_red, x_green, x_blue), 1)
        x = F.relu(self.c0(x), inplace=True)

        x = F.relu(self.c1(x), inplace=True)

        x = F.relu(self.c2(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c3(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c4(x), inplace=True)
        x = F.relu(self.c5(x), inplace=True)

        x = self.output(x)
        return x
