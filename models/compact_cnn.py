import torch.nn as nn
import torch
from torchvision import models
from .deform_conv_v2 import DeformConv2d
import collections
import torch.nn.functional as F


class CompactCNN(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    :deprecated: I think implement incorrectly, please use CompactCNNV2
    """
    def __init__(self, load_weights=False):
        super(CompactCNN, self).__init__()
        self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(40, 60, 3, padding=1)
        self.c2 = nn.Conv2d(60, 40, 3, padding=1)
        self.c3 = nn.Conv2d(40, 20, 3, padding=1)
        self.c4 = nn.Conv2d(20, 10, 3, padding=1)
        self.output = nn.Conv2d(10, 1, 1)

    def forward(self,x):
        x_red = self.max_pooling(F.relu(self.red_cnn(x), inplace=True))
        x_green = self.max_pooling(F.relu(self.green_cnn(x), inplace=True))
        x_blue = self.max_pooling(F.relu(self.blue_cnn(x), inplace=True))

        x = torch.cat((x_red, x_green, x_blue), 1)
        x = F.relu(self.c1(x), inplace=True)

        x = F.relu(self.c2(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c3(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c4(x), inplace=True)

        x = self.output(x)
        return x


class CompactCNNV2(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    """
    def __init__(self, load_weights=False):
        super(CompactCNNV2, self).__init__()
        self.model_note = "CCNN without batchnorm"
        self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)
        self.c0 = nn.Conv2d(40, 40, 3, padding=1)

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(40, 60, 3, padding=1)
        self.c2 = nn.Conv2d(60, 40, 3, padding=1)
        self.c3 = nn.Conv2d(40, 20, 3, padding=1)
        self.c4 = nn.Conv2d(20, 10, 3, padding=1)
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


class CompactCNNV7(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    """
    def __init__(self, load_weights=False):
        super(CompactCNNV7, self).__init__()
        self.model_note = "CCNN without batchnorm, max pooling after cat, does it do any good ?"
        self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)
        self.c0 = nn.Conv2d(40, 40, 3, padding=1)

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(40, 60, 3, padding=1)
        self.c2 = nn.Conv2d(60, 40, 3, padding=1)
        self.c3 = nn.Conv2d(40, 20, 3, padding=1)
        self.c4 = nn.Conv2d(20, 10, 3, padding=1)
        self.output = nn.Conv2d(10, 1, 1)

    def forward(self,x):
        x_red = F.relu(self.red_cnn(x), inplace=True)
        x_green = F.relu(self.green_cnn(x), inplace=True)
        x_blue = F.relu(self.blue_cnn(x), inplace=True)

        x = torch.cat((x_red, x_green, x_blue), 1)

        x = self.max_pooling(x)
        x = F.relu(self.c0(x), inplace=True)

        x = F.relu(self.c1(x), inplace=True)

        x = F.relu(self.c2(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c3(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c4(x), inplace=True)

        x = self.output(x)
        return x


class CompactCNNV7i(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    no inplace
    """
    def __init__(self, load_weights=False):
        super(CompactCNNV7i, self).__init__()
        self.model_note = "CCNN without batchnorm, max pooling after cat, does it do any good ?, no inplace relu"
        self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)
        self.c0 = nn.Conv2d(40, 40, 3, padding=1)

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(40, 60, 3, padding=1)
        self.c2 = nn.Conv2d(60, 40, 3, padding=1)
        self.c3 = nn.Conv2d(40, 20, 3, padding=1)
        self.c4 = nn.Conv2d(20, 10, 3, padding=1)
        self.output = nn.Conv2d(10, 1, 1)

    def forward(self,x):
        x_red = F.relu(self.red_cnn(x))
        x_green = F.relu(self.green_cnn(x))
        x_blue = F.relu(self.blue_cnn(x))

        x = torch.cat((x_red, x_green, x_blue), 1)
        x = self.max_pooling(x)

        x = F.relu(self.c0(x))
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = self.max_pooling(x)

        x = F.relu(self.c3(x))
        x = self.max_pooling(x)

        x = F.relu(self.c4(x))
        x = self.output(x)
        return x

class CompactCNNV9(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    """
    def __init__(self, load_weights=False):
        super(CompactCNNV9, self).__init__()
        self.model_note = "CCNNv7 but use leaky relu"
        self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)
        self.c0 = nn.Conv2d(40, 40, 3, padding=1)

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(40, 60, 3, padding=1)
        self.c2 = nn.Conv2d(60, 40, 3, padding=1)
        self.c3 = nn.Conv2d(40, 20, 3, padding=1)
        self.c4 = nn.Conv2d(20, 10, 3, padding=1)
        self.output = nn.Conv2d(10, 1, 1)

    def forward(self,x):
        x_red = F.leaky_relu(self.red_cnn(x), inplace=True)
        x_green = F.leaky_relu(self.green_cnn(x), inplace=True)
        x_blue = F.leaky_relu(self.blue_cnn(x), inplace=True)

        x = torch.cat((x_red, x_green, x_blue), 1)

        x = self.max_pooling(x)
        x = F.leaky_relu(self.c0(x), inplace=True)

        x = F.leaky_relu(self.c1(x), inplace=True)

        x = F.leaky_relu(self.c2(x), inplace=True)
        x = self.max_pooling(x)

        x = F.leaky_relu(self.c3(x), inplace=True)
        x = self.max_pooling(x)

        x = F.leaky_relu(self.c4(x), inplace=True)

        x = self.output(x)
        return x



class CompactCNNV8(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    """
    def __init__(self, load_weights=False):
        super(CompactCNNV8, self).__init__()
        self.model_note = "CCNNv7, now we change c0 to 40-60, see if v8 better than v7"
        self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)
        self.c0 = nn.Conv2d(40, 60, 3, padding=1)

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(60, 60, 3, padding=1)
        self.c2 = nn.Conv2d(60, 40, 3, padding=1)
        self.c3 = nn.Conv2d(40, 20, 3, padding=1)
        self.c4 = nn.Conv2d(20, 10, 3, padding=1)
        self.output = nn.Conv2d(10, 1, 1)

    def forward(self,x):
        x_red = F.relu(self.red_cnn(x), inplace=True)
        x_green = F.relu(self.green_cnn(x), inplace=True)
        x_blue = F.relu(self.blue_cnn(x), inplace=True)

        x = torch.cat((x_red, x_green, x_blue), 1)

        x = self.max_pooling(x)
        x = F.relu(self.c0(x), inplace=True)

        x = F.relu(self.c1(x), inplace=True)

        x = F.relu(self.c2(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c3(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c4(x), inplace=True)

        x = self.output(x)
        return x


class CompactCNNV6(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    with a bunch of batch norm
    """
    def __init__(self, load_weights=False):
        super(CompactCNNV6, self).__init__()
        self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)

        self.red_bn = nn.BatchNorm2d(10)
        self.green_bn = nn.BatchNorm2d(14)
        self.blue_bn = nn.BatchNorm2d(16)


        self.c0 = nn.Conv2d(40, 40, 3, padding=1)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c1 = nn.Conv2d(40, 60, 3, padding=1)
        self.c2 = nn.Conv2d(60, 40, 3, padding=1)
        self.c3 = nn.Conv2d(40, 20, 3, padding=1)
        self.c4 = nn.Conv2d(20, 10, 3, padding=1)

        self.bn0 = nn.BatchNorm2d(40)
        self.bn1 = nn.BatchNorm2d(60)
        self.bn2 = nn.BatchNorm2d(40)
        self.bn3 = nn.BatchNorm2d(20)
        self.bn4 = nn.BatchNorm2d(10)

        self.output = nn.Conv2d(10, 1, 1)

    def forward(self, x):
        x_red = self.max_pooling(F.relu(self.red_bn(self.red_cnn(x)), inplace=True))
        x_green = self.max_pooling(F.relu(self.green_bn(self.green_cnn(x)), inplace=True))
        x_blue = self.max_pooling(F.relu(self.blue_bn(self.blue_cnn(x)), inplace=True))

        x = torch.cat((x_red, x_green, x_blue), 1)
        x = F.relu(self.bn0(self.c0(x)), inplace=True)

        x = F.relu(self.bn1(self.c1(x)), inplace=True)

        x = F.relu(self.bn2(self.c2(x)), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.bn3(self.c3(x)), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.bn4(self.c4(x)), inplace=True)

        x = self.output(x)
        return x


class CompactDilatedCNN(nn.Module):
    """

    """
    def __init__(self, load_weights=False):
        super(CompactDilatedCNN, self).__init__()
        self.red_cnn = nn.Conv2d(3, 10, 5, dilation=3, padding=6)
        self.green_cnn = nn.Conv2d(3, 14, 5, dilation=2, padding=4)
        self.blue_cnn = nn.Conv2d(3, 16, 5, dilation=1, padding=2)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(40, 60, 3, padding=1)
        self.c2 = nn.Conv2d(60, 40, 3, padding=1)
        self.c3 = nn.Conv2d(40, 20, 3, padding=1)
        self.c4 = nn.Conv2d(20, 10, 3, padding=1)
        self.output = nn.Conv2d(10, 1, 1)

    def forward(self,x):
        x_red = self.max_pooling(F.relu(self.red_cnn(x), inplace=True))
        x_green = self.max_pooling(F.relu(self.green_cnn(x), inplace=True))
        x_blue = self.max_pooling(F.relu(self.blue_cnn(x), inplace=True))

        x = torch.cat((x_red, x_green, x_blue), 1)
        x = F.relu(self.c1(x), inplace=True)

        x = F.relu(self.c2(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c3(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c4(x), inplace=True)

        x = self.output(x)
        return x


class DefDilatedCCNN(nn.Module):
    """
    fail reason: out of cuda memory at red_cnn
    possible fix: try torchvision deform conv
    """
    def __init__(self, load_weights=False):
        super(DefDilatedCCNN, self).__init__()

        self.red_cnn = DeformConv2d(3, 10, 9, padding=4)
        self.green_cnn = DeformConv2d(3, 14, 7, padding=3)
        self.blue_cnn = DeformConv2d(3, 16, 5, padding=2)

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(40, 60, 3, dilation=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(60)
        self.c2 = nn.Conv2d(60, 40, 3, dilation=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(40)
        self.c3 = nn.Conv2d(40, 20, 3, dilation=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(20)
        self.c4 = nn.Conv2d(20, 10, 3, dilation=2, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(10)
        self.output = nn.Conv2d(10, 1, 1)

    def forward(self,x):
        x_red = self.max_pooling(F.relu(self.red_cnn(x), inplace=True))
        x_green = self.max_pooling(F.relu(self.green_cnn(x), inplace=True))
        x_blue = self.max_pooling(F.relu(self.blue_cnn(x), inplace=True))

        x = torch.cat((x_red, x_green, x_blue), 1)
        x = F.relu(self.bn1(self.c1(x)), inplace=True)

        x = F.relu(self.bn2(self.c2(x)), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.bn3(self.c3(x)), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.bn4(self.c4(x)), inplace=True)

        x = self.output(x)
        return x

class DilatedCCNNv2(nn.Module):
    """

    """
    def __init__(self, load_weights=False):
        super(DilatedCCNNv2, self).__init__()

        self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(40, 60, 3, dilation=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(60)
        self.c2 = nn.Conv2d(60, 40, 3, dilation=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(40)
        self.c3 = nn.Conv2d(40, 20, 3, dilation=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(20)
        self.c4 = nn.Conv2d(20, 10, 3, dilation=2, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(10)
        self.output = nn.Conv2d(10, 1, 1)

    def forward(self,x):
        x_red = self.max_pooling(F.relu(self.red_cnn(x), inplace=True))
        x_green = self.max_pooling(F.relu(self.green_cnn(x), inplace=True))
        x_blue = self.max_pooling(F.relu(self.blue_cnn(x), inplace=True))

        x = torch.cat((x_red, x_green, x_blue), 1)
        x = F.relu(self.bn1(self.c1(x)), inplace=True)

        x = F.relu(self.bn2(self.c2(x)), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.bn3(self.c3(x)), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.bn4(self.c4(x)), inplace=True)

        x = self.output(x)
        return x