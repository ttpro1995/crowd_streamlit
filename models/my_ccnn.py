import torch.nn as nn
import torch
from torchvision import models
from .deform_conv_v2 import DeformConv2d, TorchVisionBasicDeformConv2d
import collections
import torch.nn.functional as F


class CustomCNNv1(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    the improve version

    we change 5x5 7x7 9x9 with 3x3
    """
    def __init__(self, load_weights=False):
        super(CustomCNNv1, self).__init__()
        # self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        # self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        # self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)

        # ideal from crowd counting using DMCNN
        self.red_cnn_1 = nn.Conv2d(3, 10, 3, padding=1)
        self.red_cnn_2 = nn.Conv2d(10, 10, 3, padding=1)
        self.red_cnn_3 = nn.Conv2d(10, 10, 3, padding=1)
        self.red_cnn_4 = nn.Conv2d(10, 10, 3, padding=1)

        self.green_cnn_1 = nn.Conv2d(3, 14, 3, padding=1)
        self.green_cnn_2 = nn.Conv2d(14, 14, 3, padding=1)
        self.green_cnn_3 = nn.Conv2d(14, 14, 3, padding=1)

        self.blue_cnn_1 = nn.Conv2d(3, 16, 3, padding=1)
        self.blue_cnn_2 = nn.Conv2d(16, 16, 3, padding=1)

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

        x_red = F.relu(self.red_cnn_1(x), inplace=True)
        x_red = F.relu(self.red_cnn_2(x_red), inplace=True)
        x_red = F.relu(self.red_cnn_3(x_red), inplace=True)
        x_red = F.relu(self.red_cnn_4(x_red), inplace=True)
        x_red = self.max_pooling(x_red)

        x_green = F.relu(self.green_cnn_1(x), inplace=True)
        x_green = F.relu(self.green_cnn_2(x_green), inplace=True)
        x_green = F.relu(self.green_cnn_3(x_green), inplace=True)
        x_green = self.max_pooling(x_green)

        x_blue = F.relu(self.blue_cnn_1(x), inplace=True)
        x_blue = F.relu(self.blue_cnn_1(x_blue), inplace=True)
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


class CustomCNNv2(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    the improve version

    we change 5x5 7x7 9x9 with 3x3
    """
    def __init__(self, load_weights=False):
        super(CustomCNNv2, self).__init__()
        self.model_note = "We replace 5x5 7x7 9x9 with 3x3, no batchnorm yet"
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


class CustomCNNv3(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    the improve version

    v2
    we change 5x5 7x7 9x9 with 3x3

    v3
    add batch norm
    """
    def __init__(self, load_weights=False):
        super(CustomCNNv3, self).__init__()
        # self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        # self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        # self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)

        # ideal from crowd counting using DMCNN
        self.front_cnn_1 = nn.Conv2d(3, 20, 3, padding=1, bias=False)
        self.front_bn1 = nn.BatchNorm2d(20)
        self.front_cnn_2 = nn.Conv2d(20, 16, 3, padding=1, bias=False)
        self.front_bn2 = nn.BatchNorm2d(16)
        self.front_cnn_3 = nn.Conv2d(16, 14, 3, padding=1, bias=False)
        self.front_bn3 = nn.BatchNorm2d(14)
        self.front_cnn_4 = nn.Conv2d(14, 10, 3, padding=1, bias=False)
        self.front_bn4 = nn.BatchNorm2d(10)

        self.c0 = nn.Conv2d(40, 40, 3, padding=1)
        self.bn0 = nn.BatchNorm2d(40)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(40, 60, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(60)
        # ideal from CSRNet
        self.c2 = nn.Conv2d(60, 40, 3, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(40)
        self.c3 = nn.Conv2d(40, 20, 3, padding=2, dilation=2, bias=False)
        self.bn3 = nn.BatchNorm2d(20)
        self.c4 = nn.Conv2d(20, 10, 3, padding=2, dilation=2, bias=False)
        self.bn4 = nn.BatchNorm2d(10)
        self.output = nn.Conv2d(10, 1, 1)

    def forward(self,x):
        #x_red = self.max_pooling(F.relu(self.red_cnn(x), inplace=True))
        #x_green = self.max_pooling(F.relu(self.green_cnn(x), inplace=True))
        #x_blue = self.max_pooling(F.relu(self.blue_cnn(x), inplace=True))

        x_red = F.relu(self.front_bn1(self.front_cnn_1(x)), inplace=True)
        x_red = F.relu(self.front_bn2(self.front_cnn_2(x_red)), inplace=True)
        x_red = F.relu(self.front_bn3(self.front_cnn_3(x_red)), inplace=True)
        x_red = F.relu(self.front_bn4(self.front_cnn_4(x_red)), inplace=True)
        x_red = self.max_pooling(x_red)

        x_green = F.relu(self.front_bn1(self.front_cnn_1(x)), inplace=True)
        x_green = F.relu(self.front_bn2(self.front_cnn_2(x_green)), inplace=True)
        x_green = F.relu(self.front_bn3(self.front_cnn_3(x_green)), inplace=True)
        x_green = self.max_pooling(x_green)

        x_blue = F.relu(self.front_bn1(self.front_cnn_1(x)), inplace=True)
        x_blue = F.relu(self.front_bn2(self.front_cnn_2(x_blue)), inplace=True)
        x_blue = self.max_pooling(x_blue)

        x = torch.cat((x_red, x_green, x_blue), 1)
        x = F.relu(self.c0(x), inplace=True)

        x = F.relu(self.bn1(self.c1(x)), inplace=True)

        x = F.relu(self.bn2(self.c2(x)), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.bn3(self.c3(x)), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.bn4(self.c4(x)), inplace=True)

        x = self.output(x)
        return x


class CustomCNNv4(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    the improve version

    v4
    use deformable cnn


    """
    def __init__(self, load_weights=False):
        super(CustomCNNv4, self).__init__()
        self.red_cnn = TorchVisionBasicDeformConv2d(3, 10, 9, padding=4)
        self.bn_red = nn.BatchNorm2d(10)
        self.green_cnn = TorchVisionBasicDeformConv2d(3, 14, 7, padding=3)
        self.bn_green = nn.BatchNorm2d(14)
        self.blue_cnn = TorchVisionBasicDeformConv2d(3, 16, 5, padding=2)
        self.bn_blue = nn.BatchNorm2d(16)

        self.c0 = nn.Conv2d(40, 40, 3, padding=1)
        self.bn0 = nn.BatchNorm2d(40)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(40, 60, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(60)
        # ideal from CSRNet
        self.c2 = nn.Conv2d(60, 40, 3, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(40)
        self.c3 = nn.Conv2d(40, 20, 3, padding=2, dilation=2, bias=False)
        self.bn3 = nn.BatchNorm2d(20)
        self.c4 = nn.Conv2d(20, 10, 3, padding=2, dilation=2, bias=False)
        self.bn4 = nn.BatchNorm2d(10)
        self.output = nn.Conv2d(10, 1, 1)

    def forward(self, x):
        x_red = self.max_pooling(F.relu(self.bn_red(self.red_cnn(x))))
        x_green = self.max_pooling(F.relu(self.bn_green(self.green_cnn(x))))
        x_blue = self.max_pooling(F.relu(self.bn_blue(self.blue_cnn(x))))

        x = torch.cat((x_red, x_green, x_blue), 1)
        x = F.relu(self.c0(x))

        x = F.relu(self.bn1(self.c1(x)))

        x = F.relu(self.bn2(self.c2(x)))
        x = self.max_pooling(x)

        x = F.relu(self.bn3(self.c3(x)))
        x = self.max_pooling(x)

        x = F.relu(self.bn4(self.c4(x)))

        x = self.output(x)
        return x


class CustomCNNv5(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    the improve version

    v4
    use deformable cnn


    """
    def __init__(self, load_weights=False):
        super(CustomCNNv5, self).__init__()
        self.red_cnn = TorchVisionBasicDeformConv2d(3, 10, 9, padding=4)
        #self.bn_red = nn.BatchNorm2d(10)
        self.green_cnn = TorchVisionBasicDeformConv2d(3, 14, 7, padding=3)
        #self.bn_green = nn.BatchNorm2d(14)
        self.blue_cnn = TorchVisionBasicDeformConv2d(3, 16, 5, padding=2)
        #self.bn_blue = nn.BatchNorm2d(16)

        self.c0 = nn.Conv2d(40, 40, 3, padding=1)
        self.bn0 = nn.BatchNorm2d(40)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(40, 60, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(60)
        # ideal from CSRNet
        self.c2 = nn.Conv2d(60, 40, 3, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(40)
        self.c3 = nn.Conv2d(40, 20, 3, padding=2, dilation=2, bias=False)
        self.bn3 = nn.BatchNorm2d(20)
        self.c4 = nn.Conv2d(20, 10, 3, padding=2, dilation=2, bias=False)
        self.bn4 = nn.BatchNorm2d(10)
        self.output = nn.Conv2d(10, 1, 1)

    def forward(self, x):
        x_red = self.max_pooling(F.relu(self.red_cnn(x)))
        x_green = self.max_pooling(F.relu(self.green_cnn(x)))
        x_blue = self.max_pooling(F.relu(self.blue_cnn(x)))

        x = torch.cat((x_red, x_green, x_blue), 1)
        x = F.relu(self.c0(x))

        x = F.relu(self.bn1(self.c1(x)))

        x = F.relu(self.bn2(self.c2(x)))
        x = self.max_pooling(x)

        x = F.relu(self.bn3(self.c3(x)))
        x = self.max_pooling(x)

        x = F.relu(self.bn4(self.c4(x)))

        x = self.output(x)
        return x