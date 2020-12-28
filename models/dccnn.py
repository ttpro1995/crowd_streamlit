import torch.nn as nn
import torch
import collections
import torch.nn.functional as F

class DCCNN(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    """
    def __init__(self, load_weights=False):
        super(DCCNN, self).__init__()
        self.model_note = "BigTail12i, batchnorm default setting, add bn red, green, blue, i mean discard inplace"
        self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)

        self.c0 = nn.Conv2d(40, 40, 3, padding=2, dilation=2)

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pooling = nn.AvgPool2d(kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(40, 60, 3, padding=2, dilation=2)
        self.c2 = nn.Conv2d(60, 40, 3, padding=2, dilation=2)
        self.c3 = nn.Conv2d(40, 20, 3, padding=2, dilation=2)
        self.c4 = nn.Conv2d(20, 10, 3, padding=2, dilation=2)
        self.output = nn.Conv2d(10, 1, 1)

        self.bn_red = nn.BatchNorm2d(10)
        self.bn_green = nn.BatchNorm2d(14)
        self.bn_blue = nn.BatchNorm2d(16)

        self.bn00 = nn.BatchNorm2d(40)
        self.bn0 = nn.BatchNorm2d(40)
        self.bn1 = nn.BatchNorm2d(60)
        self.bn2 = nn.BatchNorm2d(40)
        self.bn3 = nn.BatchNorm2d(20)
        self.bn4 = nn.BatchNorm2d(10)

    def forward(self,x):
        x_red = F.relu(self.red_cnn(x))
        x_red = self.bn_red(x_red)
        x_green = F.relu(self.green_cnn(x))
        x_green = self.bn_green(x_green)
        x_blue = F.relu(self.blue_cnn(x))
        x_blue = self.bn_blue(x_blue)

        x = torch.cat((x_red, x_green, x_blue), 1)
        x = self.bn00(x)
        x = self.max_pooling(x)

        x = F.relu(self.c0(x))
        x = self.bn0(x)
        x = F.relu(self.c1(x))
        x = self.bn1(x)
        x = self.avg_pooling(x)

        x = F.relu(self.c2(x))
        x = self.bn2(x)

        x = F.relu(self.c3(x))
        x = self.bn3(x)
        x = self.avg_pooling(x)

        x = F.relu(self.c4(x))
        x = self.bn4(x)
        x = self.output(x)
        return x