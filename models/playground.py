import torch.nn as nn
import torch
from torchvision import models

class M1(nn.Module):
    def __init__(self):
        super(M1, self).__init__()
        self.backbone = models.vgg16(pretrained=True).features
        self.de1net = self.backbone[0:23]
        self.de2net = self.backbone[0:30]

    def forward(self, x):
        de1 = self.de1net(x)
        de2 = self.de2net(x)
        return de1, de2


class M0(nn.Module):
    def __init__(self):
        super(M0, self).__init__()
        self.backbone = models.vgg16(pretrained=True).features

    def forward(self, x):
        d = self.backbone(x)
        return d

def deconvolutuion():
    transpose = nn.ConvTranspose2d(3, 1, 1, 2, output_padding=1)
    in_tensor = torch.rand((1, 3, 40, 40))
    out = transpose(in_tensor)
    print(out.size())


if __name__ == "__main__":
    deconvolutuion()
    # m0 = M0()
    # m1 = M1()
    #
    # m0_param = list(m0.parameters())
    # m1_param = list(m1.parameters())
    #
    # print(len(m0_param))
    # print(len(m1_param))