import torch.nn as nn
import torch
import torch.nn.functional as F

from torchvision import models
import numpy as np
import copy

# ssim lost function


class PACNN(nn.Module):
    def __init__(self):
        super(PACNN, self).__init__()
        self.backbone = models.vgg16(pretrained=True).features
        self.de1net = self.backbone[0:23]
        self.de1_11 = nn.Conv2d(512, 1, kernel_size=1)
        self.de2net = self.backbone[0:30]
        self.de2_11 = nn.Conv2d(512, 1, kernel_size=1)

        list_vgg16 = list(self.backbone)
        conv6_1_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        list_vgg16.append(conv6_1_1)
        self.de3net = nn.Sequential(*list_vgg16)
        self.de3_11 = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        de1 = self.de1_11((self.de1net(x)))
        de2 = self.de2_11((self.de2net(x)))
        de3 = self.de3_11((self.de3net(x)))
        return de1.squeeze(0), de2.squeeze(0), de3.squeeze(0)


class PACNNWithPerspectiveMap(nn.Module):
    def __init__(self, perspective_aware_mode=False):
        super(PACNNWithPerspectiveMap, self).__init__()
        self.backbone = models.vgg16(pretrained=True).features
        self.de1net = self.backbone[0:23]

        self.de2net = self.backbone[0:30]


        list_vgg16 = list(self.backbone)
        self.conv6_1_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        list_vgg16.append(self.conv6_1_1)
        self.de3net = nn.Sequential(*list_vgg16)


        self.conv5_2_3_stack = copy.deepcopy(self.backbone[23:30])
        self.perspective_net = nn.Sequential(self.backbone[0:23], self.conv5_2_3_stack)


        # 1 1 convolution
        self.de1_11 = nn.Conv2d(512, 1, kernel_size=1)
        self.de2_11 = nn.Conv2d(512, 1, kernel_size=1)
        self.de3_11 = nn.Conv2d(512, 1, kernel_size=1)
        self.perspective_11 = nn.Conv2d(512, 1, kernel_size=1)

        # deconvolution upsampling
        self.up12 = nn.ConvTranspose2d(1, 1, 2, 2)
        self.up23 = nn.ConvTranspose2d(1, 1, 2, 2)
        self.up_perspective = nn.ConvTranspose2d(1, 1, 2, 2)

        # if true, use perspective aware
        # if false, use average
        self.perspective_aware_mode = perspective_aware_mode

    def forward(self, x):
        de1 = self.de1_11((self.de1net(x)))
        de2 = self.de2_11((self.de2net(x)))
        de3 = self.de3_11((self.de3net(x)))
        if self.perspective_aware_mode:
            pespective_w_s = self.perspective_11(self.perspective_net(x))
            pespective_w = self.up_perspective(pespective_w_s)

            upde3 = self.up23(de3)
            pad_3_0 = de2.size()[2] - upde3.size()[2]
            pad_3_1 = de2.size()[3] - upde3.size()[3]
            upde3pad = F.pad(upde3, (0, pad_3_1, 0, pad_3_0), value=0)
            de23_a = pespective_w_s * de2
            de23_b = (1 - pespective_w_s)*(de2 + upde3pad)
            de23 = de23_a + de23_b

            upde23 = self.up12(de23)
            pad_23_0 = de1.size()[2] - upde23.size()[2]
            pad_23_1 = de1.size()[3] - upde23.size()[3]
            upde23pad = F.pad(upde23, (0, pad_23_1, 0, pad_23_0), value=0)

            pad_perspective_0 = de1.size()[2] - pespective_w.size()[2]
            pad_perspective_1 = de1.size()[3] - pespective_w.size()[3]
            pespective_w_pad = F.pad(pespective_w, (0, pad_perspective_1, 0, pad_perspective_0), value=0)
            de_a = pespective_w_pad * de1
            de_b = (1 - pespective_w_pad)*(de1 + upde23pad)
            de = de_a + de_b
        else:
            #try:
            pespective_w_s = None
            pespective_w = None
            upde3 = self.up23(de3)
            pad_3_0 = de2.size()[2] - upde3.size()[2]
            pad_3_1 = de2.size()[3] - upde3.size()[3]
            upde3pad = F.pad(upde3,(0, pad_3_1, 0, pad_3_0), value=0)
            de23 = (de2 + upde3pad)/2

            upde23 = self.up12(de23)
            pad_23_0 = de1.size()[2] - upde23.size()[2]
            pad_23_1 = de1.size()[3] - upde23.size()[3]
            upde23pad = F.pad(upde23, (0, pad_23_1, 0, pad_23_0), value=0)

            de = (de1 + upde23pad)/2
            # except Exception as e:
            #     print("EXECEPTION ", e)
            #     print(x.size())
            #     print(de2.size(), de3.size())
        return de1, de2, de3, pespective_w_s, pespective_w, de

def count_param(net):
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    return pytorch_total_params

def parameter_count_test():
    net = PACNN()
    total_real = count_param(net)
    print("total real ", total_real)
    backbone = count_param(net.backbone)
    conv611 = count_param(net.conv6_1_1)
    de1_11 = count_param(net.de1_11)
    de2_11 = count_param(net.de2_11)
    de3_11 = count_param(net.de3_11)
    sum_of_part = backbone + de1_11 + de2_11 + de3_11 + conv611
    print(sum_of_part)

if __name__ == "__main__":
    # parameter_count_test()
    net = PACNN()
    print(net.de1net)
    # img = torch.rand(1, 3, 320, 320)
    # de1, de2, de3 = net(img)
    # print(de1.size())
    # print(de2.size())
    # print(de3.size())