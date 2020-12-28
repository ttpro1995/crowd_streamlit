import torch.nn as nn
import torch
from torchvision import models
from torchvision.models.detection import FasterRCNN
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator

def very_simple_param_count(model):
    result = sum([p.numel() for p in model.parameters()])
    return result

class PaDNet(nn.Module):
    """
    Purpose this model is to count number of parameters in PaDNet
    (https://deepai.org/publication/padnet-pan-density-crowd-counting)
    It does not implement correctly
    Just 'nearly' correct to estimate number of parameters
    """
    def __init__(self, load_weights=False):
        super(PaDNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers_vgg(self.frontend_feat)
        self.dan1_feat = [(9, 384, 1), (9, 256, 1), (7, 128, 1), (5, 64, 1), (3, 1, 1)]
        self.dan2_feat = [(7, 256, 1), (7, 128, 1), (5, 64, 1), (5, 32, 1), (3, 1, 1)]
        self.dan3_feat = [(5, 128, 1), (5, 64, 1), (5, 32, 1), (5, 16, 1), (3, 1, 1)]
        self.dan4_feat = [(5, 128, 1), (5, 64, 1), (3, 32, 1), (3, 16, 1), (3, 1, 1)]
        self.tail_feat = [(7, 64, 1), (5, 32, 1), (3, 32, 1), (3, 1, 1)]

        self.dan1 = make_layers(self.dan1_feat)
        self.dan2 = make_layers(self.dan2_feat)
        self.dan3 = make_layers(self.dan3_feat)
        self.dan4 = make_layers(self.dan4_feat)
        self.tail = make_layers(self.tail_feat)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)

        # remove channel dimension
        # (N, C_{out}, H_{out}, W_{out}) => (N, H_{out}, W_{out})
        x = torch.squeeze(x, dim=1)
        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.normal_(m.weight, std=0.01)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v is 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            kernel_size = v[0]
            n_filter = v[1]
            dilated_rate = v[2]
            conv2d = nn.Conv2d(in_channels, n_filter, kernel_size=kernel_size, padding=dilated_rate, dilation=dilated_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = n_filter
    return nn.Sequential(*layers)

def make_layers_vgg(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)

if __name__ == "__main__":
    net = PaDNet()
    n_param = very_simple_param_count(net)
    print(n_param)