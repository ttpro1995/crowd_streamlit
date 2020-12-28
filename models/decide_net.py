import torch.nn as nn
import torch
from torchvision import models
from torchvision.models.detection import FasterRCNN
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator

def very_simple_param_count(model):
    result = sum([p.numel() for p in model.parameters()])
    return result

class DecideNet(nn.Module):
    """
    Purpose this model is to count number of parameters in DecideNet
    (https://arxiv.org/pdf/1712.06679.pdf)
    It does not implement decide net correctly
    Just 'nearly' correct to estimate number of parameters
    """
    def __init__(self, load_weights=False):
        super(DecideNet, self).__init__()
        self.seen = 0
        self.reg_net_feat = [(7, 20, 1), (5, 40, 1), (5, 20, 1), (5, 10, 1), (1, 1, 1)]
        self.reg_net = make_layers(self.reg_net_feat, 3)
        self.quality_net_feat = [(7, 20, 1), (5, 40, 1), (5, 20, 1), (1, 1, 1)]
        self.quality_net = make_layers(self.quality_net_feat, 3)
        # backbone = torchvision.models.mobilenet_v2(pretrained=False).features
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet101','False')
        # model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
        # backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                        output_size=7,
                                                        sampling_ratio=2)
        self.fastercnn = FasterRCNN(backbone,
                           num_classes=2,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)



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

if __name__ == "__main__":
    net = DecideNet()
    n_param = very_simple_param_count(net)
    print(n_param)
    resnet101 = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet101','False')
    print(very_simple_param_count(resnet101))