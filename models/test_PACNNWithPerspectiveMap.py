from unittest import TestCase
from models.pacnn import PACNNWithPerspectiveMap
import torch

class TestPACNNWithPerspectiveMap(TestCase):

    def test_debug_avg_schema_pacnn(self):
        net = PACNNWithPerspectiveMap()
        image = torch.rand(1, 3, 330, 512)
        _, _, _, _, _, density_map = net(image)
        print(density_map.size())


    def test_avg_schema_pacnn(self):
        net = PACNNWithPerspectiveMap()
        # image
        # batch size, channel, h, w
        image = torch.rand(1, 3, 330, 512)
        _, _, _, _, _, density_map = net(image)
        print(density_map.size())
        image2 = torch.rand(1, 3, 225, 225)
        _, _, _, _, _, density_map2 = net(image2)
        print(density_map2.size())
        image3 = torch.rand(1, 3, 226, 226)
        _, _, _, _, _, density_map3 = net(image3)
        print(density_map3.size())

        image = torch.rand(1, 3, 227, 227)
        _, _, _, _, _, density_map = net(image)
        print(density_map.size())
        image2 = torch.rand(1, 3, 228, 228)
        _, _, _, _, _, density_map2 = net(image2)
        print(density_map2.size())
        image3 = torch.rand(1, 3, 229, 229)
        _, _, _, _, _, density_map3 = net(image3)
        print(density_map3.size())

    def test_perspective_aware_schema_pacnn(self):
        net = PACNNWithPerspectiveMap(perspective_aware_mode=True)
        # image
        # batch size, channel, h, w
        image = torch.rand(1, 3, 224, 224)
        _, _, _, _, _, density_map = net(image)
        print(density_map.size())

    def test_perspective_aware_schema_pacnn_pad(self):
        net = PACNNWithPerspectiveMap(perspective_aware_mode=True)
        # image
        # batch size, channel, h, w
        image = torch.rand(1, 3, 330, 512)
        _, _, _, _, _, density_map = net(image)
        print(density_map.size())
