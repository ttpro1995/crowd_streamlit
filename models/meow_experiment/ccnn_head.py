import torch.nn as nn
import torch
import collections
import torch.nn.functional as F


class H1(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    the improve version

    we change 5x5 7x7 9x9 with 3x3
    Keep the tail
    """
    def __init__(self, load_weights=False):
        super(H1, self).__init__()
        self.model_note = "We replace 5x5 7x7 9x9 with 3x3, no batchnorm yet, keep tail, no dilated"
        # self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        # self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        # self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)

        # ideal from crowd counting using DMCNN
        self.front_cnn_1 = nn.Conv2d(3, 10, 3, padding=1)
        self.front_cnn_2 = nn.Conv2d(10, 20, 3, padding=1)
        self.front_cnn_3 = nn.Conv2d(20, 20, 3, padding=1)
        self.front_cnn_4 = nn.Conv2d(20, 20, 3, padding=1)

        self.c0 = nn.Conv2d(60, 40, 3, padding=1)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(40, 60, 3, padding=1)

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

        # x = self.max_pooling(x)
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


class H2(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    the improve version

    we change 5x5 7x7 9x9 with 3x3
    Keep the tail
    """
    def __init__(self, load_weights=False):
        super(H2, self).__init__()
        self.model_note = "We replace 5x5 7x7 9x9 with 3x3, no batchnorm yet, keep tail, no dilated. " \
                          "Front_cnn_1 change to 20. ALl front is 20"
        # self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        # self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        # self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)

        # ideal from crowd counting using DMCNN
        self.front_cnn_1 = nn.Conv2d(3, 20, 3, padding=1)
        self.front_cnn_2 = nn.Conv2d(20, 20, 3, padding=1)
        self.front_cnn_3 = nn.Conv2d(20, 20, 3, padding=1)
        self.front_cnn_4 = nn.Conv2d(20, 20, 3, padding=1)

        self.c0 = nn.Conv2d(60, 40, 3, padding=1)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(40, 60, 3, padding=1)

        # ideal from CSRNet
        self.c2 = nn.Conv2d(60, 40, 3, padding=1)
        self.c3 = nn.Conv2d(40, 20, 3, padding=1)
        self.c4 = nn.Conv2d(20, 10, 3, padding=1)
        self.output = nn.Conv2d(10, 1, 1)

    def forward(self,x):
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

        x = self.output(x)
        return x


class H3(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    """
    def __init__(self, load_weights=False):
        super(H3, self).__init__()
        self.model_note = "CCNNv7, add branch 3x3"
        self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)
        self.black_cnn = nn.Conv2d(3, 20, 3, padding=1)
        self.c0 = nn.Conv2d(60, 60, 3, padding=1)

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
        x_black = F.relu(self.black_cnn(x), inplace=True)

        x = torch.cat((x_red, x_green, x_blue, x_black), 1)

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


class H3i(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    no inplace
    https://towardsdatascience.com/in-place-operations-in-pytorch-f91d493e970e
    """
    def __init__(self, load_weights=False):
        super(H3i, self).__init__()
        self.model_note = "CCNNv7, add branch 3x3, no inplace Relu"
        self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)
        self.black_cnn = nn.Conv2d(3, 20, 3, padding=1)
        self.c0 = nn.Conv2d(60, 60, 3, padding=1)

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(60, 60, 3, padding=1)
        self.c2 = nn.Conv2d(60, 40, 3, padding=1)
        self.c3 = nn.Conv2d(40, 20, 3, padding=1)
        self.c4 = nn.Conv2d(20, 10, 3, padding=1)
        self.output = nn.Conv2d(10, 1, 1)

    def forward(self,x):
        x_red = F.relu(self.red_cnn(x))
        x_green = F.relu(self.green_cnn(x))
        x_blue = F.relu(self.blue_cnn(x))
        x_black = F.relu(self.black_cnn(x))

        x = torch.cat((x_red, x_green, x_blue, x_black), 1)

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


class H4i(nn.Module):
    """
    A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
    https://arxiv.org/pdf/2002.06515.pdf
    the improve version

    we change 5x5 7x7 9x9 with 3x3
    Keep the tail
    """
    def __init__(self, load_weights=False):
        super(H4i, self).__init__()
        self.model_note = "From H2, fix bug redundancy forward and remove inplace" \
                          " We replace 5x5 7x7 9x9 with 3x3, no batchnorm yet, keep tail, no dilated. " \
                          "Front_cnn_1 change to 20. ALl front is 20"

        # ideal from crowd counting using DMCNN
        self.front_cnn_1 = nn.Conv2d(3, 20, 3, padding=1)
        self.front_cnn_2 = nn.Conv2d(20, 20, 3, padding=1)
        self.front_cnn_3 = nn.Conv2d(20, 20, 3, padding=1)
        self.front_cnn_4 = nn.Conv2d(20, 20, 3, padding=1)

        self.c0 = nn.Conv2d(60, 40, 3, padding=1)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c1 = nn.Conv2d(40, 60, 3, padding=1)

        # ideal from CSRNet
        self.c2 = nn.Conv2d(60, 40, 3, padding=1)
        self.c3 = nn.Conv2d(40, 20, 3, padding=1)
        self.c4 = nn.Conv2d(20, 10, 3, padding=1)
        self.output = nn.Conv2d(10, 1, 1)

    def forward(self,x):

        x1 = F.relu(self.front_cnn_1(x))
        x2 = F.relu(self.front_cnn_2(x1))
        x3 = F.relu(self.front_cnn_3(x2))
        x4 = F.relu(self.front_cnn_4(x3))

        x_red = self.max_pooling(x4)
        x_green = self.max_pooling(x3)
        x_blue = self.max_pooling(x2)

        # x = self.max_pooling(x)
        x = torch.cat((x_red, x_green, x_blue), 1)
        x = F.relu(self.c0(x))

        x = F.relu(self.c1(x))

        x = F.relu(self.c2(x))
        x = self.max_pooling(x)

        x = F.relu(self.c3(x))
        x = self.max_pooling(x)

        x = F.relu(self.c4(x))
        x = self.output(x)
        return x
