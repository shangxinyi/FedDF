from torch import flatten
from torch.nn import Module, Linear, Conv2d, BatchNorm2d, ReLU, AdaptiveAvgPool2d, Sequential, init


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.relu = ReLU(inplace=True)
        self.down_sample = down_sample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm2d(planes * self.expansion)
        self.relu = ReLU(inplace=True)
        self.down_sample = down_sample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(Module):

    def __init__(self, block, layers, num_classes: int):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = Conv2d(3, self.in_planes, kernel_size=5, bias=False)
        self.bn1 = BatchNorm2d(self.in_planes)
        self.relu = ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 128, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        down_sample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            down_sample = Sequential(conv1x1(self.in_planes, planes * block.expansion, stride),
                                     BatchNorm2d(planes * block.expansion))
        layers = list()
        layers.append(block(self.in_planes, planes, stride, down_sample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        return Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet14(num_classes: int):
    return ResNet(BasicBlock, [2, 2, 2], num_classes)


def resnet28(num_classes: int):
    return ResNet(BasicBlock, [4, 6, 3], num_classes)


def resnet41(num_classes: int):
    return ResNet(Bottleneck, [4, 6, 3], num_classes)


def resnet92(num_classes: int):
    return ResNet(Bottleneck, [4, 23, 3], num_classes)


def resnet143(num_classes: int):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
