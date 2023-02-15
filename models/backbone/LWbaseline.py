import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from models.nets.utils_nostream import AMscaling


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
                #
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_AMScale(nn.Module):
    def __init__(self, block, num_blocks, mstream=True, std_out=32):
        super(ResNet_AMScale, self).__init__()
        self.in_planes = 16
        self.mstream = mstream
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        dilation = [3, 5, 7, 9]
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)  #
        self.skip_wei1 = AMscaling(16, dilation, std_out)

        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.skip_wei2 = AMscaling(32, dilation, std_out)

        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.skip_wei3 = AMscaling(64, dilation, std_out)

        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.skip_wei4 = AMscaling(128, dilation, std_out)

        self.cnn = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])
        self.skip_weis = nn.ModuleList([self.skip_wei1, self.skip_wei2, self.skip_wei3, self.skip_wei4])
        self.last = nn.Conv2d(128 * block.expansion, 64, kernel_size=1, stride=1, bias=False)
        self.bn1l = nn.BatchNorm2d(64)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # masks for edge information
        out = F.relu(self.bn1(self.conv1(x)))
        weigted_fusion = []
        i = 0
        for layer in self.cnn:
            out = layer(out)
            target, untarget = self.skip_weis[i](out)
            out = out + untarget
            weigted_fusion.append(target)
            i = i + 1
        out_layer4 = out
        out = F.relu(self.bn1l(self.last(out)))
        return out_layer4, out, weigted_fusion


class LBResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(LBResNet, self, ).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)  #

        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)

        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)

        self.cnn = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])
        self.last = nn.Conv2d(128 * block.expansion, 64, kernel_size=1, stride=1, bias=False)
        self.bn1l = nn.BatchNorm2d(64)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # masks for edge information
        out = F.relu(self.bn1(self.conv1(x)))
        weigted_fusion = []

        for layer in self.cnn:
            out = layer(out)

        out = F.relu(self.bn1l(self.last(out)))
        out = F.interpolate(out, x.shape[2:], mode='bilinear', align_corners=True)

        return out


def LWBaseline():
    return LBResNet(BasicBlock, [3, 4, 6, 3])


def test():
    net = LWBaseline().cuda()
    # out = net(torch.randn(1, 3, 32, 32).cuda())
    summary(net)


# test()
