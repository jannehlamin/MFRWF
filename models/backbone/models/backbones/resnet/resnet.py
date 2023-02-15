'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from Models.models.Quantifiers import Attention, BBlock
from Models.models.conformer import EfficientAttentions


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
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.ea = EfficientAttentions(planes, planes, 8, planes) # BBlock(planes, 8)  # EfficientAttentions(dim, dim, num_heads, dim)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        # out = self.ea(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.tranCnn = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])
        self.last = nn.Conv2d(512*block.expansion, 128, kernel_size=1, stride=1, bias=False)
        self.bn1l = nn.BatchNorm2d(128)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        tranCnn = []
        for layer in self.tranCnn:
            out = layer(out)
            tranCnn.append(out)
        out = F.relu(self.bn1l(self.last(out)))

        # out = F.interpolate(out, size=(128, 128), mode='bilinear', align_corners=True)
        return x, out, tranCnn

# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks):
#         super(ResNet, self).__init__()
#         self.in_planes = 64
#         embed_dim = 48
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.last = nn.Conv2d(512 * block.expansion, 128, kernel_size=1, stride=1, bias=False)
#         self.bn1l = nn.BatchNorm2d(128)
#         self.layers = [layer1, layer2, layer3, layer4]
#         # transformer
#         expansion = 4
#         self.match = 4
#         isUp = False
#         in_ch = 64 * expansion  # resnet-50 and above
#         out_ch = in_ch * 2  # global_sample[0] not part of the
#
#         self.initial_map = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False)  # trans initial map
#         if self.match == 1: isUp = True
#         global_sample = [GLIntegration(in_ch, out_ch, res_conv=True, isFirst=True, isUp=False, embed_dim=embed_dim),
#                          GLIntegration(out_ch, out_ch*2, res_conv=True, isFirst=False, isUp=isUp, embed_dim=embed_dim)]
#         self.tran_last = nn.Conv2d(out_ch * 4, 128, kernel_size=1, stride=1, bias=False)
#         for i in range(2, self.match):
#             in_ch, out_ch = out_ch, out_ch * 2
#             if i == self.match-1:
#                 isUp = True
#             global_sample.append(
#                 GLIntegration(in_ch, out_ch, res_conv=True, isFirst=False, isUp=isUp, embed_dim=embed_dim))
#
#         self.globals = nn.ModuleList(global_sample)
#
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#
#         length = len(self.layers)
#         step = 0
#         _, _, H, W = out.shape
#         for i, layer in enumerate(self.layers):
#             out = layer(out)
#
#             if self.match >= (length - i):
#                 out = F.interpolate(out, size=(H//4, W//4), mode='bilinear', align_corners=True)
#
#                 if self.match == (length - i):  # initial position
#                     init_out = self.initial_map(out)  # out 256
#                     initial_map = self.globals[step](out)  # trans 1 out 256
#                     step = step + 1
#                     tran_out = self.globals[step](init_out, initial_map)
#                     # print("Lamin->", i, step, out.shape, tran_out.shape, initial_map.shape, (H, W))
#                 else:
#                     # out = F.interpolate(out, size=(H // 2, W // 2), mode='bilinear', align_corners=True)
#                     step = step + 1
#                     if step < self.match:
#                         tran_out = self.globals[step](out, tran_out)
#
#         trans_out = self.tran_last(tran_out)
#         out = F.relu(self.bn1l(self.last(out)))
#         out = torch.concat((out, trans_out), dim=1)
#         return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


# def test():
#     net = ResNet34().cuda()
#     out, y = net(torch.randn(1, 3, 32, 32).cuda())
#     summary(net)
#     print(out.shape)
#
# test()
