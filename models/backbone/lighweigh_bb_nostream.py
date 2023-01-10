import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from models.nets.utils_nostream import IntraScaled, AMscaling, SFCM, EGLF, PPM


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
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        # out = self.ea(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_IntraScale(nn.Module):
    def __init__(self, block, num_blocks, mstream=True, std_out=32):
        super(ResNet_IntraScale, self, ).__init__()
        self.in_planes = 16
        self.mstream = mstream
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        basewidth, scale = 26, 4
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)  #
        self.skip_wei1 = IntraScaled(16, basewidth, scale, std_out)

        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.skip_wei2 = IntraScaled(32, basewidth, scale, std_out)

        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.skip_wei3 = IntraScaled(64, basewidth, scale, std_out)

        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.skip_wei4 = IntraScaled(128, basewidth, scale, std_out)

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


class LResNet_SFCM(nn.Module):
    def __init__(self, block, num_blocks, mstream=True, std_out=32, is_add=False):
        super(LResNet_SFCM, self, ).__init__()
        self.in_planes = 16
        self.mstream = mstream
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)  #
        self.skip_wei1 = SFCM(16, std_out, is_add)

        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.skip_wei2 = SFCM(32, std_out, is_add)

        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.skip_wei3 = SFCM(64, std_out, is_add)

        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.skip_wei4 = SFCM(128, std_out, is_add)

        self.cnn = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])
        self.skip_weis = nn.ModuleList([self.skip_wei1, self.skip_wei2, self.skip_wei3, self.skip_wei4])
        self.last = nn.Conv2d(128 * block.expansion, 64, kernel_size=1, stride=1, bias=False)
        self.bn1l = nn.BatchNorm2d(64)
        # dilation = [3, 5, 7, 9]
        self.skip_wei_R2net = IntraScaled(128, 26, 4, std_out)

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

        ms_target, _ = self.skip_wei_R2net(out)
        out = F.relu(self.bn1l(self.last(out)))

        out_layer4 = ms_target
        return out_layer4, out, weigted_fusion


class PdLResNet_SFCM(nn.Module):
    def __init__(self, block, num_blocks, mstream=True, std_out=32, is_add=False):
        super(PdLResNet_SFCM, self, ).__init__()
        self.in_planes = 16
        self.mstream = mstream
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)  #
        self.skip_wei1 = nn.Conv2d(16, std_out, kernel_size=3, stride=1, padding=1, bias=False)  # SFCM(16, std_out, is_add)

        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.skip_wei2 = nn.Conv2d(32, std_out, kernel_size=3, stride=1, padding=1, bias=False)  # SFCM(32, std_out, is_add)

        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.skip_wei3 = nn.Conv2d(64, std_out, kernel_size=3, stride=1, padding=1, bias=False)  # SFCM(64, std_out, is_add)

        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.skip_wei4 = nn.Conv2d(128, std_out, kernel_size=3, stride=1, padding=1, bias=False)  # SFCM(128, std_out, is_add)

        self.cnn = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])
        self.skip_weis = nn.ModuleList([self.skip_wei1, self.skip_wei2, self.skip_wei3, self.skip_wei4])
        self.last = nn.Conv2d(128 * block.expansion, 64, kernel_size=1, stride=1, bias=False)
        self.bn1l = nn.BatchNorm2d(64)
        # dilation = [3, 5, 7, 9]
        # self.skip_wei_R2net = IntraScaled(128, 26, 4, std_out)
        self.conv = nn.Conv2d(std_out, std_out, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(std_out)
        self.relu = nn.ReLU(inplace=True)

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
            target = F.interpolate(target, x.shape[2:], mode='bilinear', align_corners=True)
            out = out + untarget
            weigted_fusion.append(self.relu(self.bn(self.conv(target))))
            i = i + 1

        out_layer4 = out
        out = F.relu(self.bn1l(self.last(out)))
        return out_layer4, out, weigted_fusion


class PLResNet_SFCM(nn.Module):
    def __init__(self, block, num_blocks, mstream=True, std_out=32, is_add=False):
        super(PLResNet_SFCM, self, ).__init__()
        self.in_planes = 16
        self.mstream = mstream
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)  #
        self.skip_wei1 = nn.Conv2d(16, std_out, kernel_size=3, stride=1)  # SFCM(16, std_out, is_add)

        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.skip_wei2 = nn.Conv2d(32, std_out, kernel_size=3, stride=1)  # SFCM(32, std_out, is_add)

        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.skip_wei3 = nn.Conv2d(64, std_out, kernel_size=3, stride=1)  # SFCM(64, std_out, is_add)

        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.skip_wei4 = nn.Conv2d(128, std_out, kernel_size=3, stride=1)  # SFCM(128, std_out, is_add)

        self.cnn = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])
        self.skip_weis = nn.ModuleList([self.skip_wei1, self.skip_wei2, self.skip_wei3, self.skip_wei4])
        self.last = nn.Conv2d(128 * block.expansion, 64, kernel_size=1, stride=1, bias=False)
        self.bn1l = nn.BatchNorm2d(64)
        # dilation = [3, 5, 7, 9]
        # self.skip_wei_R2net = IntraScaled(128, 26, 4, std_out)
        self.conv = nn.Conv2d(std_out, std_out, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(std_out)
        self.relu = nn.ReLU(inplace=True)

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
        # x = self.cnn[0](out)
        weigted_fusion = []
        i = 0
        for layer in self.cnn:
            out = layer(out)
            target = self.skip_weis[i](out)
            # print("Janneh-->", target.shape)
            target = F.interpolate(target, x.shape[2:], mode='bilinear', align_corners=True)
            out = out  # + (1-target)
            weigted_fusion.append(self.relu(self.bn(self.conv(target))))
            i = i + 1

        out_layer4 = out
        out = F.relu(self.bn1l(self.last(out)))
        weigted_fusion = torch.cat(weigted_fusion, dim=1)
        return out_layer4, out, weigted_fusion


class ResNet_SFCM(nn.Module):
    def __init__(self, block, num_blocks, mstream=True, std_out=32, is_add=False, expansion=1):
        super(ResNet_SFCM, self, ).__init__()
        self.in_planes = 64
        self.mstream = mstream
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  #
        self.skip_wei1 = SFCM(64 * expansion, std_out, is_add)

        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.skip_wei2 = SFCM(128 * expansion, std_out, is_add)

        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.skip_wei3 = SFCM(256 * expansion, std_out, is_add)

        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.skip_wei4 = SFCM(512 * expansion, std_out, is_add)

        # self.skip_wei_R2net = IntraScaled(expansion*512, 26, 4, std_out)

        self.cnn = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])
        self.skip_weis = nn.ModuleList([self.skip_wei1, self.skip_wei2, self.skip_wei3, self.skip_wei4])
        self.last = nn.Conv2d(512 * block.expansion, 128, kernel_size=1, stride=1, bias=False)
        self.bn1l = nn.BatchNorm2d(128)

        # self.ea = EfficientAttentions(std_out, std_out, 8, std_out)

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


class ResNet_EGLF(nn.Module):
    def __init__(self, block, num_blocks, bins=(1, 2, 3, 6), mstream=True, std_out=32, is_add=False, use_ppm=False):
        super(ResNet_EGLF, self, ).__init__()
        self.in_planes = 16
        self.use_ppm = use_ppm
        self.mstream = mstream
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)  #
        self.skip_wei1 = EGLF(16, std_out, is_add)

        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.skip_wei2 = EGLF(32, std_out, is_add)

        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.skip_wei3 = EGLF(64, std_out, is_add)

        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.skip_wei4 = EGLF(128, std_out, is_add)

        self.cnn = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])
        self.skip_weis = nn.ModuleList([self.skip_wei1, self.skip_wei2, self.skip_wei3, self.skip_wei4])
        self.last = nn.Conv2d(128 * block.expansion, 64, kernel_size=1, stride=1, bias=False)
        self.bn1l = nn.BatchNorm2d(64)

        fea_dim = 128
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins)
            fea_dim /= 4

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

        ppm = out
        if self.use_ppm:
            ppm = self.ppm(out)
        out = F.relu(self.bn1l(self.last(out)))
        return ppm, out, weigted_fusion


# def LResNet18IS(mstream=True, std_out=32):
#     return ResNet_IntraScale(BasicBlock, [2, 2, 2, 2], mstream=mstream, std_out=std_out)
#
# def LResNet18AMS(mstream=True, std_out=32):
#     return ResNet_AMScale(BasicBlock, [2, 2, 2, 2], mstream=mstream, std_out=std_out)
#
# def LResNet18SFCM(mstream=True, std_out=32):
#     return ResNet_SFCM(BasicBlock, [2, 2, 2, 2], mstream=mstream, std_out=std_out)
#
# def LResNet18EGLF(mstream=True, std_out=32):
#     return ResNet_SFCM(BasicBlock, [2, 2, 2, 2], mstream=mstream, std_out=std_out)

# ============================= another ============================

# def LResNet34IS(mstream=True, std_out=32):
#     return ResNet_IntraScale(BasicBlock, [3, 4, 6, 3], mstream=mstream, std_out=std_out)
#
# def LResNet34AMS(mstream=True, std_out=32):
#     return ResNet_AMScale(BasicBlock, [3, 4, 6, 3], mstream=mstream, std_out=std_out)

def PLResNet34FRWM(mstream=True, std_out=32):
    return PLResNet_SFCM(BasicBlock, [3, 4, 6, 3], mstream=mstream, std_out=std_out)


def PdLResNet34FRWM(mstream=True, std_out=32):  # our CWFF - decoder
    return PdLResNet_SFCM(BasicBlock, [3, 4, 6, 3], mstream=mstream, std_out=std_out)


def LResNet34FRWM(mstream=True, std_out=32):
    return LResNet_SFCM(BasicBlock, [3, 4, 6, 3], mstream=mstream, std_out=std_out)


def ResNet18FRWM(mstream=True, std_out=256):
    return ResNet_SFCM(BasicBlock, [3, 4, 6, 3], mstream=mstream, std_out=std_out)


def ResNet34FRWM(mstream=True, std_out=256):
    return ResNet_SFCM(BasicBlock, [3, 4, 6, 3], mstream=mstream, std_out=std_out)


def ResNet50FRWM(mstream=True, std_out=256):
    return ResNet_SFCM(Bottleneck, [3, 4, 6, 3], mstream=mstream, std_out=std_out, expansion=4)


def test():
    from pytorch_memlab import MemReporter
    net = PLResNet34FRWM().cuda()
    ppm, out, y = net(torch.randn(1, 3, 32, 32).cuda())
    # summary(net)
    # print(out.shape, ppm.shape)
    reporter = MemReporter()
    reporter.report()

# test()
