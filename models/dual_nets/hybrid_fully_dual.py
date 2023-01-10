import math
from torch import nn
import torch
import torch.nn.functional as F
from ..backbone.lighweigh_bb import LResNet34SFCM
from ..backbone.resnet import ResNet18, ResNet50, ResNet34
from ..dual_nets.utils import E_GFF, ConvWFusion, MSFCM_GFF

ALIGN_CORNERS = True


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class General_Flow(nn.Module):
    def __init__(self, out_ch, size):
        super(General_Flow, self).__init__()
        self.g_avg = nn.AdaptiveAvgPool2d((size))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.g_avg(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Context_Flow(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(Context_Flow, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation, groups=inplanes, bias=bias)
        self.bn = nn.BatchNorm2d(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        x = self.relu(x)
        return x

class MLevel_Funsion(nn.Module):
    def __init__(self, channel=32):
        super(MLevel_Funsion, self).__init__()
        # enhance fully gated fusion - encoder
        self.layer1 = E_GFF(16, [32, 64, 128])
        self.layer2 = E_GFF(32, [16, 64, 128])
        self.layer3 = E_GFF(64, [16, 32, 128])
        self.layer4 = E_GFF(128, [16, 32, 64])
        self.E_GFF = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])

        # convolution weighted feature fusion - decoder
        self.conwf4 = ConvWFusion(channel)
        self.conwf3 = ConvWFusion(channel)
        self.conwf2 = ConvWFusion(channel)
        self.conwf1 = ConvWFusion(channel)

        self.convwfs = nn.ModuleList([self.conwf4, self.conwf3, self.conwf2, self.conwf1])

    def forward(self, layers, mask):
        # gated feature fusion
        r_layers = layers.copy()
        i = 0
        egff = []
        for layer_out in layers:
            r_layers.pop(i)
            # print(len(r_layers))
            gff_out = self.E_GFF[i](layer_out, r_layers, mask)
            egff.append(gff_out)  # stack of the all fully gated fusion
            r_layers = layers.copy()
            i = i + 1

        egff.reverse()  # bottom up operation as decoder
        x = egff[0]
        for j in range(1, len(egff)):
            y = egff[j]
            out = self.convwfs[j-1](x, y)
            x = y

        return out

class MLevel_Funsion_R(nn.Module):
    def __init__(self, backbone, channel=256):
        super(MLevel_Funsion_R, self).__init__()
        # enhance fully gated fusion - encoder
        expansion = 1
        if backbone == "r50":
            expansion = 4
        self.layer1 = E_GFF(expansion*64, [expansion*128, expansion*256, expansion*512], std_out=256)
        self.layer2 = E_GFF(expansion*128, [expansion*64, expansion*256, expansion*512], std_out=256)
        self.layer3 = E_GFF(expansion*256, [expansion*64, expansion*128, expansion*512], std_out=256)
        self.layer4 = E_GFF(expansion*512, [expansion*64, expansion*128, expansion*256], std_out=256)
        self.E_GFF = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])

        # convolution weighted feature fusion - decoder
        self.conwf4 = ConvWFusion(channel)
        self.conwf3 = ConvWFusion(channel)
        self.conwf2 = ConvWFusion(channel)
        self.conwf1 = ConvWFusion(channel)

        self.convwfs = nn.ModuleList([self.conwf4, self.conwf3, self.conwf2, self.conwf1])

    def forward(self, layers, mask):
        # gated feature fusion
        r_layers = layers.copy()
        i = 0
        egff = []
        for layer_out in layers:
            r_layers.pop(i)
            # print(len(r_layers))
            gff_out = self.E_GFF[i](layer_out, r_layers, mask)
            egff.append(gff_out)  # stack of the all fully gated fusion
            r_layers = layers.copy()
            # print(i, egff[-1].shape)
            i = i + 1

        egff.reverse()  # bottom up operation as decoder
        x = egff[0]
        for j in range(1, len(egff)):
            y = egff[j]
            out = self.convwfs[j-1](x, y)
            x = y

        return out


class MLD_SFCM(nn.Module):
    def __init__(self, backbone, channel=256):
        super(MLD_SFCM, self).__init__()
        # enhance fully gated fusion - encoder
        self.layer1 = MSFCM_GFF(channel, channel)
        self.layer2 = MSFCM_GFF(channel, channel)
        self.layer3 = MSFCM_GFF(channel, channel)
        self.layer4 = MSFCM_GFF(channel, channel)
        self.Mr2N_GFF = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])

        # convolution weighted feature fusion - decoder
        self.conwf4 = ConvWFusion(channel)
        self.conwf3 = ConvWFusion(channel)
        self.conwf2 = ConvWFusion(channel)
        self.conwf1 = ConvWFusion(channel)
        self.convwfs = nn.ModuleList([self.conwf4, self.conwf3, self.conwf2, self.conwf1])

    def forward(self, layers, edge):
        # gated feature fusion
        r_layers = layers.copy()
        i = 0
        MR2N_GFF = []
        for layer_out in layers:
            r_layers.pop(i)
            gff_out = self.Mr2N_GFF[i](layer_out, r_layers, edge)
            MR2N_GFF.append(gff_out)  # stack of the all fully gated fusion
            r_layers = layers.copy()
            i = i + 1

        MR2N_GFF.reverse()  # bottom up operation as decoder
        x = MR2N_GFF[0]
        for j in range(1, len(MR2N_GFF)):
            y = MR2N_GFF[j]
            out = self.convwfs[j - 1](x, y)
            x = y
        return out

class OHybridCR(nn.Module):
    def __init__(self, ocr_key_channels, num_classes, backbone="hrnet"):
        super(OHybridCR, self).__init__()

        # Backbones
        self.backbone = backbone
        # if backbone == "l1":
        #     self.resnet = LResNet18()
        if backbone == "l2":
            self.resnet = LResNet34SFCM()
        elif backbone == "r18":
            self.resnet = ResNet18()
        elif backbone == "r34":
            self.resnet = ResNet34()
        elif backbone == "r50":
            self.resnet = ResNet50()

        if backbone == "l2" or backbone == "l2":
            self.cls_head = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            # this is the multi-scale section
            self.ctrancnn = MLD_SFCM(backbone, channel=32)
            self.aux_head = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        else:
            self.cls_head = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            # this is the multi-scale section
            self.ctrancnn = MLevel_Funsion_R(self.backbone)
            self.aux_head = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, mask):

        out_aux_seg = []

        if self.backbone == "l2":
            feats, layers, edge = self.resnet(x, mask)
        elif self.backbone == "r18":
            feats, layers, edge = self.resnet(x, mask)
        elif self.backbone == "r34":
            feats, layers, edge = self.resnet(x, mask)
        elif self.backbone == "r50":
            feats, layers, edge = self.resnet(x, mask)

        out_aux = self.aux_head(feats)  # backbone
        if self.backbone == "l2" or self.backbone == "l2":
            out = self.ctrancnn(layers, edge)  # Dual Re-sample feature multi-scale
        else:
            out = self.ctrancnn(layers, edge)  # Dual Re-sample feature multi-scale

        out = self.cls_head(out)
        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)
        return out_aux_seg
