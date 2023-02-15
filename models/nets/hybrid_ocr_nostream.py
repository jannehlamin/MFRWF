from torch import nn
import torch
import torch.nn.functional as F
from .utils_nostream import DFP, DecoderFF, MR2N_GFF
from ..backbone.lighweigh_bb_nostream import LResNet34FRWM, PLResNet34FRWM, \
    PdLResNet34FRWM
from ..dual_nets.utils import E_GFF, ConvWFusion

ALIGN_CORNERS = True


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
            out = self.convwfs[j - 1](x, y)
            x = y
        return out


class MLevel_Funsion_R(nn.Module):
    def __init__(self, backbone, channel=256):
        super(MLevel_Funsion_R, self).__init__()
        # enhance fully gated fusion - encoder
        expansion = 1
        if backbone == "r50":
            expansion = 4
        self.layer1 = E_GFF(expansion * 64, [expansion * 128, expansion * 256, expansion * 512], std_out=256)
        self.layer2 = E_GFF(expansion * 128, [expansion * 64, expansion * 256, expansion * 512], std_out=256)
        self.layer3 = E_GFF(expansion * 256, [expansion * 64, expansion * 128, expansion * 512], std_out=256)
        self.layer4 = E_GFF(expansion * 512, [expansion * 64, expansion * 128, expansion * 256], std_out=256)
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
            gff_out = self.E_GFF[i](layer_out, r_layers, mask)
            egff.append(gff_out)  # stack of the all fully gated fusion
            r_layers = layers.copy()
            i = i + 1

        egff.reverse()  # bottom up operation as decoder
        x = egff[0]
        for j in range(1, len(egff)):
            y = egff[j]
            out = self.convwfs[j - 1](x, y)
            x = y
        return out


class MLevel_MR2N(nn.Module):
    def __init__(self, channel=256):
        super(MLevel_MR2N, self).__init__()
        # enhance fully gated fusion - encoder
        self.layer1 = MR2N_GFF(channel, channel)
        self.layer2 = MR2N_GFF(channel, channel)
        self.layer3 = MR2N_GFF(channel, channel)
        self.layer4 = MR2N_GFF(channel, channel)
        self.Mr2N_GFF = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])

        # convolution weighted feature fusion - decoder
        self.conwf4 = ConvWFusion(channel)
        self.conwf3 = ConvWFusion(channel)
        self.conwf2 = ConvWFusion(channel)
        self.conwf1 = ConvWFusion(channel)

        self.convwfs = nn.ModuleList([self.conwf4, self.conwf3, self.conwf2, self.conwf1])

    def forward(self, layers, ppm):
        # gated feature fusion
        r_layers = layers.copy()
        i = 0
        MR2N_GFF = []
        for layer_out in layers:
            r_layers.pop(i)
            # print(len(r_layers))
            gff_out = self.Mr2N_GFF[i](layer_out, r_layers)
            MR2N_GFF.append(gff_out)  # stack of the all fully gated fusion
            r_layers = layers.copy()
            # print(i, egff[-1].shape)
            i = i + 1

        # MR2N_GFF.append(ppm)  # m-scale
        MR2N_GFF.reverse()  # bottom up operation as decoder
        x = MR2N_GFF[0]
        for j in range(1, len(MR2N_GFF)):
            y = MR2N_GFF[j]
            out = self.convwfs[j - 1](x, y)
            x = y

        return out


class PMLevel_MR2N(nn.Module):
    def __init__(self, channel=256):
        super(PMLevel_MR2N, self).__init__()
        # enhance fully gated fusion - encoder
        self.layer1 = MR2N_GFF(channel, channel)
        self.layer2 = MR2N_GFF(channel, channel)
        self.layer3 = MR2N_GFF(channel, channel)
        self.layer4 = MR2N_GFF(channel, channel)
        self.Mr2N_GFF = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])

        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, layers, x):
        # gated feature fusion
        r_layers = layers.copy()
        i = 0
        MR2N_GFF = []
        for layer_out in layers:
            r_layers.pop(i)
            # print(len(r_layers))
            gff_out = self.Mr2N_GFF[i](layer_out, r_layers)
            gff_out = self.relu(
                self.bn(self.conv(F.interpolate(gff_out, x.shape[2:], mode='bilinear', align_corners=True))))
            MR2N_GFF.append(gff_out)  # stack of the all fully gated fusion
            r_layers = layers.copy()
            i = i + 1

        out = torch.cat(MR2N_GFF, dim=1)
        return out


class OHybridCR(nn.Module):
    def __init__(self, args, num_classes, backbone="hrnet", isup_decoder=False):
        super(OHybridCR, self).__init__()

        self.isup_decoder = isup_decoder
        # Backbones
        self.backbone = backbone
        if backbone == "ours_l34rw_partial_weight":
            self.resnet = PLResNet34FRWM()  # PLResNet34FRWM
            std_out = 32
        # elif backbone == "ours_l34rw_partial_cwffd":
        #     self.resnet = PdLResNet34FRWM()  # PdLResNet34FRWM
        #     std_out = 32
        elif backbone == "ours_l34rw_partial_decoder":
            self.resnet = LResNet34FRWM()
            std_out = 32
        elif backbone == "ours_l34rw_fully":
            self.resnet = LResNet34FRWM()
            std_out = 32
        # elif backbone == "ours_r18rw_fully":
        #     self.resnet = ResNet18FRWM(std_out=256)
        #     std_out = 256
        # elif backbone == "ours_r34rw_fully":
        #     self.resnet = ResNet34FRWM(std_out=256)
        #     std_out = 256
        # elif backbone == "ours_r50rw_fully":
        #     self.resnet = ResNet50FRWM(std_out=256)
        #     std_out = 256

        # out_channel = 32
        # if isup_decoder:
        #     out_channel = 128
        self.cls_head = nn.Conv2d(std_out, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        if self.backbone == "ours_l34rw_partial_weight":
            self.ctrancnn = nn.Sequential(
                nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
        elif self.backbone == "ours_l34rw_partial_decoder":
            self.cls_head = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            self.ctrancnn = PMLevel_MR2N(channel=std_out)
        else:
            self.ctrancnn = MLevel_MR2N(channel=std_out)  #

        feat_out = 128
        if self.backbone == "ours_l34rw_partial_weight" or self.backbone == "ours_l34rw_fully" or self.backbone == "baseline" \
                or self.backbone == "ours_l34rw_partial_decoder" or self.backbone == "ours_l34rw_partial_cwffd":
            feat_out = 64

        self.aux_head = nn.Sequential(
            nn.Conv2d(feat_out, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):

        out_aux_seg = []

        if self.backbone == "ours_l34rw_partial_weight":
            ppm, feats, layers = self.resnet(x)
        elif self.backbone == "ours_l34rw_partial_cwffd":
            ppm, feats, layers = self.resnet(x)
        elif self.backbone == "ours_l34rw_partial_decoder":
            ppm, feats, layers = self.resnet(x)
        elif self.backbone == "ours_l34rw_fully":
            ppm, feats, layers = self.resnet(x)
        # elif self.backbone == "ours_r18rw_fully":
        #     ppm, feats, layers = self.resnet(x)
        # elif self.backbone == "ours_r34rw_fully":
        #     ppm, feats, layers = self.resnet(x)
        # elif self.backbone == "ours_r50rw_fully":
        #     ppm, feats, layers = self.resnet(x)

        out_aux = self.aux_head(feats)  # backbone

        if self.backbone == "ours_l34rw_partial_weight":
            out = self.ctrancnn(layers)  # Dual Re-sample feature multi-scale
        else:
            out = self.ctrancnn(layers, x)

        out = self.cls_head(out)
        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)
        return out_aux_seg
