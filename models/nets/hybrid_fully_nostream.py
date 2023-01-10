from torch import nn
import torch
import torch.nn.functional as F

from .utils_nostream import DFP, DecoderFF, MR2N_GFF
from ..backbone.lighweigh_bb_nostream import LResNet34IS, LResNet34AMS, \
    LResNet34SFCM
from ..backbone.resnet import ResNet18, ResNet50, ResNet34
from ..fully_nets.utils import E_GFF, ConvWFusion
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

    def forward(self, layers):
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

        MR2N_GFF.reverse()  # bottom up operation as decoder
        x = MR2N_GFF[0]
        for j in range(1, len(MR2N_GFF)):
            y = MR2N_GFF[j]
            out = self.convwfs[j-1](x, y)
            x = y

        return out

class MDFP_MR2N(nn.Module):
    def __init__(self, channel=256):
        super(MDFP_MR2N, self).__init__()
        # enhance fully gated fusion - encoder
        self.layer1 = MR2N_GFF(channel, channel)
        self.layer2 = MR2N_GFF(channel, channel)
        self.layer3 = MR2N_GFF(channel, channel)
        self.layer4 = MR2N_GFF(channel, channel)
        self.Mr2N_GFF = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])

        # convolution weighted feature fusion - decoder
        self.decoderDFP = DFP(channel, channel)

    def forward(self, layers, x):
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

        out = self.decoderDFP(MR2N_GFF, x)

        return out
        
        

class OHybridCR(nn.Module):
    def __init__(self, num_classes, backbone="hrnet", std_out=32, isup_decoder=True):
        super(OHybridCR, self).__init__()

        self.isup_decoder = isup_decoder
        # Backbones
        self.backbone = backbone
        if backbone == "lr2net":
            self.resnet = LResNet34IS()
        elif backbone == "lams":
            self.resnet = LResNet34AMS()
        elif backbone == "lsfcm":
            self.resnet = LResNet34SFCM()

        out_channel = 32
        if isup_decoder:
            out_channel = 128
        self.cls_head = nn.Conv2d(out_channel, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # this is the multi-scale section
        # if isup_decoder:
        #     self.ctrancnn = DFP(std_out, std_out)
        # else:
        self.ctrancnn = MDFP_MR2N(channel=std_out)

        self.aux_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        # else:
        #     self.cls_head = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        #     # this is the multi-scale section
        #     self.ctrancnn = DFP(std_out, std_out)
        #     self.aux_head = nn.Sequential(
        #         nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
        #         nn.BatchNorm2d(64),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        #     )

    def forward(self, x):

        out_aux_seg = []
        if self.backbone == "lr2net":
            ppm, feats, layers = self.resnet(x)
        elif self.backbone == "lams":
            ppm, feats, layers = self.resnet(x)
        elif self.backbone == "lsfcm":
            ppm, feats, layers = self.resnet(x)

        out_aux = self.aux_head(feats)  # backbone
        # if self.isup_decoder:
        #     out = self.ctrancnn(ppm, layers, x)  # Dual Re-sample feature multi-scale
        # else:
        out = self.ctrancnn(layers, x)

        out = self.cls_head(out)
        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)
        return out_aux_seg


