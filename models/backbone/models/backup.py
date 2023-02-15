import math
from torch import nn
import torch
import torch.nn.functional as F
from .backbones.resnet.regnet import RegNetX_200MF
from .backbones.resnet.resnet import ResNet101, ResNet50
from .bn_helper import BatchNorm2d, relu_inplace
from .fusion import AFF, iAFF
from .seg_hrnet import get_seg_model
from ..config import config
ALIGN_CORNERS = True

class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return BatchNorm2d

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


class Multi_Context(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,  size, stride=1, downsample=None, baseWidth=26, scale=12, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Multi_Context, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))

        # init
        self.conv = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        # Encoder
        self.ms_encoder = MS_Coder(inplanes, width,  scale, size, k_size=3, stride=stride)
        self.aff = iAFF(channels=width, r=4)
        # decoder
        self.ms_decoder = MS_Coder(inplanes, width,  scale, size, k_size=3, stride=stride)

        self.conv3 = nn.Conv2d(width * scale, planes, kernel_size=3, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width
        # print("Conv3->", width * scale)

    def forward(self, x):
        out = self.conv(x)
        enc_list = self.ms_encoder(x)
        dec_list = self.ms_encoder(x)

        value = len(enc_list)-1
        while value >= 0:
            if value == len(enc_list)-1:
                enc = enc_list[value]
                dec = dec_list[value]

                out = self.aff(enc, dec)
            else:
                enc = enc_list[value]
                dec = dec_list[value]
                out_sp = self.aff(enc, dec)

                out = torch.cat((out, out_sp), 1)
            # print(enc.shape, dec.shape, out.shape)
            value = value - 1

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out

class MS_Coder(nn.Module):
    def __init__(self, inplanes, width, scale, size, k_size=3, stride=1, stype='normal'):
        super(MS_Coder, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.relu = nn.ReLU(inplace=True)

        self.cf = Context_Flow(width, width)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=k_size, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # First tier of the multi-scale  and the Encoder section of the hidden U-Net
        spx = torch.split(out, self.width, 1)
        coder = []
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]

            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))

            if i == 0:
                out = self.cf(sp)
            else:
                sp = self.cf(sp)
                out = sp
                # out = torch.cat((out, sp), 1)
            coder.append(out)
        return coder


class OHybridCR(nn.Module):
    def __init__(self, ocr_key_channels, num_classes, backbone="hrnet"):
        super(OHybridCR, self).__init__()

        # Backbones
        self.backbone = backbone
        if backbone == "resnet_101":
            self.resnet = ResNet101()
            size = 64
        elif backbone == "resnet_50":
            self.resnet = ResNet50()
            size = 64
        elif backbone == "regnet":
            self.regnet = RegNetX_200MF()
            size = 128
        elif backbone == "hrnet":
            self.hrnet = get_seg_model(config)
            # print("PP- Janneh")
            size = 32

        self.cls_head = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # this is the multi-scale section
        self.multi_context = Multi_Context(ocr_key_channels, ocr_key_channels, size=size)

        self.aux_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(256),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(256, num_classes,  kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        out_aux_seg = []
        if self.backbone == "resnet_50":
            feats = self.resnet(x)
        elif self.backbone == "resnet_101":
            feats = self.resnet(x)
        elif self.backbone == "regnet":
            feats = self.regnet(x)
        elif self.backbone == "hrnet":
            feats = self.hrnet(x)

        out_aux = self.aux_head(feats)  # backbone

        out = self.multi_context(feats)  # Dual Re-sample feature multi-scale
        out = self.cls_head(out)

        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)
        return out_aux_seg

