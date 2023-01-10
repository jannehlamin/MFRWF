import math
from torch import nn
import torch
import torch.nn.functional as F

from .Quantifiers import EfficientAttentions, LGPF, ASPP_module
from .backbones.resnet.regnet import RegNetX_200MF
from .backbones.resnet.resnet_new import ResNet101, ResNet50
from .bn_helper import BatchNorm2d
from .seg_hrnet2 import get_seg_model2
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


# Object region representations
class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)  # batch x k x c
        return ocr_context


# Object contextual representations
class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''

    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))

        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )

        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )

        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )

        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)

        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)

        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])

        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)
        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


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


class MS_localContext(nn.Module):
    def __init__(self, inplanes, basewidth, scale=8, k_size=3, stride=1, stype='normal'):
        super(MS_localContext, self).__init__()
        width = int(math.floor(inplanes * (basewidth / 64.0)))

        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        self.conv3 = nn.Conv2d(width * scale, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
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
        for i in range(self.nums):
            if i == 0 or self.stype == 'normal':
                sp = spx[i]
            else:
                sp = sp + spx[i]

            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))

            if i == 0:
                out = sp
            else:
                sp = self.cf(sp)
                out = torch.cat((out, sp), 1)

        out = self.relu(self.bn3(self.conv3(out)))
        return out


class OHybridCR(nn.Module):
    def __init__(self, ocr_key_channels, num_classes, backbone="hrnet"):
        super(OHybridCR, self).__init__()

        # Backbones
        self.backbone = backbone
        if backbone == "resnet_101":
            self.resnet = ResNet101()
        elif backbone == "resnet_50":
            self.resnet = ResNet50()
        elif backbone == "regnet":
            self.regnet = RegNetX_200MF()
        elif backbone == "hrnet":
            self.hrnet = get_seg_model2(config)

        self.cls_head = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # this is the multi-scale section
        self.ms_local_context = MS_localContext(ocr_key_channels, basewidth=26, scale=4)

        self.eff_attention_l1 = EfficientAttentions(192, ocr_key_channels, 8, ocr_key_channels)
        # self.eff_attention_l2 = EfficientAttentions(384, ocr_key_channels, 8, ocr_key_channels)

        self.com_LGPF_d1 = LGPF(ocr_key_channels, ocr_key_channels, 192)
        self.com_LGPF_d2 = LGPF(ocr_key_channels, ocr_key_channels, 256)
        self.com_LGPF_out = LGPF(ocr_key_channels, ocr_key_channels, 192)

        self.atrous_out1 = ASPP_module(256, 256, dilation=12)
        self.atrous_out2 = ASPP_module(256, 256, dilation=6)
        self.atrous_out3 = ASPP_module(256, 256, dilation=3)
        self.expand = 2
        self.aux_head = nn.Sequential(
            # nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            # BatchNorm2d(256),
            # nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        out_aux_seg = []
        if self.backbone == "resnet_50":
            feats, low_feat = self.resnet(x)
        elif self.backbone == "resnet_101":
            feats, low_feat = self.resnet(x)
        elif self.backbone == "regnet":
            feats = self.regnet(x)
        elif self.backbone == "hrnet":
            feats, low_feat = self.hrnet(x)

        out_aux = self.aux_head(feats)  # backbone

        # Dual Re-sample feature multi-scale
        sample1 = self.ms_local_context(feats)  # s1
        sample2 = self.ms_local_context(feats)  # s2

        Attn_res1 = self.eff_attention_l1(low_feat)

        sample1 = F.interpolate(sample1, size=(Attn_res1.shape[2], Attn_res1.shape[2]), mode='bilinear',
                                align_corners=True)
        sample2 = F.interpolate(sample2, size=(Attn_res1.shape[2], Attn_res1.shape[2]), mode='bilinear',
                                align_corners=True)

        samplgpf_d1 = self.com_LGPF_d1(sample1, Attn_res1)
        samplgpf_d2 = self.com_LGPF_d2(sample2, samplgpf_d1)  # first output

        sample_out = self.com_LGPF_out(samplgpf_d2, Attn_res1)  # out
        out = self.atrous_out1(sample_out)
        out = self.atrous_out2(out)

        out = F.interpolate(out, size=(out.shape[2] * self.expand, out.shape[2] * self.expand),
                            mode='bilinear', align_corners=True)

        out = self.atrous_out3(out)
        out = self.cls_head(out)

        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)
        return out_aux_seg
