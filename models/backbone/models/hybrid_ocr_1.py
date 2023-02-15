import math
from torch import nn
import torch
import torch.nn.functional as F
from .conformer import ConvTransBlock, Block
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=12, stype='normal', sample_size=3):
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
        embed_dim = 48
        self.sample_size = sample_size

        # integration
        local_sample = []
        global_sample = [GLIntegration(3, 256, res_conv=True, isFirst=True, isUp=False, embed_dim=embed_dim)]
        if sample_size < 2: sample_size = 2
        for i in range(sample_size):
            local_sample.append(MS_Coder(inplanes, planes, width, scale, k_size=3, stride=stride))
            if i > 0:
                global_sample.append(GLIntegration(256, 256, res_conv=True, isFirst=False, isUp=False, embed_dim=embed_dim))
        global_sample.append(GLIntegration(256, 256, res_conv=True, isFirst=False, isUp=True, embed_dim=embed_dim))

        self.locals = nn.ModuleList(local_sample)
        self.globals = nn.ModuleList(global_sample)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x, classMap):
        # initial value to capture the transformer initial patches
        class_map = self.globals[0](classMap)  # trans 1
        sample = self.locals[0](x)
        out = self.globals[1](sample, class_map)
        # FCU --> Feature coupling Unit
        for i in range(1, self.sample_size):
            sample = self.locals[i](x)  # sample from the multiscale
            out = self.globals[i+1](sample, out)
        return out


class GLIntegration(nn.Module):
    def __init__(self, inplanes, outplanes, res_conv, embed_dim, stride=1, patch_size=16, depth=2, num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., isFirst=False,
                 isUp=False):
        super(GLIntegration, self).__init__()
        trans_dw_stride = patch_size//4
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.isFirst = isFirst

        # transit
        self.trans_patch_conv = nn.Conv2d(inplanes, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride,
                                          padding=0)
        # transformer
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0]
                             )
        if not isFirst:
            self.contrans = ConvTransBlock(inplanes, outplanes, res_conv, stride=stride, dw_stride=trans_dw_stride,
                                           embed_dim=embed_dim, last_fusion=False,
                                           num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                           qk_scale=qk_scale,
                                           drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                           drop_path_rate=self.trans_dpr[0],
                                           num_med_block=0, isUp=isUp)

    def forward(self, x, x_tr=None):
        if self.isFirst:
            B = x.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x_t = self.trans_patch_conv(x).flatten(2).transpose(1, 2)
            x_t = torch.cat([cls_tokens, x_t], dim=1)
            x_t = self.trans_1(x_t)
            return x_t
        else:
            cnn_trans = self.contrans(x, x_tr)
            return cnn_trans


class MS_Coder(nn.Module):
    def __init__(self, inplanes, planes, width, scale, k_size=3, stride=1, stype='normal'):
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

        self.conv3 = nn.Conv2d(width * scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu3 = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):

        out = self.relu(self.bn1(self.conv1(x)))
        # First tier of the multi-scale  and the Encoder section of the hidden U-Net
        spx = torch.split(out, self.width, 1)
        # print("Hello=>", spx[0].shape, spx[1].shape, spx[2].shape, len(spx))
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]

            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            sp = self.cf(sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = self.relu3(self.bn3(self.conv3(out)))
        return out


class OHybridCR(nn.Module):
    def __init__(self, ocr_key_channels, num_classes, backbone="hrnet", sample_size=2):
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
            self.hrnet = get_seg_model(config)

        self.conv1 = nn.Conv2d(256, 256, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(256)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        self.cls_head = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # this is the multi-scale section
        self.multi_context = Multi_Context(ocr_key_channels, ocr_key_channels, sample_size=sample_size)

        self.aux_head = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

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

        # print("Lamin=>", feats.shape)
        feats = self.maxpool(self.act1(self.bn1(self.conv1(feats))))
        out_aux = self.aux_head(feats)  # backbone

        out = self.multi_context(feats, out_aux)  # Dual Re-sample feature multi-scale
        out = self.cls_head(out)

        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        out_aux = F.interpolate(out_aux, size=x.shape[2:], mode='bilinear', align_corners=True)

        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)
        return out_aux_seg
