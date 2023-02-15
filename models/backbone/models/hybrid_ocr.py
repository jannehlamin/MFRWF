import math
from torch import nn
import torch
import torch.nn.functional as F

from .Quantifiers import LMarsking, GMarsking, GLMarskingFusion, GLIntegration, TargetAwared, AtrousDecoder, \
    multiScaleCNN
from .backbones.resnet.regnet import RegNetX_200MF
from .backbones.resnet.resnet import ResNet101, ResNet50, ResNet34, ResNet18
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

    def __init__(self, inplanes, planes, size, stride=1, downsample=None, baseWidth=26, scale=12, stype='normal'):
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
        self.ms_encoder = MS_Coder(inplanes, width, scale, size, k_size=3, stride=stride)
        self.aff = iAFF(channels=width, r=4)
        # decoder
        self.ms_decoder = MS_Coder(inplanes, width, scale, size, k_size=3, stride=stride)

        self.conv3 = nn.Conv2d(width * scale, planes, kernel_size=3, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        out = self.conv(x)
        enc_list = self.ms_encoder(x)
        dec_list = self.ms_encoder(x)

        value = len(enc_list) - 1
        while value >= 0:
            if value == len(enc_list) - 1:
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


class CoupleTranCNN(nn.Module):
    def __init__(self, match=4):
        super(CoupleTranCNN, self).__init__()

        # transformer
        embed_dim = 48
        expansion = 4

        if match < 2: match = 2
        self.match = match  # [[4, 64], [3, 128], [2, 256], [1, 512]]
        # this is the transition backbone layer transformer global extraction
        channel_converter = []
        inplane = 64
        for i in range(self.match):
            inplane = inplane * expansion
            channel_converter.append(nn.Conv2d(inplane, 256, kernel_size=3, stride=1, padding=1, bias=True))
            inplane = inplane // 2
        self.cnnformer = nn.ModuleList(channel_converter)
        is_up = False
        is_last = False
        in_ch = out_ch = 256
        # enforcing local extraction before transformer take action
        self.local_pre_cnnformer = LMarsking(out_ch)

        self.initial_map = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False)  # trans initial map
        if self.match == 2:
            is_up = True
            is_last = True

        # target aware module retain our edges of the target object
        tart_aware_module = [TargetAwared(out_ch, out_ch, is_last=is_last),
                             TargetAwared(out_ch, out_ch, is_last=is_last)]
        global_sample = [GLIntegration(in_ch, out_ch, res_conv=True, isFirst=True, isUp=False, embed_dim=embed_dim),
                         GLIntegration(out_ch, out_ch, res_conv=True, isFirst=False, isUp=is_up,
                                       embed_dim=embed_dim)]

        self.global_post_cnnformer = GMarsking(out_ch)  # global extraction
        self.gl_fusion = GLMarskingFusion(out_ch)  # global and local for target edge extraction
        # get all the remaining layer where last layer output only CNN
        for i in range(2, self.match):
            if i == self.match - 1:  # output only CNN
                is_up = True
                is_last = True
            global_sample.append(
                GLIntegration(in_ch, out_ch, res_conv=True, isFirst=False, isUp=is_up, embed_dim=embed_dim))
            tart_aware_module.append(TargetAwared(out_ch, out_ch, is_last=is_last))

        self.tran_last = nn.Conv2d(out_ch * 2, 128, kernel_size=1, stride=1, bias=False)
        self.tart_aware_module = nn.ModuleList(tart_aware_module)
        self.globals = nn.ModuleList(global_sample)

        self.decoder_level1 = AtrousDecoder(256, 64, dilation=6)
        self.decoder_level2 = AtrousDecoder(192, 128, dilation=12)

    def forward(self, output, layers):
        step = 0
        _, _, H, W = layers[0].shape

        for i, out in enumerate(layers[:self.match]):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)

            out = self.cnnformer[i](out)  # channel transformation to standardize the multi-level fusion
            out_l = self.local_pre_cnnformer(out)  # get the local details
            # This is the beginning of the transformer global feature extraction # default case
            if i < 1:  # initial position
                init_out = self.initial_map(out)  # out 256
                initial_map = self.globals[step](out)  # trans 1 out 256
                step = step + 1
                tran = self.globals[step](init_out, initial_map)
                # and 2
                if not isinstance(tran, tuple):
                    tran_out_cnn = tran
                    tran_out = tran
                else:
                    tran_out = tran[0]
                    tran_out_cnn = tran[1]

                out_g = self.global_post_cnnformer(tran_out_cnn)  # get the global details
                gl_fusion = self.gl_fusion(out_l, out_g) # edge-oriented feature
                target_aware = self.tart_aware_module[i](tran_out_cnn + gl_fusion)
            else:  # other multi-level cases
                step = step + 1
                if step < self.match:
                    tran = self.globals[step](out, tran_out)  # updating the transformer section
                    # more than 2
                    if not isinstance(tran, tuple):  # if conformer return only cnn
                        tran_out_cnn = tran
                        tran_out = tran
                    else:                            # if conformer return both cnn and transformer output
                        tran_out = tran[0]
                        tran_out_cnn = tran[1]

                    out_g = self.global_post_cnnformer(tran_out_cnn)
                    gl_fusion = gl_fusion + self.gl_fusion(out_l, out_g)
                    target_aware = target_aware + self.tart_aware_module[i](gl_fusion)

        # This is the decoder section
        expose_edge = torch.cat((tran_out, target_aware), dim=1)
        expose_edge = self.tran_last(expose_edge)
        output = F.interpolate(output, size=expose_edge.shape[2:], mode='bilinear', align_corners=True)
        output = self.decoder_level1(output, expose_edge)

        expose_edge = F.interpolate(expose_edge, size=output.shape[2:], mode='bilinear', align_corners=True)
        output = self.decoder_level2(output, expose_edge)

        return output


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


class CoupleScaleCNN(nn.Module):
    def __init__(self, match=4):
        super(CoupleScaleCNN, self).__init__()

        # CNN base scaling global and local feature entraction
        expansion = 1

        if match < 2: match = 2
        self.match = match  # [[4, 64], [3, 128], [2, 256], [1, 512]]
        # this is the transition backbone layer transformer global extraction
        channel_converter = []
        inplane = 64
        for i in range(self.match):
            inplane = inplane * expansion
            channel_converter.append(nn.Conv2d(inplane, 256, kernel_size=3, stride=1, padding=1, bias=True))
            inplane = inplane * 2
        self.cnnformer = nn.ModuleList(channel_converter)

        in_ch = out_ch = 256
        # enforcing local extraction before transformer take action
        self.local_pre_cnnformer = LMarsking(out_ch)

        # target aware module that retain out edges of out target object
        tart_aware_module = [TargetAwared(out_ch, out_ch)]
        global_sample = [multiScaleCNN(in_ch, out_ch, dilation=1)]

        self.global_post_cnnformer = GMarsking(out_ch)  # global extraction
        self.gl_fusion = GLMarskingFusion(out_ch)  # global and local for target edge extraction
        # get all the remaining layer where last layer output only CNN
        for i in range(1, self.match):
            global_sample.append(multiScaleCNN(in_ch, out_ch, dilation=3 * i))
            tart_aware_module.append(TargetAwared(out_ch, out_ch))

        self.tran_last = nn.Conv2d(out_ch * 2, 128, kernel_size=1, stride=1, bias=False)
        self.tart_aware_module = nn.ModuleList(tart_aware_module)
        self.globals = nn.ModuleList(global_sample)

        self.decoder_level1 = AtrousDecoder(256, 64, dilation=6)
        self.decoder_level2 = AtrousDecoder(192, 128, dilation=12)

    def forward(self, output, layers):
        _, _, H, W = layers[0].shape

        for i, out in enumerate(layers[:self.match]):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)

            out = self.cnnformer[i](out)  # channel transformation
            out_l = self.local_pre_cnnformer(out)  # local details

            tran_out_cnn = self.globals[i](out)  # multi-scaling change the sematics

            out_g = self.global_post_cnnformer(tran_out_cnn)  # global details
            gl_fusion = self.gl_fusion(out_l, out_g)  # edge-oriented
            target_aware = self.tart_aware_module[i](gl_fusion)

        # This is the decoder section
        expose_edge = torch.cat((tran_out_cnn, target_aware), dim=1)
        expose_edge = self.tran_last(expose_edge)
        output = F.interpolate(output, size=expose_edge.shape[2:], mode='bilinear', align_corners=True)
        output = self.decoder_level1(output, expose_edge)

        expose_edge = F.interpolate(expose_edge, size=output.shape[2:], mode='bilinear', align_corners=True)
        output = self.decoder_level2(output, expose_edge)
        return output


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
            size = 32

        self.cls_head = nn.Conv2d(131, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # this is the multi-scale section
        # self.multi_context = Multi_Context(ocr_key_channels, ocr_key_channels, size=size)
        self.ctrancnn = CoupleTranCNN(match=3)
        # self.multi_context = Multi_Context(ocr_key_channels//2, ocr_key_channels, size=size)
        self.aux_head = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(256),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        out_aux_seg = []
        if self.backbone == "resnet_50":
            inca, feats, layers = self.resnet(x)
        elif self.backbone == "resnet_101":
            inca, feats, layers = self.resnet(x)
        elif self.backbone == "regnet":
            feats = self.regnet(x)
        elif self.backbone == "hrnet":
            feats, layers = self.hrnet(x)

        # for i, layer in enumerate(layers):
        #     print("Janneh->", i, layer.shape)
        # print(feats.shape, "Eyode")
        out_aux = self.aux_head(feats)  # backbone

        out = self.ctrancnn(feats, layers)  # Dual Re-sample feature multi-scale
        inca = F.interpolate(inca, size=x.size()[2:], mode='bilinear', align_corners=True)
        out = torch.cat((out, inca), dim=1)
        out = self.cls_head(out)
        # out_ = self.multi_context(feats)
        # out = out + out_

        out_aux = F.interpolate(out_aux, size=x.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=True)
        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)
        return out_aux_seg
