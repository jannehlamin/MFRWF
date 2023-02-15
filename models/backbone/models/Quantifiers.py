import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
from dataloaders.data_util.utils import cdecode_segmap
from models.backbone.conformer import ConvTransBlock, Block


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

class Context_Flow(nn.Module):
    def __init__(self, channel, r=4, kernel_size=3, stride=1, dilation=1, bias=False):
        super(Context_Flow, self).__init__()
        plane = channel//r
        self.conv1 = nn.Conv2d(channel, plane, kernel_size, stride, 0, dilation, groups=plane, bias=bias)
        self.bn = nn.BatchNorm2d(plane)
        self.pointwise = nn.Conv2d(plane, channel, 1, 1, 0, 1, 1, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        x = self.relu(x)
        return x

class TargetAwared(nn.Module):
    def __init__(self, in_ch, out_ch, is_last=False):
        super(TargetAwared, self).__init__()
        self.conv3x3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)

        # if is_last:
        #     self.conv7x7 = nn.Conv2d(out_ch, out_ch//128, kernel_size=1, padding=1, stride=1, bias=False)
        # else:
        self.conv7x7 = nn.Conv2d(out_ch, out_ch, kernel_size=1, padding=0, stride=1, bias=False)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv3x3(x)))
        x = self.conv7x7(x)
        return x

class LMarsking(nn.Module):
    def __init__(self, channel, r=4):
        super(LMarsking, self).__init__()
        self.locals = Context_Flow(channel, r)

    def forward(self, x):
        return self.locals(x)

class GMarsking(nn.Module):
    def __init__(self, channel, r=4):
        super(GMarsking, self).__init__()
        plane = channel // r
        self.globals = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, plane, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(plane),
            nn.ReLU(inplace=True),
            nn.Conv2d(plane, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
        )

    def forward(self, x):
        return self.globals(x)

class GLMarskingFusion(nn.Module):
    def __init__(self, channel, r=4):
        super(GLMarskingFusion, self).__init__()
        interchannel = channel//r
        self.locals = LMarsking(channel, interchannel)
        # self.globals = GMarsking(channel, interchannel)  # comment for the case of transformer

        self.sigmoid = nn.Sigmoid()

    def forward(self, xl, xg):

        x1 = xl + xg
        xls = self.locals(xl)
        # xgs = self.globals(xg)
        r = xls  # + xgs

        weight = self.sigmoid(r)

        target = x1 * weight  # target
        untarget = x1 * (1 - weight)  # untarget
        out = (target - untarget)
        return out



class GLIntegration(nn.Module):
    def __init__(self, inplanes, outplanes, res_conv, embed_dim, stride=1, patch_size=16, depth=2, num_heads=8,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 isFirst=False, isUp=False):
        super(GLIntegration, self).__init__()
        trans_dw_stride = patch_size // 4
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
            # self.trans_adjust = TranReshape(48, outplanes)

    def forward(self, x, x_tr=None):
        if self.isFirst:
            B = x.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x_t = self.trans_patch_conv(x).flatten(2).transpose(1, 2)
            x_t = torch.cat([cls_tokens, x_t], dim=1)
            x_t = self.trans_1(x_t)
            return x_t
        else:
            # self.trans_adjust(x_tr, x)
            trans_cnn = self.contrans(x, x_tr)
            return trans_cnn

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # print("Janneh=>", (C // self.num_heads), B, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  attn_drop=attn_drop, proj_drop=drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        # self-attention 
        x = self.norm1(x)
        eA_x = self.attn(x)  # it is linear time complexity
        x = self.attn_drop(eA_x)
        x = x.transpose(2, 1).reshape(B, C, H, W)
        return x

class AtrousDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super(AtrousDecoder, self).__init__()
        self.expand = 1
        self.con1 = ASPP_module(in_ch, out_ch, dilation=dilation)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        out = self.con1(output)
        out = F.interpolate(out, size=(output.shape[2]*self.expand, output.shape[2]*self.expand), mode='bilinear', align_corners=True)
        return out

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, stride=1):
        super(ASPP_module, self).__init__()
        if dilation == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = dilation
        self.dilation = dilation

        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)


class multiScaleCNN(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super(multiScaleCNN, self).__init__()
        self.con1 = ASPP_module(in_ch, out_ch, dilation=dilation)
        # self.con2 = ASPP_module(out_ch, in_ch, dilation=dilation)

    def forward(self, x):
        out = self.con1(x)
        # out = self.con2(out)
        return out

# =========================================================================================================== #
class LGPF(nn.Module):
    def __init__(self, in_ch, out_ch, inter_ch):
        super(LGPF, self).__init__()
        self.con = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.con1 = nn.Conv2d(inter_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()
        self.preservation = PresCnn(in_ch, out_ch)

    def forward(self, x_cnn, x_attn):
        x_cnn = self.relu(self.bn(self.con(x_cnn)))
        x_attn = self.relu(self.bn(self.con1(x_attn)))

        gl = x_cnn + x_attn
        weight = self.sigmoid(gl)
        out_feat = ((2 * x_cnn * weight) + (1-weight)) + ((2 * x_attn * weight) + (1-weight))

        return self.preservation(out_feat)

# This is the
class EfficientAttentions(nn.Module):

    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):

        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                            :,
                            i * head_key_channels: (i + 1) * head_key_channels,
                            :
                            ], dim=2)
            query = F.softmax(queries[
                              :,
                              i * head_key_channels: (i + 1) * head_key_channels,
                              :
                              ], dim=1)
            value = values[
                    :,
                    i * head_value_channels: (i + 1) * head_value_channels,
                    :
                    ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                    context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention

class PresCnn(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=[12, 6, 3]):
        super(PresCnn, self).__init__()
        self.con1 = ASPP_module(in_ch, out_ch, dilation=dilation[0])
        self.con2 = ASPP_module(out_ch, out_ch, dilation=dilation[1])
        self.con3 = ASPP_module(out_ch, in_ch, dilation=dilation[2])

    def forward(self, x):
        out = self.con1(x)
        out = self.con2(out)
        out = self.con3(out)
        return out

# x = torch.randn(2, 3, 32, 32)
# y = torch.randn(2, 3, 32, 32)
# model = AtrousDecoder(6, 3)
# result = model(x, y)
# print(result.shape)

# cont1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=True)
# bn1 = nn.BatchNorm2d(64)
# rel1 = nn.ReLU()
# cont2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, bias=True)
# bn2 = nn.BatchNorm2d(128)
# rel2 = nn.ReLU()
# cont3 = nn.Conv2d(3, 256, kernel_size=1, stride=1, bias=True)
# out = GLMarskingFusion(256)
# outfin = nn.Conv2d(256, 3, kernel_size=1, stride=1)


# filename = "Label_7.png"
# img = Image.open(filename)
# img = np.array(img)
#
# segmap = cdecode_segmap(img, dataset="cweeds", plot=True)
# print(segmap.shape)
# segmap = np.array(segmap * 255).astype(np.uint8)
# pred = Image.fromarray(segmap.astype(np.uint8))
# pred = pred.resize((1024, 1024))
# pred.save("_result.png")
# rgb_img = cv2.resize(segmap, (912, 1024), interpolation=cv2.INTER_NEAREST)
# bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
# cv2.imwrite("1_result.png", bgr)

# segmap = np.array(segmap * 255).astype(np.uint8)
# img_ten = torch.from_numpy(img).unsqueeze(1).permute(1, 3, 0, 2).float()
# # print(img_ten.shape)
# # ====================================
# # x = rel1(bn1(cont1(img_ten)))
# # x = rel2(bn2(cont2(x)))
# x = cont3(img_ten)
# x = out(x, x)
# x = outfin(x)
# x = x.squeeze(0).permute(1, 2, 0).detach().numpy().astype('uint8')
# # print("Janneh->", x.shape, img.shape)
#
# plt.subplot(221), plt.imshow(img)
# plt.subplot(222), plt.imshow(x)
# plt.show()
