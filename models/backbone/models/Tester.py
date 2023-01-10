import torch
from torchinfo import summary
from MSOHCR.Models.models.conformer import ConvTransBlock, Block
from torch import nn


class GLIntegration(nn.Module):
    def __init__(self, inplanes, outplanes, res_conv, stride, embed_dim, patch_size=12, depth=2, num_heads=6, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,isFirst=False, isUp=False):
        super(GLIntegration, self).__init__()
        trans_dw_stride = patch_size // 3
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.isFirst = isFirst

        # transit
        self.trans_patch_conv = nn.Conv2d(inplanes, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride,
                                          padding=0)
        # transformer
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads,  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                           qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0]
                        )
        if not isFirst:
            self.contrans = ConvTransBlock(inplanes, outplanes, res_conv, stride=stride, dw_stride=trans_dw_stride,
                                embed_dim=embed_dim, last_fusion=False,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[0],
                                num_med_block=0, isUp=isUp)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x_t = self.trans_patch_conv(x).flatten(2).transpose(1, 2)
        print("Lamin->", x.shape)
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t = self.trans_1(x_t)
        if not self.isFirst:
            cnn_trans = self.contrans(x, x_t)
            return cnn_trans
        return x_t


# input = torch.randn(2, 64, 12, 12).cuda()
#
# model = GLIntegration(64, 64, res_conv=True, stride=1, embed_dim=48, isFirst=True, isUp=True).cuda()
# y = model(input)
# summary(model)
# print(y.shape)
import math
# from MSOHCR.Models.models.hybrid_ocr import MS_Coder
# inputs = torch.randn(2, 3, 12, 12)
# model = MS_Coder(3, 64)
# y = model(inputs)
# print(y.shape)
# conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
# bn1 = nn.BatchNorm2d(64)
# act1 = nn.ReLU(inplace=True)
# maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]
#
# x = torch.randn(1, 3, 64, 64)
# x_base = maxpool(act1(bn1(conv1(x))))
# print("Janneh->", x_base.shape)
# # 2D conv
# conv = nn.Conv2d(64, 64, 4, 4)
# m = conv(x_base)
# print(m.shape)
# p = conv(x).reshape(-1, 64).transpose(0, 1)
# print(p.shape, conv(x).shape)
import numpy as np
p = (np.array([[1, 1, 1]]) == (0, 0, 0)).all(1)
print(p)