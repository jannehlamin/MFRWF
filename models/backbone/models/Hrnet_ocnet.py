##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from Models.models.backbones.backbone_selector import BackboneSelector
from Models.utils.helpers.module_helper import ModuleHelper


class BaseOCNet(nn.Module):
    """
    OCNet: Object Context Network for Scene Parsing
    """
    def __init__(self, configer):
        self.inplanes = 128
        super(BaseOCNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('network', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096] 
        else:
            in_channels = [192, 384]
        self.oc_module_pre = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
        )
        from Models.models.modules.base_oc_block import BaseOC_Module
        self.oc_module = BaseOC_Module(in_channels=512, 
                                       out_channels=512,
                                       key_channels=256, 
                                       value_channels=256,
                                       dropout=0.05, 
                                       sizes=([1]),
                                       bn_type=self.configer.get('network', 'bn_type'))
        self.cls = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x_):
        aux_seg = []
        x = self.backbone(x_)
        x_dsn = self.dsn(x[-2])
        x = self.oc_module_pre(x[-1])
        x = self.oc_module(x)
        x = self.cls(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)

        aux_seg.append(x_dsn)
        aux_seg.append(x)
        return aux_seg


class AspOCNet(nn.Module):
    """
    OCNet: Object Context Network for Scene Parsing
    """
    def __init__(self, configer):
        self.inplanes = 128
        super(AspOCNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('network', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096] 
        else:
            in_channels = [192, 384]
        from Models.models.modules.asp_oc_block import ASP_OC_Module
        self.context = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            ASP_OC_Module(512, 256, bn_type=self.configer.get('network', 'bn_type')),
        )
        self.cls = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x_):
        aux_seg = []
        x = self.backbone(x_)
        aux_x = self.dsn(x[-2])
        x = self.context(x[-1])
        x = self.cls(x)
        aux_x = F.interpolate(aux_x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)

        aux_seg.append(aux_x)
        aux_seg.append(x)
        return aux_seg
