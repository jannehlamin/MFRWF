##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Donny You, RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Models.models.backbones.resnet.resnet_backbone import ResNetBackbone
from Models.models.backbones.hrnet.hrnet_backbone import HRNetBackbone
from Models.utils.helpers.logger import Logger as Log


class BackboneSelector(object):

    def __init__(self, configer):
        self.configer = configer

    def get_backbone(self, **params):
        backbone = self.configer.get('network', 'backbone')

        model = None
        if ('resnet' in backbone or 'resnext' in backbone or 'resnest' in backbone) and 'senet' not in backbone:
            model = ResNetBackbone(self.configer)(**params)
            # print("Janneh 1")

        elif 'hrne' in backbone:
            model = HRNetBackbone(self.configer)(**params)
            # print("Janneh 2")

        else:
            Log.error('Backbone {} is invalid.'.format(backbone))
            exit(1)

        return model

