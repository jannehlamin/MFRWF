import torch.nn as nn
import torch.nn.functional as F
from Models.models.backbones.backbone_selector import BackboneSelector
from Models.models.bn_helper import BatchNorm2d


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


class ISANet(nn.Module):
    """
    Interlaced Sparse Self-Attention for Semantic Segmentation
    """
    def __init__(self, configer):
        self.inplanes = 128
        super(ISANet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('network', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        bn_type = self.configer.get('network', 'bn_type')
        factors = self.configer.get('network', 'factors')
        from Models.models.modules.isa_block import ISA_Module
        self.isa_head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, bias=False),
            ModuleHelper.BNReLU(512, bn_type=bn_type),
            ISA_Module(in_channels=512, key_channels=256, value_channels=512, 
                out_channels=512, down_factors=factors, dropout=0.05, bn_type=bn_type),
        )

        self.cls_head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False),
            ModuleHelper.BNReLU(512, bn_type=bn_type),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x_):
        aux_seg = []
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.isa_head(x[-1])
        x = self.cls_head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)

        aux_seg.append(x_dsn)
        aux_seg.append(x)
        return aux_seg


# parser = argparse.ArgumentParser()
# parser.add_argument('--configs', default=None, type=str,
#                     dest='configs', help='The file of the hyper parameters.')
# parser.add_argument('--phase', default='train', type=str,
#                     dest='phase', help='The phase of module.')
# parser.add_argument('--gpu', default=[0], nargs='+', type=int,
#                     dest='gpu', help='The gpu list used.')

# args = parser.parse_args()
# config = Configer(args_parser=args)
# update_config(config, args)
#
# import torch as t
# model = ISANet(config)
# inputs = t.randn(2, 3, 32, 32)
# aux, x = model(inputs)
# summary(model)
# print("Aux->", aux.shape)
# print("x->", x.shape)
