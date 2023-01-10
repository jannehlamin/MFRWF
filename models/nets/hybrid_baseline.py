from torch import nn

from ..backbone.LWbaseline import LWBaseline
from ..backbone.lighweigh_bb_nostream import LResNet34FRWM
ALIGN_CORNERS = True


class OHybridCR(nn.Module):
    def __init__(self, args, num_classes, backbone="hrnet", isup_decoder=False):
        super(OHybridCR, self).__init__()

        self.isup_decoder = isup_decoder
        # Backbones
        self.backbone = backbone
        self.resnet = LWBaseline()

        self.aux_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):

        feats = self.resnet(x)
        out = self.aux_head(feats)  # backbone

        return [out]
