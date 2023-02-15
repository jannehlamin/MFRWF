import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class MaskEdge(nn.Module):
    def __init__(self, std_out):
        super(MaskEdge, self).__init__()
        self.edge1 = nn.Conv2d(1, std_out, kernel_size=1, stride=1, padding=0)
        self.edge2 = nn.Conv2d(std_out, std_out, kernel_size=3, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(std_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, mask):
        mask = mask.unsqueeze(1).type(torch.FloatTensor).cuda()
        x = self.edge1(mask)
        x = self.edge2(x)
        x = self.bn(x)
        edge = self.relu(x)
        return edge

class MSFCM_GFF(nn.Module):
    def __init__(self, inplane, std_out):
        super(MSFCM_GFF, self).__init__()
        self.conv = nn.Conv2d(inplane, std_out, kernel_size=1)

    def forward(self, target, other_targets, edge):
        other_targets.append(self.conv(edge))
        target = target  # already gated
        untarget = 1 - target
        others_gated_sum = []
        for othe_t in other_targets:
            # size regualtion both spatial and channel-wise
            otarget = othe_t  # already re-weighted
            other_target = F.interpolate(otarget, size=target.shape[2:], mode='bilinear', align_corners=True)
            others_gated_sum.append(other_target)

        others_gated_sums = torch.sum(torch.stack(others_gated_sum, dim=0), dim=0)

        output_other = untarget * others_gated_sums
        final_result = output_other + target
        return final_result

class E_GFF(nn.Module):
    def __init__(self, t_ch, other_tchs, std_out=32, gated_method=True):
        super(E_GFF, self).__init__()
        # masks for edge information
        other_tchs.append(std_out)  # mask or edge channel
        # standard channels
        channel_regulation = [nn.Conv2d(t_ch, std_out, kernel_size=1, stride=1, padding=0)]
        for other_tch in other_tchs:
            channel_regulation.append(nn.Conv2d(other_tch, std_out, kernel_size=1, stride=1, padding=0))
        self.channel_regulation = nn.ModuleList(channel_regulation)
        self.other_tchs = other_tchs
        self.gated_method = gated_method

    def forward(self, target, other_targets, edge):

        other_targets.append(edge)  # mask or edge channel
        if self.gated_method: gated_method = nn.Sigmoid()
        else: gated_method = nn.Softmax(dim=1)
        # standardization of the feature fusion
        if len(self.other_tchs) != len(other_targets):
            raise ValueError('error: unmatch size for the untargeted feature maps')

        target = self.channel_regulation[0](target)
        p_gated = gated_method(target)
        n_gated = 1 - p_gated
        p_target = target * (p_gated + 1)
        others_gated_sum = []
        for i in range(1, len(self.channel_regulation)):
            # size regualtion both spatial and channel-wise
            other_target = F.interpolate(other_targets[i-1], size=target.shape[2:], mode='bilinear', align_corners=True)
            other_target = self.channel_regulation[i](other_target)

            # gating mechanism
            po_target = gated_method(other_target)
            po_target = other_target * (po_target + 1)
            others_gated_sum.append(po_target)

        others_gated_sums = torch.sum(torch.stack(others_gated_sum, dim=0), dim=0)

        output = n_gated * others_gated_sums
        final_result = output + p_target + target
        return final_result


class ConvWFusion(nn.Module):
    def __init__(self, channel):
        super(ConvWFusion, self).__init__()
        self.conv = nn.Conv2d(2*channel, channel, kernel_size=1, stride=1, padding=0)
        self.convwf = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        x = F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=True)
        x_product = torch.cat((x, y), dim=1)
        x = self.conv(x_product) + x
        x = self.relu(self.bn(self.convwf(x)))
        return x
        
        
