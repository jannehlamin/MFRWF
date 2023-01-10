import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
from models.dual_nets.utils import ConvWFusion

def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

class General_Flow(nn.Module):
    def __init__(self, out_ch):
        super(General_Flow, self).__init__()
        self.g_avg = nn.AdaptiveAvgPool2d(1)
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

class SFCM(nn.Module):
    def __init__(self, in_ch, std_out, gated_method=False, is_add=False):
        super(SFCM, self).__init__()
        self.gated_method = gated_method
        self.is_add = is_add
        self.conv3x3 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv_out = nn.Conv2d(in_ch, std_out, kernel_size=1, bias=False)

    def forward(self, x):
        x_skip = x
        if self.gated_method:
            gated = nn.Sigmoid()
        else:
            gated = nn.Softmax(dim=1)

        x = self.relu(self.bn(self.conv3x3(x)))
        x = gated(x) + 1
        untarget = 1 - x
        out = F.interpolate(x_skip, size=x.shape[2:], mode='bilinear', align_corners=True)
        target = x * out
        if self.is_add:
            target = target + x
        target = self.conv_out(target)
        return target, untarget

class EGLF(nn.Module):
    def __init__(self, in_ch, std_out=32, gated_method=False):
        super(EGLF, self).__init__()
        self.gated_method = gated_method

        self.ncon = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.dcon = ASPP_module(in_ch, in_ch, dilation=3)  # nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1)
        # self.ea = EfficientAttentions(in_ch, in_ch, 16, in_ch)
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv_in = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.conv_out = nn.Conv2d(in_ch, std_out, kernel_size=1, bias=False)
        self.xg = General_Flow(in_ch)
        self.xl = Context_Flow(in_ch, in_ch)

    def forward(self, x):
        x_skip = x
        if self.gated_method:
            gated = nn.Sigmoid()
        else:
            gated = nn.Softmax(dim=1)

        x1 = self.relu(self.bn(self.ncon(x)))
        x2 = self.relu(self.bn(self.dcon(x)))
        x = x1 + x2

        xl = self.xl(x)
        xg = self.xg(x)
        m = xl+xg
        # m = self.ea(m)
        x = gated(m)+1

        untarget = self.conv_in(1-x)
        out = F.interpolate(x_skip, size=x.shape[2:], mode='bilinear', align_corners=True)
        target = self.conv_out(x * out)
        return target, untarget


class IntraScaled(nn.Module):
    def __init__(self, inplanes, width, scale, std_out=32, k_size=3, stride=1, stype='normal', gated_method=False):
        super(IntraScaled, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.relu = nn.ReLU(inplace=True)
        self.gated_method = gated_method

        self.conv_out = nn.Conv2d(width * scale, std_out, kernel_size=1, bias=False)
        self.conv_in = nn.Conv2d(width * scale, inplanes, kernel_size=1, bias=False)

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
        x_skip = out
        if self.gated_method:
            gated = nn.Sigmoid()
        else:
            gated = nn.Softmax(dim=1)

        # First tier of the multi-scale  and the Encoder section of the hidden U-Net
        spx = torch.split(out, self.width, 1)
        output = []
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
                output.append(sp)
            else:
                sp = sp + spx[i]

                sp = self.convs[i](sp)
                sp = self.relu(self.bns[i](sp))
                sp = gated(sp) + 1  # multi-scale gate fusion
                output.append(sp)
        output = torch.cat(output, dim=1)
        x_skip = F.interpolate(x_skip, size=out.shape[2:], mode='bilinear', align_corners=True)
        output = output * x_skip
        untarget = self.conv_in((1-output))
        target = self.conv_out(output)
        return target, untarget


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

class AMscaling(nn.Module):
    def __init__(self, in_ch, dilation, std_out=32, gated_method=False):
        super(AMscaling, self).__init__()
        self.gated_method = gated_method

        self.dcon1 = ASPP_module(in_ch, in_ch, dilation=dilation[0])
        self.dcon2 = ASPP_module(in_ch, in_ch, dilation=dilation[1])
        self.dcon3 = ASPP_module(in_ch, in_ch, dilation=dilation[2])
        self.dcon4 = ASPP_module(in_ch, in_ch, dilation=dilation[3])
        self.conv_x = nn.Conv2d(in_ch * 4, in_ch, kernel_size=1, bias=False)
        self.conv_out = nn.Conv2d(in_ch, std_out, kernel_size=1, bias=False)
        self.conv_in = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)

    def forward(self, x):
        x_skip = x
        if self.gated_method:
            gated = nn.Sigmoid()
        else:
            gated = nn.Softmax(dim=1)

        dcon1 = gated(self.dcon1(x)) + 1
        dcon2 = gated(self.dcon2(x)) + 1
        dcon3 = gated(self.dcon3(x)) + 1
        dcon4 = gated(self.dcon4(x)) + 1

        x = torch.cat((dcon1, dcon2, dcon3, dcon4), dim=1)
        x = self.conv_x(x)
        out = F.interpolate(x_skip, size=x.shape[2:], mode='bilinear', align_corners=True)
        target = self.conv_out(x * out)
        untarget = self.conv_in(1-x)

        return target, untarget


class DFP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DFP, self).__init__()
        # self.conv0 = nn.Conv2d(128, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xlist, inp):
        # xlist.append(ppm)
        xlist.reverse()
        out = []
        for wei_layer in xlist:
            wei_layer = F.interpolate(wei_layer, inp.shape[2:], mode='bilinear', align_corners=True)
            out.append(wei_layer)

        # layers
        # x0 = self.relu(self.bn(self.conv0(out[0])))
        # x1 = out[1] + x0

        x1 = self.relu(self.bn(self.conv1(out[0])))
        x2 = out[1] + x1

        x2 = self.relu(self.bn(self.conv1(x2)))
        x3 = out[2] + x2 + x1
        x3 = self.relu(self.bn(self.conv1(x3)))

        x4 = out[3] + x3 + x2 + x1
        x4 = self.relu(self.bn(self.conv1(x4)))

        out = [x1, x2, x3, x4]
        out = torch.cat(out, dim=1)
        return out

class DecoderFF(nn.Module):
    def __init__(self, channel):
        super(DecoderFF, self).__init__()
        # convolution weighted feature fusion - decoder
        self.conv0 = nn.Conv2d(128, channel, kernel_size=3, stride=1, padding=1)
        self.conwf4 = ConvWFusion(channel)
        self.conwf3 = ConvWFusion(channel)
        self.conwf2 = ConvWFusion(channel)
        self.conwf1 = ConvWFusion(channel)

        self.convwfs = nn.ModuleList([self.conwf4, self.conwf3, self.conwf2, self.conwf1])

    def forward(self, ppm, xlist):
        x = xlist[0]
        if ppm is not None:
            xlist.append(ppm)
            xlist.reverse()
            x = self.conv0(xlist[0])  # # PPM

        for i in range(1, len(xlist)):
            y = xlist[i]
            out = self.convwfs[i-1](x, y)
            x = y
        return out


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class MR2N_GFF(nn.Module):
    def __init__(self, inplane, std_out):
        super(MR2N_GFF, self).__init__()
        self.conv = nn.Conv2d(inplane, std_out, kernel_size=1)

    def forward(self, target, other_targets):

        #other_targets.append(ppm)
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
#
# inputs = torch.randn(2, 64, 8, 8)
# # model = AMscaling(64, 64, [3, 5, 7, 9])
# model = EGLF(64, 64)
# summary(model)
# target, untarget = model(inputs)
# print(target.shape)
