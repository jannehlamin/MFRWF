import torch as t
import torchvision
from torch import nn
import torch.nn.functional as F
from torchinfo import summary


class Double_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Double_conv, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        incat = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1)

    def forward(self, x):
        return self.conv(x)


class Enconvolution(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Enconvolution, self).__init__()
        self.conv = Double_conv(in_ch, out_ch)
        self.max = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        x = self.max(input)
        x = self.conv(x)
        return x

class Deconvolution(nn.Module):

    def __init__(self, in_ch, out_ch, islast=False):
        super(Deconvolution, self).__init__()
        # self.ups1 = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        pad = 0
        k_size = 3
        if islast:
            pad = 1
            k_size = 1
        self.conv = nn.Conv2d(in_ch*2, out_ch, kernel_size=k_size, stride=1, padding=pad)

    def forward(self, x1, x2):
        # x1 = self.ups1(x1)
        x1 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x = t.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class MUNET(nn.Module):

    def __init__(self, num_channel=3, num_class=3):
        super(MUNET, self).__init__()

        # Encoder section
        self.con_3_1 = Enconvolution(num_channel, 64)
        self.con_3_2 = Enconvolution(64, 128)
        self.con_3_3 = Enconvolution(128, 256)
        self.con_7 = nn.Conv2d(256, 256, kernel_size=3, stride=1) # end encoder

        # Decoder section
        self.dec1 = Deconvolution(256, 128)
        self.dec2 = Deconvolution(128, 64)
        self.dec3 = Deconvolution(64, 3)
        self.dec4 = Deconvolution(3, num_class, islast=True) # concatenate the last layer with the input of 3 channel
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder
        incat = x
        x1 = self.con_3_1(x)
        x2 = self.con_3_2(x1)
        x3 = self.con_3_3(x2)
        x = self.con_7(x3)
        # Decoder
        x = self.dec1(x, x3)
        x = self.dec2(x, x2)
        x = self.dec3(x, x1)
        x = self.dec4(x, incat)
        x = self.out(x)

        return [x]

def get_10x_lr_params_unet(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.con_3_1, model.con_3_2, model.con_3_3, model.con_3_4, model.con_3_5, model.dec1, model.dec2,
         model.dec3, model.dec4, model.out]

    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


# model = MUNET(num_channel=3, num_class=3)
# inputs = t.randn(2, 3, 32, 32)
# result = model(inputs)
# print(result.shape)
# summary(model)

