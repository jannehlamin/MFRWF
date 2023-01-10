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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

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

    def __init__(self, in_ch, out_ch, bilinear=False):
        super(Deconvolution, self).__init__()

        if bilinear:
            self.ups1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.ups1 = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = Double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.ups1(x1)
        # x2 = self.crop(x2, x1)
        x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x = t.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNET(nn.Module):

    def __init__(self, num_channel=3, n_class=3):
        super(UNET, self).__init__()

        # Encoder section
        self.con_3_1 = Double_conv(num_channel, 64)
        self.con_3_2 = Enconvolution(64, 128)
        self.con_3_3 = Enconvolution(128, 256)
        self.con_3_4 = Enconvolution(256, 512)
        self.con_3_5 = Enconvolution(512, 1024)

        # Decoder section
        self.dec1 = Deconvolution(1024, 512)
        self.dec2 = Deconvolution(512, 256)
        self.dec3 = Deconvolution(256, 128)
        self.dec4 = Deconvolution(128, 64)
        self.out = OutConv(64, n_class)

    def forward(self, x):
        # encoder
        x1 = self.con_3_1(x)
        x2 = self.con_3_2(x1)
        x3 = self.con_3_3(x2)
        x4 = self.con_3_4(x3)
        x5 = self.con_3_5(x4)
        # Decoder
        x = self.dec1(x5, x4)

        x = self.dec2(x, x3)
        x = self.dec3(x, x2)
        x = self.dec4(x, x1)
        x = self.out(x)

        return x

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


# model = UNET(num_channel=3, num_class=5)
# inputs = t.randn(2, 3, 32, 32)
# result = model(inputs)
# summary(model)
# print(result.shape)


