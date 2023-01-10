from torch import nn
import torch.nn.functional as F
import torch

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=2):
        super(BasicBlock, self).__init__()
        self.bb1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bb2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.Conv2d(self.expansion*planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.bb1(x)
        bbs = out
        out = F.relu(self.bn1(self.bb2(out)))
        out = self.bn2(self.conv(out))
        out += self.shortcut(x)
        out = F.relu(out)
        # print(out.shape)
        return out, bbs

class Double_conv(nn.Module):

    def __init__(self, in_ch, out_ch, n):
        super(Double_conv, self).__init__()
        self.n_skip = n

        self.cont = nn.ConvTranspose2d(out_ch*2, out_ch, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(64*in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skips):

        x = self.cont(x)
        out = skips[0]
        for i in range(1, self.n_skip):
           skip = F.interpolate(input=skips[i], size=skips[0].size()[2:], mode='bilinear', align_corners=True)
           out = torch.cat((out, skip), dim=1)

        x = torch.cat((x, out), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class ULeafNet(nn.Module):
    global skip_values

    def __init__(self, num_channel=3, num_classes=3):
        super(ULeafNet, self).__init__()
        self.skip_values = []
        self.conv = nn.Conv2d(num_channel, 64, kernel_size=1, stride=1)
        self.enc1 = BasicBlock(64, 128)
        self.enc2 = BasicBlock(128, 256)
        self.enc3 = BasicBlock(256, 512)
        self.enc4 = BasicBlock(512, 1024)

        self.dec1 = Double_conv(38, 512, 4)
        self.dec2 = Double_conv(18, 256, 3)
        self.dec3 = Double_conv(8, 128, 2)
        self.dec4 = Double_conv(3, 64, 1)
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        # decoder
        x = self.dec1(x, [skip4, skip3, skip2, skip1])
        x = self.dec2(x, [skip3, skip2, skip1])
        x = self.dec3(x, [skip2, skip1])
        x = self.dec4(x, [skip1])
        x = self.out(x)
        return x


# inputs = torch.randn(2, 3, 256, 256).cuda()
# model = ULeafNet(num_channel=3, num_classes=3).cuda()
# from torchinfo import summary
# summary(model)
# result = model(inputs)
# print(result.shape)
