import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = DoubleConv(3, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.u1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.u2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.u4 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.c1 = DoubleConv(1024, 512)
        self.c2 = DoubleConv(512, 256)
        self.c3 = DoubleConv(256, 128)
        self.c4 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))

        b = self.bottleneck(self.pool(d4))

        u1 = self.u1(b)
        c1 = self.c1(torch.cat([u1, d4], dim=1))

        u2 = self.u2(c1)
        c2 = self.c2(torch.cat([u2, d3], dim=1))

        u3 = self.u3(c2)
        c3 = self.c3(torch.cat([u3, d2], dim=1))

        u4 = self.u4(c3)
        c4 = self.c4(torch.cat([u4, d1], dim=1))

        return self.out(c4)   # NO SIGMOID (correct)
