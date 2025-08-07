import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )
        def deconv_block(in_c, out_c, norm=True):
            layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)
        # Encoder
        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.enc5 = conv_block(512, 512)
        self.enc6 = conv_block(512, 512)
        # Decoder (skip norm on last)
        self.dec6 = deconv_block(512, 512)
        self.dec5 = deconv_block(1024, 512)
        self.dec4 = deconv_block(1024, 256)
        self.dec3 = deconv_block(512, 128)
        self.dec2 = deconv_block(256, 64)
        self.dec1 = deconv_block(128, 3, norm=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        d6 = self.dec6(e6)
        d5 = self.dec5(torch.cat([d6, e5], 1))
        d4 = self.dec4(torch.cat([d5, e4], 1))
        d3 = self.dec3(torch.cat([d4, e3], 1))
        d2 = self.dec2(torch.cat([d3, e2], 1))
        d1 = self.dec1(torch.cat([d2, e1], 1))
        return self.tanh(d1)