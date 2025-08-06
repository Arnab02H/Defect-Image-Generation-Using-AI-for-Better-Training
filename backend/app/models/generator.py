
## This Code is Wokring 
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, feature_maps=64, channels=3):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)
# ### Defect GAN final Model
# import torch.nn as nn
# import torch
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
#             return nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
#                 nn.BatchNorm2d(out_channels),
#                 nn.LeakyReLU(0.2, inplace=True)
#             )
#         def deconv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
#             return nn.Sequential(
#                 nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True)
#             )

#         # Encoder
#         self.enc1 = conv_block(1, 64)
#         self.enc2 = conv_block(64, 128)
#         self.enc3 = conv_block(128, 256)
#         self.enc4 = conv_block(256, 512)
#         self.enc5 = conv_block(512, 512)
#         self.enc6 = conv_block(512, 512)

#         # Decoder with skip connections
#         self.dec6 = deconv_block(512, 512)
#         self.dec5 = deconv_block(1024, 512)
#         self.dec4 = deconv_block(1024, 256)
#         self.dec3 = deconv_block(512, 128)
#         self.dec2 = deconv_block(256, 64)
#         self.dec1 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1)
#         self.tanh = nn.Tanh()

#     def forward(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(e1)
#         e3 = self.enc3(e2)
#         e4 = self.enc4(e3)
#         e5 = self.enc5(e4)
#         e6 = self.enc6(e5)

#         d6 = self.dec6(e6)
#         d5 = self.dec5(torch.cat([d6, e5], dim=1))
#         d4 = self.dec4(torch.cat([d5, e4], dim=1))
#         d3 = self.dec3(torch.cat([d4, e3], dim=1))
#         d2 = self.dec2(torch.cat([d3, e2], dim=1))
#         d1 = self.dec1(torch.cat([d2, e1], dim=1))
#         output = self.tanh(d1)
#         return output + x