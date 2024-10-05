import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlockBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))



class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        return self.conv2(x)


class FusionBlock(nn.Module):
    def __init__(self, rgb_channels, sparse_channels, fusion_channels):
        super(FusionBlock, self).__init__()
        self.rgb_conv = ConvBlock(rgb_channels, rgb_channels)
        self.sparse_conv = ConvBlock(sparse_channels, sparse_channels)
        self.fuse_conv1 = ConvBlock(rgb_channels * 2, sparse_channels)

        
        self.fuse_conv2 = ConvBlock(sparse_channels, sparse_channels)
        self.fuse_conv3 = ConvBlock(sparse_channels, sparse_channels)
        self.fuse_conv4 = ConvBlock(sparse_channels, fusion_channels)
        self.fuse_conv5 = ConvBlock(fusion_channels, fusion_channels)
        self.fuse_conv6 = ConvBlock(fusion_channels, fusion_channels)

    def forward(self, rgb, sparse, dense):
        rgb_feat = self.rgb_conv(rgb)
        sparse_feat = self.sparse_conv(sparse)

        fused = torch.cat((rgb_feat, dense),1)
        fused = self.fuse_conv1(fused)
        
        fused = self.fuse_conv2(fused + sparse_feat)
        fused = self.fuse_conv3(fused)
        fused = self.fuse_conv4(fused)
        fused = self.fuse_conv5(fused)
        fused = self.fuse_conv6(fused)
        return fused

class FusionBlockFinal(nn.Module):
    def __init__(self, rgb_channels, sparse_channels, fusion_channels):
        super(FusionBlockFinal, self).__init__()
        self.rgb_conv = ConvBlock(rgb_channels, rgb_channels)
        self.sparse_conv = ConvBlock(sparse_channels, sparse_channels)
        self.fuse_conv1 = ConvBlock(rgb_channels * 2, sparse_channels)

        
        self.fuse_conv2 = ConvBlock(sparse_channels, sparse_channels)
        self.fuse_conv3 = ConvBlock(sparse_channels, sparse_channels)
        self.fuse_conv4 = ConvBlock(sparse_channels, sparse_channels)
        self.fuse_conv5 = ConvBlock(sparse_channels, fusion_channels)
        self.fuse_conv6 = ConvBlock(fusion_channels, fusion_channels)

    def forward(self, rgb, sparse, dense):
        rgb_feat = self.rgb_conv(rgb)
        sparse_feat = self.sparse_conv(sparse)

        fused = torch.cat((rgb_feat, dense),1)
        fused = self.fuse_conv1(fused)
        
        fused = self.fuse_conv2(fused + sparse_feat)
        fused = self.fuse_conv3(fused)
        fused = self.fuse_conv4(fused)
        fused = self.fuse_conv5(fused)
        fused = self.fuse_conv6(fused)
        return fused

class Fusion0(nn.Module):
    def __init__(self, rgb_channels, sparse_channels, fusion_channels):
        super(Fusion0, self).__init__()
        self.rgb_conv = ConvBlock(rgb_channels, rgb_channels)
        self.sparse_conv = ConvBlock(sparse_channels, sparse_channels)

        self.fuse_conv2 = ConvBlock(sparse_channels, sparse_channels)
        self.fuse_conv3 = ConvBlock(sparse_channels, fusion_channels)
        self.fuse_conv4 = ConvBlock(fusion_channels, fusion_channels)

    def forward(self, rgb, sparse):
        rgb_feat = self.rgb_conv(rgb)
        sparse_feat = self.sparse_conv(sparse)
        
        fused = self.fuse_conv2(rgb_feat + sparse_feat)
        fused = self.fuse_conv3(fused)
        fused = self.fuse_conv4(fused)
        return fused

class MASK_NET(nn.Module):
    def __init__(self):
        super(MASK_NET, self).__init__()
        
        # Encoder for RGB/Normal and Sparse Inputs
        self.rgb_encoder1 = ConvBlock(3, 8)
        self.rgb_encoder2 = ConvBlock(8, 16)
        self.rgb_encoder3 = ConvBlock(16, 32)
        self.rgb_encoder4 = ConvBlock(32, 64)
        self.rgb_encoder5 = ConvBlock(64, 128)

        self.sparse_encoder1 = ConvBlock(1, 8)
        self.sparse_encoder2 = ConvBlock(8, 16)
        self.sparse_encoder3 = ConvBlock(16, 32)
        self.sparse_encoder4 = ConvBlock(32, 64)
        self.sparse_encoder5 = ConvBlock(64, 128)
        
        # Decoder blocks with fusion
        self.decoder0 = Fusion0(128, 128, 64)
        self.decoder1 = FusionBlock(64, 64, 32)
        self.decoder2 = FusionBlock(32, 32, 16)
        self.decoder3 = FusionBlock(16, 16, 8)
        self.decoder4 = FusionBlockFinal(8, 8, 1)

    def forward(self, rgb, sparse):
        # Encoder for RGB
        rgb1 = self.rgb_encoder1(rgb)
        rgb2 = self.rgb_encoder2(rgb1)
        rgb3 = self.rgb_encoder3(rgb2)
        rgb4 = self.rgb_encoder4(rgb3)
        rgb5 = self.rgb_encoder5(rgb4)
        
        # Encoder for Sparse Depth
        sparse1 = self.sparse_encoder1(sparse)
        sparse2 = self.sparse_encoder2(sparse1)
        sparse3 = self.sparse_encoder3(sparse2)
        sparse4 = self.sparse_encoder4(sparse3)
        sparse5 = self.sparse_encoder5(sparse4)

        # Decoder with fusion
        dense5 = self.decoder0(rgb5, rgb5)
        dense4 = self.decoder1(rgb4, rgb4, dense5)
        dense3 = self.decoder2(rgb3, rgb3, dense4)
        dense2 = self.decoder3(rgb2, rgb2, dense3)
        dense1 = self.decoder4(rgb1, rgb1, dense2)

        

        return dense1