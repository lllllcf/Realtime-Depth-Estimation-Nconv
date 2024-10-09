import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from .utils import *
from torch.cuda.amp import custom_fwd, custom_bwd
# from .nconv import *
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
import numpy as np
from scipy.stats import poisson
from scipy import signal
import torch.nn as nn

# Non-negativity enforcement class        
class EnforcePos(object):
    def __init__(self, pos_fn, name):
        self.name = name
        self.pos_fn = pos_fn


    @staticmethod
    def apply(module, name, pos_fn):
        fn = EnforcePos(pos_fn, name)
        
        module.register_forward_pre_hook(fn)                    

        return fn

    def __call__(self, module, inputs):
       if module.training:
            weight = getattr(module, self.name)
            weight.data = self._pos(weight).data
       else:
            pass

    def _pos(self, p):
        pos_fn = self.pos_fn.lower()
        if pos_fn == 'softmax':
            p_sz = p.size()
            p = p.view(p_sz[0],p_sz[1], -1)
            p = F.softmax(p, -1)
            return p.view(p_sz)
        elif pos_fn == 'exp':
            return torch.exp(p)
        elif pos_fn == 'softplus':
            return F.softplus(p, beta=10)
        elif pos_fn == 'sigmoid':
            return F.sigmoid(p)
        else:
            print('Undefined positive function!')
            return      

class GenKernel(nn.Module):
    def __init__(self, in_channels, pk, norm_layer=nn.BatchNorm2d, act=nn.ReLU, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.conv = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, pk * pk - 1, norm_layer=norm_layer, act=nn.Identity),
        )

    def forward(self, fout):
        weight = self.conv(fout)
        weight_sum = torch.sum(weight.abs(), dim=1, keepdim=True)
        weight = torch.div(weight, weight_sum + self.eps)
        weight_mid = 1 - torch.sum(weight, dim=1, keepdim=True)
        weight_pre, weight_post = torch.split(weight, [weight.shape[1] // 2, weight.shape[1] // 2], dim=1)
        weight = torch.cat([weight_pre, weight_mid, weight_post], dim=1).contiguous()
        return weight[:, weight.shape[1] // 2, :, :]

class SMALL_FINAL_2(nn.Module):

    def __init__(self, pos_fn=None):
        super().__init__() 

        self.step2 = SMALL_STEP2()
            
    def forward(self, x0_rgb0, x0_d0, x0_rgb1, x0_d1): 
        
        # Depth Network
        xout = self.step2(x0_rgb0, x0_d0, x0_rgb1, x0_d1) 

        xout[:, :, :45, :] = 0
        xout[:, :, -45:, :] = 0
        xout[:, :, :, :20] = 0
        
        return xout[0:1], xout[1:2]

class SMALL_STEP2(nn.Module):

    def __init__(self, pos_fn=None):
        super().__init__() 

        self.step1 = SMALL_STEP1()
        # checkpoint = torch.load("./checkpoints/epoch=100.checkpoint.pth.tar")
        # state_dict = checkpoint["state_dict"]

        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     name = k[7:] if k.startswith("module.") else k
        #     new_state_dict[name] = v
        # self.step1.load_state_dict(new_state_dict, strict=False)
        
        # # Disable Training for the unguided module
        # for p in self.step1.parameters():            
        #     p.requires_grad=False

        
        
        # U-Net
        self.layer_num = 64
        self.conv1 = nn.Conv2d(4, self.layer_num, (3,3), 2, 1, bias=True)
        self.conv2 = nn.Conv2d(self.layer_num, self.layer_num, (3,3), 2,1, bias=True)
        self.conv3 = nn.Conv2d(self.layer_num, self.layer_num, (3,3), 2, 1, bias=True)
        self.conv4 = nn.Conv2d(self.layer_num, self.layer_num, (3,3), 2, 1, bias=True)
        self.conv5 = nn.Conv2d(self.layer_num, self.layer_num, (3,3), 2, 1, bias=True)
                
        self.conv6 = nn.Conv2d(self.layer_num+self.layer_num, 64, (3,3), 1, 1, bias=True)
        self.conv7 = nn.Conv2d(64+self.layer_num, 64, (3,3), 1, 1, bias=True)
        self.conv8 = nn.Conv2d(64+self.layer_num, 32, (3,3), 1, 1, bias=True)
        self.conv9 = nn.Conv2d(32+self.layer_num, 32, (3,3), 1, 1, bias=True)
        self.conv10 = nn.Conv2d(32+1, 1, (3,3), 1, 1, bias=True)
            
    def forward(self, x0_rgb0, x0_d0, x0_rgb1, x0_d1): 

        x0_rgb = torch.cat((x0_rgb0, x0_rgb1), dim=0)
        
        # Depth Network
        xout_d = self.step1(x0_d0, x0_d1)

        # U-Net
        x1 = F.relu(self.conv1(torch.cat((xout_d, x0_rgb),1)))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))

        # Upsample 1 
        x5u = F.interpolate(x5, x4.size()[2:], mode='nearest')
        x6 = F.leaky_relu(self.conv6(torch.cat((x5u, x4),1)), 0.2)
        
        # Upsample 2
        x6u = F.interpolate(x6, x3.size()[2:], mode='nearest')
        x7 = F.leaky_relu(self.conv7(torch.cat((x6u, x3),1)), 0.2)
        
        # Upsample 3
        x7u = F.interpolate(x7, x2.size()[2:], mode='nearest')
        x8 = F.leaky_relu(self.conv8(torch.cat((x7u, x2),1)), 0.2)
        
        # Upsample 4
        x8u = F.interpolate(x8, x1.size()[2:], mode='nearest')
        x9 = F.leaky_relu(self.conv9(torch.cat((x8u, x1),1)), 0.2)
                
        # Upsample 5
        x9u = F.interpolate(x9, xout_d.size()[2:], mode='nearest')
        xout = F.leaky_relu(self.conv10(torch.cat((x9u, xout_d),1)), 0.2)

        # rgb = self.rgb(torch.cat((x0_rgb, cout_d), 1))
        # dout = self.cspn(rgb, xout, x0_d)

        # xout[:, :, :45, :] = x0_d[:, :, :45, :]
        # xout[:, :, -45:, :] = x0_d[:, :, -45:, :]

        xout[:, :, :45, :] = 0
        xout[:, :, -45:, :] = 0
        xout[:, :, :, :20] = 0
        
        return xout[0:1], xout[1:2]
        
        # return xout 



class LARGER_STEP1(nn.Module):
    def __init__(self):
        super().__init__() 

        self.d_net = DNET(32)
               
            
    def forward(self, x0_d0, x0_d1): 

        x0_d = torch.cat((x0_d0, x0_d1), dim=0)
        
        # Depth Network
        xout_d = self.d_net(x0_d)
        
        return xout_d



class LARGER_STEP2(nn.Module):

    def __init__(self, pos_fn=None):
        super().__init__() 

        self.step1 = LARGER_STEP1()
        # checkpoint = torch.load("./checkpoints/LARGER_STEP1.pth.tar")
        # state_dict = checkpoint["state_dict"]

        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     name = k[7:] if k.startswith("module.") else k
        #     new_state_dict[name] = v
        # self.step1.load_state_dict(new_state_dict, strict=False)
        
        # # Disable Training for the unguided module
        # for p in self.step1.parameters():            
        #     p.requires_grad=False

        
        
        # # U-Net
        # self.conv1 = nn.Conv2d(5, 80, (3,3), 2, 1, bias=True)
        # self.conv2 = nn.Conv2d(80, 80, (3,3), 2,1, bias=True)
        # self.conv3 = nn.Conv2d(80, 80, (3,3), 2, 1, bias=True)
        # self.conv4 = nn.Conv2d(80, 80, (3,3), 2, 1, bias=True)
        # self.conv5 = nn.Conv2d(80, 80, (3,3), 2, 1, bias=True)
                
        # self.conv6 = nn.Conv2d(80+80, 64, (3,3), 1, 1, bias=True)
        # self.conv7 = nn.Conv2d(64+80, 64, (3,3), 1, 1, bias=True)
        # self.conv8 = nn.Conv2d(64+80, 32, (3,3), 1, 1, bias=True)
        # self.conv9 = nn.Conv2d(32+80, 32, (3,3), 1, 1, bias=True)
        # self.conv10 = nn.Conv2d(32+1, 1, (3,3), 1, 1, bias=True)

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
        self.decoder4 = FusionBlock(8, 8, 1)
            
    def forward(self, rgb0, depth0, rgb1, depth1): 
        
        # # Depth Network
        sparse = self.step1(depth0, depth1)
        rgb = torch.cat((rgb0, rgb1), dim=0)
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
        dense5 = self.decoder0(rgb5, sparse5)
        dense4 = self.decoder1(rgb4, sparse4, dense5)
        dense3 = self.decoder2(rgb3, sparse3, dense4)
        dense2 = self.decoder3(rgb2, sparse2, dense3)
        dense1 = self.decoder4(rgb1, sparse1, dense2)

        dense1[:, :, :45, :] = 0
        dense1[:, :, -45:, :] = 0
        dense1[:, :, :, :20] = 0
        
        return dense1[0:1], dense1[1:2]

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

        
        self.fuse_conv2 = ConvBlock(sparse_channels, fusion_channels)
        self.fuse_conv3 = ConvBlock(fusion_channels, fusion_channels)

    def forward(self, rgb, sparse, dense):
        rgb_feat = self.rgb_conv(rgb)
        sparse_feat = self.sparse_conv(sparse)

        fused = torch.cat((rgb_feat, dense),1)
        fused = self.fuse_conv1(fused)
        
        fused = self.fuse_conv2(fused + sparse_feat)
        fused = self.fuse_conv3(fused)
        return fused

class Fusion0(nn.Module):
    def __init__(self, rgb_channels, sparse_channels, fusion_channels):
        super(Fusion0, self).__init__()
        self.rgb_conv = ConvBlock(rgb_channels, rgb_channels)
        self.sparse_conv = ConvBlock(sparse_channels, sparse_channels)

        self.fuse_conv2 = ConvBlock(sparse_channels, fusion_channels)
        self.fuse_conv3 = ConvBlock(fusion_channels, fusion_channels)

    def forward(self, rgb, sparse):
        rgb_feat = self.rgb_conv(rgb)
        sparse_feat = self.sparse_conv(sparse)
        
        fused = self.fuse_conv2(rgb_feat + sparse_feat)
        fused = self.fuse_conv3(fused)
        return fused



class STEP3(nn.Module):

    def __init__(self, pos_fn=None):
        super().__init__() 

        self.step2 = STEP2()
        checkpoint = torch.load("./checkpoints/step2.pth.tar")
        state_dict = checkpoint["state_dict"]

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        self.step2.load_state_dict(new_state_dict, strict=False)
        
        # Disable Training for the unguided module
        for p in self.step2.parameters():            
            p.requires_grad=False

        self.rgb = nn.Sequential(
          nn.Conv2d(4,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,32,3,1,1),
          nn.ReLU(),                                            
        )#186,624 Params

        self.cspn = CSPN(32, pt=2 * (6 - 0))
               
            
    def forward(self, x0_rgb, x0_d): 

        xout_d, cout_d = self.step2(x0_rgb, x0_d)

        rgb = self.rgb(torch.cat((x0_rgb, cout_d), 1))
        dout = self.cspn(rgb, xout_d, x0_d)
        
        return dout, cout_d 

class SIMPLE(nn.Module):

    def __init__(self, pos_fn=None):
        super().__init__() 

        self.d_net = DNET(32)
        
        # U-Net
        self.conv1 = nn.Conv2d(5, 80, (3,3), 2, 1, bias=True)
        self.conv2 = nn.Conv2d(80, 80, (3,3), 2,1, bias=True)
        self.conv3 = nn.Conv2d(80, 80, (3,3), 2, 1, bias=True)
        self.conv4 = nn.Conv2d(80, 80, (3,3), 2, 1, bias=True)
        self.conv5 = nn.Conv2d(80, 80, (3,3), 2, 1, bias=True)
                
        self.conv6 = nn.Conv2d(80+80, 64, (3,3), 1, 1, bias=True)
        self.conv7 = nn.Conv2d(64+80, 64, (3,3), 1, 1, bias=True)
        self.conv8 = nn.Conv2d(64+80, 32, (3,3), 1, 1, bias=True)
        self.conv9 = nn.Conv2d(32+80, 32, (3,3), 1, 1, bias=True)
        self.conv10 = nn.Conv2d(32+1, 1, (3,3), 1, 1, bias=True)

        # self.cspn = CSPN(32, pt=2 * (6 - 0))

        self.rgb = nn.Sequential(
          nn.Conv2d(4,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,32,3,1,1),
          nn.ReLU(),                                            
        )#186,624 Params
               
            
    def forward(self, x0_rgb, x0_d): 

        c0 = (x0_d > 0.01).float() 
        
        # Depth Network
        xout_d, cout_d = self.d_net(x0_d)

        # U-Net
        x1 = F.relu(self.conv1(torch.cat((xout_d, x0_rgb,cout_d),1)))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))

        # Upsample 1 
        x5u = F.interpolate(x5, x4.size()[2:], mode='nearest')
        x6 = F.leaky_relu(self.conv6(torch.cat((x5u, x4),1)), 0.2)
        
        # Upsample 2
        x6u = F.interpolate(x6, x3.size()[2:], mode='nearest')
        x7 = F.leaky_relu(self.conv7(torch.cat((x6u, x3),1)), 0.2)
        
        # Upsample 3
        x7u = F.interpolate(x7, x2.size()[2:], mode='nearest')
        x8 = F.leaky_relu(self.conv8(torch.cat((x7u, x2),1)), 0.2)
        
        # Upsample 4
        x8u = F.interpolate(x8, x1.size()[2:], mode='nearest')
        x9 = F.leaky_relu(self.conv9(torch.cat((x8u, x1),1)), 0.2)
                
        # Upsample 5
        x9u = F.interpolate(x9, x0_d.size()[2:], mode='nearest')
        xout = F.leaky_relu(self.conv10(torch.cat((x9u, x0_d),1)), 0.2)

        # rgb = self.rgb(torch.cat((x0_rgb, cout_d), 1))
        # dout = self.cspn(rgb, xout, x0_d)
        
        return xout, cout_d 

class NConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, pos_fn='softplus', init_method='k', stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=1, bias=True):
        
        # Call _ConvNd constructor
        super(NConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, output_padding=(0, 0), groups=groups, bias=bias, padding_mode='zeros')

        
        self.eps = 1e-20
        self.pos_fn = pos_fn
        self.init_method = init_method
        
        # Initialize weights and bias
        self.init_parameters()
        
        if self.pos_fn is not None :
            EnforcePos.apply(self, 'weight', pos_fn)

    def forward(self, data, conf):
        
        # Normalized Convolution
        denom = F.conv2d(conf, self.weight, None, self.stride,
                        self.padding, self.dilation, self.groups)        
        nomin = F.conv2d(data*conf, self.weight, None, self.stride,
                        self.padding, self.dilation, self.groups)        
        nconv = nomin / (denom+self.eps)
        
        
        # Add bias
        b = self.bias
        sz = b.size(0)
        b = b.view(1,sz,1,1)
        b = b.expand_as(nconv)
        nconv += b
        
        # Propagate confidence
        cout = denom
        sz = cout.size()
        cout = cout.view(sz[0], sz[1], -1)
        
        k = self.weight
        k_sz = k.size()
        k = k.view(k_sz[0], -1)
        s = torch.sum(k, dim=-1, keepdim=True)        

        cout = cout / s
        cout = cout.view(sz)
        
        return nconv, cout
    
    
    def init_parameters(self):
        # Init weights
        if self.init_method == 'x': # Xavier            
            torch.nn.init.xavier_uniform_(self.weight)
        elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.weight)
        elif self.init_method == 'p': # Poisson
            mu=self.kernel_size[0]/2 
            dist = poisson(mu)
            x = np.arange(0, self.kernel_size[0])
            y = np.expand_dims(dist.pmf(x),1)
            w = signal.convolve2d(y, y.transpose(), 'full')
            w = torch.Tensor(w).type_as(self.weight)
            w = torch.unsqueeze(w,0)
            w = torch.unsqueeze(w,1)
            w = w.repeat(self.out_channels, 1, 1, 1)
            w = w.repeat(1, self.in_channels, 1, 1)
            self.weight.data = w + torch.rand(w.shape)
            
        # Init bias
        self.bias = torch.nn.Parameter(torch.zeros(self.out_channels)+0.01)


class DNET(nn.Module):
    def __init__(self, out_ch):
        super().__init__()

        pos_fn = "softplus"
        # pos_fn = None
        num_channels=16

        self.nconv1 = NConv2d(1, num_channels, (5,5), pos_fn, 'p', padding=(2, 2))
        self.nconv2 = NConv2d(num_channels, num_channels, (5,5), pos_fn, 'p', padding=(2, 2))
        self.nconv3 = NConv2d(num_channels, num_channels, (5,5), pos_fn, 'p', padding=(2, 2))
        
        self.nconv4 = NConv2d(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding=(1, 1))
        self.nconv5 = NConv2d(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding=(1, 1))
        self.nconv6 = NConv2d(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding=(0, 0))

        self.nconv7 = NConv2d(num_channels, 1, (1,1), pos_fn, 'k')

    def forward(self, S):

        c0 = (S > 0.01).float()
        x0 = S

        x1, c1 = self.nconv1(x0, c0)
        x1, c1 = self.nconv2(x1, c1)
        # x1, c1 = self.nconv3(x1, c1) 

        # Downsample 1
        ds = 2
        c1_ds, _ = F.max_pool2d(c1, ds, ds, return_indices=True)
        x1_ds, _ = F.max_pool2d(x1, ds, ds, return_indices=True)
        # c1_ds /= 4
        # c1_ds[c1_ds > 0.7] = 0.99
        
        x2_ds, c2_ds = self.nconv2(x1_ds, c1_ds)        
        x2_ds, c2_ds = self.nconv3(x2_ds, c2_ds)
        
        # Downsample 2
        c2_dss, _ = F.max_pool2d(c2_ds, ds, ds, return_indices=True)
        x2_dss, _ = F.max_pool2d(x2_ds, ds, ds, return_indices=True)
        # c2_dss /= 4
        # c2_dss[c2_dss > 0.7] = 0.99

        x3_ds, c3_ds = self.nconv2(x2_dss, c2_dss)
        
        # Downsample 3
        c3_dss, _ = F.max_pool2d(c3_ds, ds, ds, return_indices=True)
        x3_dss, _ = F.max_pool2d(x3_ds, ds, ds, return_indices=True)
        # c3_dss /= 4 
        # c3_dss[c3_dss > 0.7] = 0.99

        x4_ds, c4_ds = self.nconv2(x3_dss, c3_dss)                


        # Upsample 1
        x4 = F.interpolate(x4_ds, c3_ds.size()[2:], mode='nearest') 
        c4 = F.interpolate(c4_ds, c3_ds.size()[2:], mode='nearest')   
        x34_ds, c34_ds = self.nconv4(torch.cat((x3_ds,x4), 1),  torch.cat((c3_ds,c4), 1))     

        # c34_ds[c34_ds > 0.6] = 0.99      
              
        
        # Upsample 2
        x34 = F.interpolate(x34_ds, c2_ds.size()[2:], mode='nearest') 
        c34 = F.interpolate(c34_ds, c2_ds.size()[2:], mode='nearest')
        # f34 = self.frgb(torch.cat((resize_batch_images(fout, c34.shape[2], c34.shape[3]), c34),1))
        # d34 = self.smallfuse(torch.cat((x2_ds,x34,f34), 1))

        x23_ds, c23_ds = self.nconv5(torch.cat((x2_ds,x34), 1), torch.cat((c2_ds,c34), 1))

        # c23_ds[c23_ds > 0.6] = 0.99  
        
        
        # Upsample 3
        x23 = F.interpolate(x23_ds, x0.size()[2:], mode='nearest') 
        c23 = F.interpolate(c23_ds, c0.size()[2:], mode='nearest') 
        # f23 = self.frgb(torch.cat((resize_batch_images(fout, c23.shape[2], c23.shape[3]), c23),1))
        # d23 = self.smallfuse(torch.cat((x23,x1,f23), 1))
        xout, cout = self.nconv6(torch.cat((x23,x1), 1), torch.cat((c23,c1), 1))

        
        xout, cout = self.nconv7(xout, cout)

        # a = zero_tensor = torch.zeros(cout.shape)
        # bound = 0.1
        # a[cout < bound] = 1.0 
        # a[cout > bound] = 0.0 

        # cout = a
        return xout[:, :, 1:481, 1:641]

        # S[0, 0, 0, 0] = x4_ds[0, 0, 0, 0]
        # S[0, 0, 0, 1] = c4_ds[0, 0, 0, 0]
        # return S, S

