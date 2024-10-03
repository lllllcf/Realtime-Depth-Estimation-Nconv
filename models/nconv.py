import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
import numpy as np
from scipy.stats import poisson
from scipy import signal
import torch.nn as nn
from .utils import *

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

class Ident(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, *args):
        return args[0]

class DNET(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        # self.level = 0
        # self.wpool = Ident()
        # self.dist = Dist(n=1) # NUM
        # self.prop = Prop(out_ch)

        # self.conv_img = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        # )

        # self.postprocess_img = UNet()
        # self.cspn = CSPN(out_ch, pt=2 * (6 - self.level))

        # pos_fn = "Softplus"
        pos_fn = None
        num_channels=8

        self.nconv1 = NConv2d(1, num_channels, (5,5), pos_fn, 'p', padding=(2, 2))
        self.nconv2 = NConv2d(num_channels, num_channels, (5,5), pos_fn, 'p', padding=(2, 2))
        self.nconv3 = NConv2d(num_channels, num_channels, (5,5), pos_fn, 'p', padding=(2, 2))
        
        self.nconv4 = NConv2d(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding=(1, 1))
        self.nconv5 = NConv2d(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding=(1, 1))
        self.nconv6 = NConv2d(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding=(0, 0))
        
        self.nconv7 = NConv2d(num_channels, num_channels, (3,3), pos_fn, 'k', padding=(1, 1))
        self.nconv7 = NConv2d(num_channels, 1, (1,1), pos_fn, 'k')

    #     self.d = nn.Sequential(
    #       nn.Conv2d(1,16,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(16,16,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(16,16,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(16,16,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(16,16,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(16,16,3,1,1),
    #       nn.ReLU(),                                              
    #     )#11,664 Params

    #            # RGB stream
    #     self.rgb = nn.Sequential(
    #       nn.Conv2d(4,64,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(64,64,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(64,64,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(64,64,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(64,64,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(64,64,3,1,1),
    #       nn.ReLU(),                                            
    #     )#186,624 Params

    #     self.cspnrgb = nn.Sequential(
    #       nn.Conv2d(3,32,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(32,64,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(64,64,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(64,64,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(64,64,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(64,32,3,1,1),
    #     #   nn.ReLU(),                                            
    #     )#186,624 Params

    #     self.frgb = nn.Sequential(
    #       nn.Conv2d(11,32,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(32,64,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(64,64,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(64,64,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(64,64,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(64,32,3,1,1),
    #       nn.ReLU(),                                            
    #     )#186,624 Params

    #     # Fusion stream
    #     self.fuse = nn.Sequential(
    #       nn.Conv2d(80,64,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(64,64,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(64,64,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(64,32,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(32,32,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(32,32,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(32,1,1,1),
    #     )# 156,704 Params

    #     # Fusion stream
    #     self.smallfuse = nn.Sequential(
    #       nn.Conv2d(48,64,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(64,64,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(64,64,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(64,32,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(32,32,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(32,32,3,1,1),
    #       nn.ReLU(),
    #       nn.Conv2d(32,16,1,1),
    #     )# 156,704 Params


    # def pinv(self, S, K, xx, yy):
    #     fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2, 1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
    #     S = S.view(S.shape[0], 1, -1)
    #     xx = xx.reshape(1, 1, -1)
    #     yy = yy.reshape(1, 1, -1)
    #     Px = S * (xx - cx) / fx
    #     Py = S * (yy - cy) / fy
    #     Pz = S
    #     Pxyz = torch.cat([Px, Py, Pz], dim=1).contiguous()
    #     return Pxyz

    def forward(self, fout, S, K):

        c0 = (S > 0.01).float()
        x0 = S

        x1, c1 = self.nconv1(x0, c0)
        x1, c1 = self.nconv2(x1, c1)
        x1, c1 = self.nconv3(x1, c1) 

        # print("x1")
        # print(x1.shape)

        # Downsample 1
        ds = 2 
        c1_ds, idx = F.max_pool2d(c1, ds, ds, return_indices=True)
        x1_ds = torch.zeros(c1_ds.size()).cuda()
        for i in range(x1_ds.size(0)):
            for j in range(x1_ds.size(1)):
                x1_ds[i,j,:,:] = x1[i,j,:,:].view(-1)[idx[i,j,:,:].view(-1)].view(idx.size()[2:])
        c1_ds /= 4
        
        x2_ds, c2_ds = self.nconv2(x1_ds, c1_ds)        
        x2_ds, c2_ds = self.nconv3(x2_ds, c2_ds)
        
        # print("x2")
        # print(x2_ds.shape)
        
        # Downsample 2
        ds = 2 
        c2_dss, idx = F.max_pool2d(c2_ds, ds, ds, return_indices=True)
        
        x2_dss = torch.zeros(c2_dss.size()).cuda()
        for i in range(x2_dss.size(0)):
            for j in range(x2_dss.size(1)):
                x2_dss[i,j,:,:] = x2_ds[i,j,:,:].view(-1)[idx[i,j,:,:].view(-1)].view(idx.size()[2:])
        c2_dss /= 4        

        x3_ds, c3_ds = self.nconv2(x2_dss, c2_dss)

        # print("x3")
        # print(x3_ds.shape)
        
        
        # Downsample 3
        ds = 2 
        c3_dss, idx = F.max_pool2d(c3_ds, ds, ds, return_indices=True)
        
        x3_dss = torch.zeros(c3_dss.size()).cuda()
        for i in range(x3_dss.size(0)):
            for j in range(x3_dss.size(1)):
                x3_dss[i,j,:,:] = x3_ds[i,j,:,:].view(-1)[idx[i,j,:,:].view(-1)].view(idx.size()[2:])
        c3_dss /= 4        
        x4_ds, c4_ds = self.nconv2(x3_dss, c3_dss)    

        # print("x4")
        # print(x4_ds.shape)            


        # Upsample 1
        x4 = F.interpolate(x4_ds, c3_ds.size()[2:], mode='nearest') 
        c4 = F.interpolate(c4_ds, c3_ds.size()[2:], mode='nearest')  
        # print("c4")  
        # print(c4.shape)   
        x34_ds, c34_ds = self.nconv4(torch.cat((x3_ds,x4), 1),  torch.cat((c3_ds,c4), 1))                
         
         

        # f4 = self.frgb(torch.cat((resize_batch_images(fout, c4.shape[2], c4.shape[3]), c4),1))
        # d4 = self.smallfuse(torch.cat((x3_ds,x4,f4), 1))

        # x34_ds, c34_ds = self.nconv4(d4,  torch.cat((c3_ds,c4), 1))       

        # print("x5") # 126 160
        # print(x34_ds.shape)         
        
        # Upsample 2
        x34 = F.interpolate(x34_ds, c2_ds.size()[2:], mode='nearest') 
        c34 = F.interpolate(c34_ds, c2_ds.size()[2:], mode='nearest')
        # f34 = self.frgb(torch.cat((resize_batch_images(fout, c34.shape[2], c34.shape[3]), c34),1))
        # d34 = self.smallfuse(torch.cat((x2_ds,x34,f34), 1))

        x23_ds, c23_ds = self.nconv5(torch.cat((x2_ds,x34), 1), torch.cat((c2_ds,c34), 1))
        
        # x23_ds, c23_ds = self.nconv5(d34, torch.cat((c2_ds,c34), 1))

        # print("x6") # 240 320
        # print(x23_ds.shape)       
        
        
        # Upsample 3
        x23 = F.interpolate(x23_ds, x0.size()[2:], mode='nearest') 
        c23 = F.interpolate(c23_ds, c0.size()[2:], mode='nearest') 
        # f23 = self.frgb(torch.cat((resize_batch_images(fout, c23.shape[2], c23.shape[3]), c23),1))
        # d23 = self.smallfuse(torch.cat((x23,x1,f23), 1))
        xout, cout = self.nconv6(torch.cat((x23,x1), 1), torch.cat((c23,c1), 1))
        
        # xout, cout = self.nconv6(d23, torch.cat((c23,c1), 1))

        # print("x7") # 478 638
        # print(xout.shape)   
        
        
        
        xout, cout = self.nconv7(xout, cout)

        # print("x8")
        # print(xout.shape) 

        ################################################################

        # dout = self.d(dout)
                
        # # # RGB network
        # rgb = self.rgb(torch.cat((fout, cout),1))
        
        # # # Fusion Network
        # dout = self.fuse(torch.cat((rgb, dout),1))

        # ################################################################

        # # fout = self.conv_img(fout)

        # Sp = self.wpool(S, fout)
        # # Kp = K.clone()
        # # Kp[:, :2] = Kp[:, :2] / 2 ** self.level
        # # B, _, height, width = Sp.shape
        # # xx, yy = torch.meshgrid(torch.arange(width, device=Sp.device), torch.arange(height, device=Sp.device),
        # #                         indexing='xy')
        # # ###############################################################
        # # Pxyz = self.pinv(Sp, Kp, xx, yy)
        # # # Ofnum, args = self.dist(Sp, xx, yy)
        # # Ofnum, args = self.dist(Sp)
        # # dout = self.prop(fout, Pxyz, Ofnum, args)
        # # dout = Sp
        # # dout = self.postprocess_img(dout)
        # ###############################################################
        # # S_clone = S.clone()
        # # valid = (S_clone < 0.01)
        # # S_clone[valid] = dout[valid]
        # # dout = S_clone
        # ###############################################################
        # # print("fout")
        # # print(fout.shape)
        # # print("dout")
        # # print(dout.shape)
        # # print("Sp")
        # # print(Sp.shape)
        # fout = self.cspnrgb(fout)
        # # dout = self.cspn(fout, dout, Sp)
        # ###############################################################

        # dout = self.postprocess_img(dout)
        return xout[:, :, 1:481, 1:641], cout[:, :, 1:481, 1:641]