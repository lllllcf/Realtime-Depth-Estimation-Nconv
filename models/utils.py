from copy import deepcopy
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import math
import numpy as np
from collections import OrderedDict
import BpOps
import torch.distributed as dist
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath
from torch.cuda.amp import custom_fwd, custom_bwd
import functools
from datetime import datetime, timedelta

class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1, padding_mode='zeros',
                 act=nn.ReLU, stride=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=False, padding_mode=padding_mode)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=True, padding_mode=padding_mode)
        self.conv = nn.Sequential(OrderedDict([('conv', conv)]))
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', act())

    def forward(self, x):
        out = self.conv(x)
        return out

class Coef(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels=3, kernel_size=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=1, padding=padding, bias=True)

    def forward(self, x):
        feat = self.conv(x)
        XF, XB, XW = torch.split(feat, [1, 1, 1], dim=1)
        return XF, XB, XW

class BpDist(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, xy, idx, valid, n, H, W):
        """
        """

        assert xy.is_contiguous()
        assert valid.is_contiguous()
        _, Cc, M = xy.shape
        B = valid.shape[0]
        N = H * W           # If everything is alright, N == M
        args = torch.zeros((B, n, N), dtype=torch.long, device=xy.device).contiguous()
        IPCnum = torch.zeros((B, Cc, n, N), dtype=xy.dtype, device=xy.device).contiguous()
        for b in range(B):
            #Pc = torch.masked_select(xy, valid[b:b + 1]).reshape(1, 2, -1)
            #Pc = xy.reshape(1,2,-1)
            #print("Pc shape before",Pc.shape)
            #print("Yooooo",valid.shape)
            Pc = torch.where(valid[b:b+1], torch.ones((1,valid.shape[1], valid.shape[2]),device = xy.device), torch.zeros((1,valid.shape[1], valid.shape[2]), device = xy.device)).reshape(1,2,-1)
            Pc = torch.cat([Pc,Pc],dim=1)
            #print("Shapy shape", Pc.shape, H, W, H*W, IPCnum.shape, args.shape)
            #print(args[0][0])
            # print(3)
            BpOps.Dist(Pc, IPCnum[b:b + 1], args[b:b + 1], H, W)
            # print(4)
            #args[b:b+1] = IPCnum[b:b+1,0] + IPCnum[b:b+1,1] * W
            #print(args[b:b+1], H , W, H*W, idx.shape, args[b:b+1].max())
            #idx_valid = torch.masked_select(idx, valid[b:b + 1].view(1, 1, N))
            #args[b:b + 1] = torch.index_select(idx, 0, args[b:b + 1].reshape(-1)).reshape(1, n, N)
        #args2 = torch.zeros((B, n, N), dtype=torch.long, device=xy.device).contiguous()
        return IPCnum, args   # IPCnum (B,2,n,N) has the coordinates of the n neighbours, args (B,n,N) has the idx of the coordinates ?
    @staticmethod
    @custom_bwd
    def backward(ctx, ga=None, gb=None):
        return None, None, None, None

# class BpDist(Function):

#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float32)
#     def forward(ctx, xy, idx, Valid, num, H, W):
#         """
#         """
#         assert xy.is_contiguous()
#         assert Valid.is_contiguous()
#         _, Cc, M = xy.shape
#         B = Valid.shape[0]
#         N = H * W
#         args = torch.zeros((B, num, N), dtype=torch.long, device=xy.device)
#         IPCnum = torch.zeros((B, Cc, num, N), dtype=xy.dtype, device=xy.device)
#         for b in range(B):
#             Pc = torch.masked_select(xy, Valid[b:b + 1].view(1, 1, N)).reshape(1, 2, -1)
#             BpOps.Dist(Pc, IPCnum[b:b + 1], args[b:b + 1], H, W)
#             idx_valid = torch.masked_select(idx, Valid[b:b + 1].view(1, 1, N))
#             args[b:b + 1] = torch.index_select(idx_valid, 0, args[b:b + 1].reshape(-1)).reshape(1, num, N)
#         return IPCnum, args

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, ga=None, gb=None):
#         return None, None, None, None

bpdist = BpDist.apply

class Dist(nn.Module):
    """ Get the n closest neighbours """
    def __init__(self, n, vThresh = 1e-3):
        super().__init__()
        self.n = n
        self.validThresh = vThresh
    def forward(self, S):
        """ S shape : B,1,H,W """
        n = self.n
        B, _, height, width = S.shape
        N = height * width
        S = S.reshape(B, 1, N)
        xx, yy = torch.meshgrid(torch.arange(width, device=S.device), torch.arange(height, device=S.device),indexing='xy')
        Valid = (S > self.validThresh)
        #print(Valid.shape,torch.sum(torch.where(Valid,torch.ones_like(Valid),torch.zeros_like(Valid))))
        xy = torch.stack((xx, yy), axis=0).reshape(1, 2, -1).float()
        idx = torch.arange(N, device=S.device).reshape(1, 1, N)
        # print(1)
        Ofnum, args = bpdist(xy, idx, Valid, n, height, width)
        # print(2)
        #print(Ofnum.shape,Ofnum[0,0,0:4,1000:1010],Ofnum.max())
        return Ofnum, args

# class Dist(nn.Module):
#     """
#     """

#     def __init__(self, num):
#         super().__init__()
#         """
#         """
#         self.num = num

#     def forward(self, S, xx, yy):
#         """
#         """
#         num = self.num
#         B, _, height, width = S.shape
#         N = height * width
#         S = S.reshape(B, 1, N)
#         Valid = (S > 1e-3)
#         xy = torch.stack((xx, yy), axis=0).reshape(1, 2, -1).float()
#         idx = torch.arange(N, device=S.device).reshape(1, 1, N)
#         Ofnum, args = bpdist(xy, idx, Valid, num, height, width)
#         return Ofnum, args

class Prop(nn.Module):
    """
    """

    def __init__(self, Cfi, Cfp=3, Cfo=2, act=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        """
        """
        self.dist = lambda x: (x * x).sum(1)
        Ct = Cfo + Cfi + Cfi + Cfp
        self.convXF = nn.Sequential(
            Basic2d(in_channels=Ct, out_channels=128, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=128, out_channels=128, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=128, out_channels=128, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=128, out_channels=64, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=64, out_channels=Cfi, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
        )
        self.convXL = nn.Sequential(
            Basic2d(in_channels=Cfi, out_channels=64, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=64, out_channels=64, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=64, out_channels=64, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=64, out_channels=Cfi, norm_layer=norm_layer, act=nn.Identity, kernel_size=1,
                    padding=0),
        )
        self.act = act()
        self.coef = Coef(Cfi, 3)

    def forward(self, If, Pf, Ofnum, args):
        """
        """
        # print(If.shape)
        num = args.shape[-2]
        B, Cfi, H, W = If.shape
        N = H * W
        B, Cfp, M = Pf.shape
        If = If.view(B, Cfi, 1, N)
        Pf = Pf.view(B, Cfp, 1, M)
        Ifnum = If.expand(B, Cfi, num, N)  ## Ifnum is BxCfixnumxN
        IPfnum = torch.gather(
            input=If.expand(B, Cfi, num, N),
            dim=-1,
            index=args.view(B, 1, num, N).expand(B, Cfi, num, N))  ## IPfnum is BxCfixnumxN
        Pfnum = torch.gather(
            input=Pf.expand(B, Cfp, num, M),
            dim=-1,
            index=args.view(B, 1, num, N).expand(B, Cfp, num, N))  ## Pfnum is BxCfpxnumxN
        X = torch.cat([Ifnum, IPfnum, Pfnum, Ofnum], dim=1)
        XF = self.convXF(X)
        XF = self.act(XF + self.convXL(XF))
        Alpha, Beta, Omega = self.coef(XF)
        Omega = torch.softmax(Omega, dim=2)
        dout = torch.sum(((Alpha + 1) * Pfnum[:, -1:] + Beta) * Omega, dim=2, keepdim=True)
        return dout.view(B, 1, H, W)