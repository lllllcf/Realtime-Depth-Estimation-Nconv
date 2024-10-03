import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from .utils import *
from torch.cuda.amp import custom_fwd, custom_bwd
from .nconv import *
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2

def save_depth(depth_data, path):
    depth_data_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
    depth_data_8bit = depth_data_normalized.astype(np.uint8)
    colored_depth = cv2.applyColorMap(depth_data_8bit, cv2.COLORMAP_INFERNO)
    cv2.imwrite(path, colored_depth)

def resize_batch_images(image_batch, target_height, target_width):
    resized_images = []
    for image in image_batch:
        # Resize each image in the batch
        resized_image = TF.resize(image, size=(target_height, target_width))
        resized_images.append(resized_image)
    
    # Stack all the resized images back into a single tensor
    return torch.stack(resized_images)

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
        return weight

class CSPN(nn.Module):
    """
    implementation of CSPN++
    """

    def __init__(self, in_channels, pt, norm_layer=nn.BatchNorm2d, act=nn.ReLU, eps=1e-6):
        super().__init__()
        self.pt = pt
        self.weight3x3 = GenKernel(in_channels, 3, norm_layer=norm_layer, act=act, eps=eps)
        self.weight5x5 = GenKernel(in_channels, 5, norm_layer=norm_layer, act=act, eps=eps)
        self.weight7x7 = GenKernel(in_channels, 7, norm_layer=norm_layer, act=act, eps=eps)
        self.convmask = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=nn.Sigmoid),
        )
        self.convck = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=functools.partial(nn.Softmax, dim=1)),
        )
        self.convct = nn.Sequential(
            Basic2d(in_channels + 3, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=functools.partial(nn.Softmax, dim=1)),
        )

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, fout, hn, h0):
        bpconvlocal = BpConvLocal.apply
        
        weight3x3 = self.weight3x3(fout)
        weight5x5 = self.weight5x5(fout)
        weight7x7 = self.weight7x7(fout)
        mask3x3, mask5x5, mask7x7 = torch.split(self.convmask(fout) * (h0 > 1e-3).float(), 1, dim=1)
        conf3x3, conf5x5, conf7x7 = torch.split(self.convck(fout), 1, dim=1)
        hn3x3 = hn5x5 = hn7x7 = hn
        hns = [hn, ]
        for i in range(self.pt):
            hn3x3 = (1. - mask3x3) * bpconvlocal(hn3x3, weight3x3) + mask3x3 * h0
            hn5x5 = (1. - mask5x5) * bpconvlocal(hn5x5, weight5x5) + mask5x5 * h0
            hn7x7 = (1. - mask7x7) * bpconvlocal(hn7x7, weight7x7) + mask7x7 * h0
            if i == self.pt // 2 - 1:
                hns.append(conf3x3 * hn3x3 + conf5x5 * hn5x5 + conf7x7 * hn7x7)
        hns.append(conf3x3 * hn3x3 + conf5x5 * hn5x5 + conf7x7 * hn7x7)
        hns = torch.cat(hns, dim=1)
        wt = self.convct(torch.cat([fout, hns], dim=1))
        hn = torch.sum(wt * hns, dim=1, keepdim=True)
        return hn

class BpConvLocal(Function):
    @staticmethod
    def forward(ctx, input, weight):
        assert input.is_contiguous()
        assert weight.is_contiguous()
        ctx.save_for_backward(input, weight)
        output = BpOps.Conv2dLocal_F(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input, grad_weight = BpOps.Conv2dLocal_B(input, weight, grad_output)
        return grad_input, grad_weight

class Ident(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, *args):
        return args[0]

class Net(nn.Module):
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

        pos_fn = "softplus"
        # pos_fn = None
        num_channels=4

        self.nconv1 = NConv2d(1, num_channels, (5,5), pos_fn, 'p', padding=(2, 2))
        self.nconv2 = NConv2d(num_channels, num_channels, (5,5), pos_fn, 'p', padding=(2, 2))
        self.nconv3 = NConv2d(num_channels, num_channels, (5,5), pos_fn, 'p', padding=(2, 2))
        
        self.nconv4 = NConv2d(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding=(1, 1))
        self.nconv5 = NConv2d(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding=(1, 1))
        self.nconv6 = NConv2d(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding=(0, 0))


        self.nconv65 = NConv2d(num_channels, num_channels, (3,3), pos_fn, 'k', padding=(1, 1))
        self.nconv7 = NConv2d(num_channels, 1, (1,1), pos_fn, 'k')


        # self.d = nn.Sequential(
        #   nn.Conv2d(1,16,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(16,16,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(16,16,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(16,16,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(16,16,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(16,16,3,1,1),
        #   nn.ReLU(),                                              
        # )#11,664 Params

               # RGB stream
        # self.rgb = nn.Sequential(
        #   nn.Conv2d(4,64,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(64,64,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(64,64,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(64,64,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(64,64,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(64,64,3,1,1),
        #   nn.ReLU(),                                            
        # )#186,624 Params

        # self.cspnrgb = nn.Sequential(
        #   nn.Conv2d(3,32,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(32,64,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(64,64,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(64,64,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(64,64,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(64,32,3,1,1),
        # #   nn.ReLU(),                                            
        # )#186,624 Params

        # self.frgb = nn.Sequential(
        #   nn.Conv2d(11,32,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(32,64,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(64,64,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(64,64,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(64,64,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(64,32,3,1,1),
        #   nn.ReLU(),                                            
        # )#186,624 Params

        # # Fusion stream
        # self.fuse = nn.Sequential(
        #   nn.Conv2d(80,64,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(64,64,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(64,64,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(64,32,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(32,32,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(32,32,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(32,1,1,1),
        # )# 156,704 Params

        # # Fusion stream
        # self.smallfuse = nn.Sequential(
        #   nn.Conv2d(48,64,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(64,64,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(64,64,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(64,32,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(32,32,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(32,32,3,1,1),
        #   nn.ReLU(),
        #   nn.Conv2d(32,16,1,1),
        # )# 156,704 Params


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
        
        
        xout, cout = self.nconv65(xout, cout)
        xout, cout = self.nconv7(xout, cout)

        # print("x8")
        # print(xout.shape) 

        dout = F.adaptive_avg_pool2d(xout, (480, 640))
        cout = F.adaptive_avg_pool2d(xout, (480, 640))

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
        return dout

class CNN(nn.Module):

    def __init__(self, pos_fn=None):
        super().__init__() 

        self.cspn = CSPN(32, pt=2 * (6 - 0))

        self.d_net = nvonvDNET()
        # checkpoint = torch.load("./checkpoints/nconv.pth.tar")
        # state_dict = checkpoint["state_dict"]
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     name = k[7:] if k.startswith("module.") else k
        #     new_state_dict[name] = v
        # self.d_net.load_state_dict(new_state_dict, strict=False)

        # self.d_net.to(device)

        # Disable Training for the unguided module
        # for p in self.d_net.parameters():            
        #     p.requires_grad=False
        
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

        self.rgb = nn.Sequential(
          nn.Conv2d(3,64,3,1,1),
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
               
            
    def forward(self, x0_rgb, x0_d, k): 

        c0 = (x0_d > 0.01).float() 
        
        # Depth Network
        xout_d, cout_d = self.d_net(x0_rgb, x0_d, k)

        # save_depth((xout_d[0, 0, :, :]).detach().cpu().numpy(), 'tmp/color_middle.png')

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

        rgb = self.rgb(x0_rgb)
        xout = self.cspn(rgb, xout, x0_d)
        
        return xout, cout_d



def BilateralMLP():
    net = Net(32)
    return net

def nvonvDNET():
    net = DNET(32)
    return net

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1,
                     padding=1, padding_mode='zeros', groups=1, bias=False, dilation=1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out