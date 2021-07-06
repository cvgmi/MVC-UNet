import torch
import torch.nn as nn
import torch.nn.functional as F

from MVCUnet.sphere_valued_conv import SphereValuedConv
from MVCUnet.sphere_valued_upsample import interpolator
from MVCUnet.voxel_conv import SphereVoxelConv

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DownSampleBlock, self).__init__()
        self.conv = SphereValuedConv(in_channels, out_channels, kernel_size, 1, zero_init=True)

    def forward(self, x):
        return self.conv(x)

class VoxelNet(nn.Module):
    def __init__(self, in_c, out_c):
        super(VoxelNet, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv1 = nn.Conv3d(in_c, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv3d(64, 16, 3, padding=1)
        self.conv5 = nn.Conv3d(16, out_c, 3, padding=1)

    def forward(self, x):
        # [batch, 1, rows, cols, EAP]
        x_s = x.shape
 
        shape_cube_root = int((x.shape[-1])**(1/3)+0.5)
        assert(shape_cube_root**3 == x_s[-1])

        # [batch, rows, cols, c, EAP]
        x = x.permute(0,2,3,1,4)
        # [batch*rows*cols, c, eap, eap, eap]
        x = x.reshape(-1, self.in_c, shape_cube_root, shape_cube_root, shape_cube_root)
        # apply series of conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)

        # [batch, rows, cols, c, EAP]
        x = x.view(x_s[0], x_s[2], x_s[3], self.out_c, x_s[-1])
        x = x.permute(0, 3, 1, 2, 4)
        #x_n = torch.norm(x, dim=-1)+0.001

        return x#/x_n[...,None]



class MVCUnet(nn.Module):
    def __init__(self):
        super(MVCUnet, self).__init__()
        kernel_size = 4
        layers = 3
        self.down1 = DownSampleBlock(1, 8, kernel_size) 
        self.down2 = DownSampleBlock(8, 16, kernel_size) 
        self.down3 = DownSampleBlock(16, 8, kernel_size) 
        self.unet = SphereVoxelConv()

    def forward(self, x):
        x = self.down1(x)
        x = interpolator(x, 25)
        x = self.down2(x)
        x = interpolator(x, 25)
        x = self.down3(x)
        x = interpolator(x, 25)
        x = self.unet(x)
        return x
