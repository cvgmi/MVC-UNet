import torch
import torch.nn as nn
from deepsphere.models.spherical_unet.unet_model import SphericalUNet

class VoxelConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super(VoxelConv).__init__()
        conv = nn.Conv3d(in_c, out_c, kernel_size=kernel_size, padding=1)

    def forward(self,x):
        # [batch, channels, rows, cols, EAP]
        x_s = x.shape

        shape_cube_root = int((x.shape[-1])**(1/3))
        x.view(x_s[0], -1, x_s[-1], shape_cube_root, shape_cube_root, shape_cube_root)
        x = conv(x)
        return x.view(x_s)


class SphereVoxelConv(nn.Module):
    def __init__(self, in_c=None, out_c=None, kernel_size=4):
        super().__init__()
        self.unet = SphericalUNet(pooling_class='healpix', N=768, depth=4, laplacian_type='combinatorial',
                                  kernel_size=kernel_size)
    def forward(self, x):
        # [batch, channels, rows, cols, ODF]
        x_s = x.shape

        # [batch, channels, rows*cols, ODF]
        x = x.view(x_s[0], x_s[1], -1, x_s[-1])

        # [batch, rows*cols, ODF, channels]
        x = x.permute(0, 2, 3, 1)

        # [batch*rows*cols, ODF, channels]
        x = x.view(-1, x_s[-1], x_s[1])

        # [batch*rows*cols, ODF, channels]
        x = self.unet(x)

        # [batch, rows, cols, ODF, 1]
        x = x.reshape(x_s[0], x_s[2], x_s[3], x_s[-1], 1)

        # [batch, 1, rows, cols, ODF]
        x = x.permute(0, 4, 1, 2, 3)
        

        return x

