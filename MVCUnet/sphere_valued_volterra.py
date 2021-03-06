import torch
import torch.nn as nn

import MVCUnet.sphere_ops as sphere_ops

class SphereValuedVolterra(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, zero_init=False, ReLU=True):
        super(SphereValuedVolterra, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.ReLU = ReLU

        if zero_init:
            self.weight_matrix = torch.nn.Parameter(
                torch.zeros(
                    3,
                    out_channels,
                    (kernel_size**2) *
                    in_channels),
                requires_grad=True)
        else:
            self.weight_matrix = torch.nn.Parameter(
                torch.rand(
                    3,
                    out_channels,
                    (kernel_size**2) *
                    in_channels),
                requires_grad=True)
        
    # x: [batches, channels, rows, cols, N]
    def forward(self, x):
        # x: [batches, channels, rows, cols, N] ->
        #    [batches, channels, N, rows, cols]
        x = x.permute(0, 1, 4, 2, 3).contiguous()

        # x_windows: [batches, channels, N, rows_reduced, cols_reduced, window_x, window_y]
        x_windows = x.unfold(3, self.kernel_size, self.stride).contiguous()
        x_windows = x_windows.unfold(
            4, self.kernel_size, self.stride).contiguous()

        x_s = x_windows.shape
        #x_windows: [batches, channels, N  rows_reduced, cols_reduced, window]
        x_windows = x_windows.view(
            x_s[0], x_s[1], x_s[2], x_s[3], x_s[4], -1)

        #x_windows: [batches, rows_reduced, cols_reduced, window, channels, N]
        x_windows = x_windows.permute(0, 3, 4, 5, 1, 2).contiguous()

        x_s = x_windows.shape
        # x_windows: [batches, rows_reduced, cols_reduced, window*channels, N]
        x_windows = x_windows.view(
            x_s[0], x_s[1], x_s[2], -1, x_s[5]).contiguous()
        
        return sphere_ops.tangentCombinationVolterra(x_windows, self.weight_matrix, self.ReLU)
