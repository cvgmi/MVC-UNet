import torch
import torch.nn as nn

from sphere_ops import SphereLog, SphereExp


def tangent_relu(x):
    """
    x : [batches, channels, rows, cols, N]
    """
    x_s = x.shape
    # B: [batches, channels, N]
    B = x[:, :, round(x.shape[2] / 2.0), round(x.shape[2] / 2.0), :]

    x = x.view(-1, x.shape[2]*x.shape[3], x.shape[-1])
    B = B.view(-1, B.shape[-1])

    tangent_space = SphereLog(x, B)
    tangent_space = nn.F.relu(tangent_space)
    out = SphereExp(x, B)
    out = out.view(*x_s)

    return out
