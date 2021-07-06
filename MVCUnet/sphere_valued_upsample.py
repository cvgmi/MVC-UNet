import torch
import torch.nn as nn
import torch.nn.functional as F

def interpolator(x, target_size):
        x_s = x.shape
        #x : [batches, EAP, channels, rows, cols]
        x = x.permute(0,4,1,2,3).contiguous()
        #x : [batches, EAP*channels, rows, cols]
        x = x.view(x_s[0], -1, x_s[2], x_s[3])
        #x : [batches, EAP*channels, rows_up, cols_up]
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        #x : [batches, EAP,channels, rows_up, cols_up]
        x = x.view(x_s[0], x_s[-1], x_s[1], target_size, target_size)
        #x : [batches, channels, rows_up, cols_up, EAP]
        x = x.permute(0,2,3,4,1).contiguous()

        return x
