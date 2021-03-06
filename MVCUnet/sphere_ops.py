import torch

# for references for many of the low level operations see:
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.95.9801&rep=rep1&type=pdf

# windows: [batch,window,N]
# B: [batch,N]
# out : [batch, window,N]
def SphereLog(windows, B):
    # [batch, windows, N]
    B_expa = B.unsqueeze(1)

    inner = torch.sum(windows*B_expa, dim=-1).unsqueeze(-1)
    inner = inner.clamp(0.00001, 0.999999)
    u = windows - inner*B_expa
    u_inner_sqrt = torch.norm(u, dim=-1).unsqueeze(-1).clone()

    u_inner_zero = u_inner_sqrt == 0
    u_inner_sqrt[u_inner_zero] = 1
    v = u*torch.acos(inner)/(u_inner_sqrt)
    # replace nan with 0. Nan shows up when base point = lift point.
    v[v!=v] = 0
    return v

# windows: [window,N]
# B: [batch,N]
# out : [batch, window,N]
def SphereExp(windows, B):
    # [batch, windows]
    w_norm = torch.norm(windows, dim=1).unsqueeze(-1)
    exp1 = torch.cos(w_norm)*B
    w_norm_ = torch.where(w_norm==0, torch.ones_like(w_norm), w_norm)
    exp2 = torch.sin(w_norm_)*windows/w_norm_

    return exp1+exp2

# windows: [batches, rows_reduced, cols_reduced, window, N]
# weights: [out_channels, in_channels*kern_size**2]
def tangentCombination(windows, weights):
    w_s = windows.shape
    # windows: [batches*rows_reduced*cols_reduced, window, 3, 3]
    windows = windows.view(-1,
                           windows.shape[3],
                           windows.shape[4])

    oc = weights.shape[0]

    # weights: [1, out_channels, in_channels*kern_size**2, 1]
    weights = weights.view(1, weights.shape[0], weights.shape[1], 1)
    
    B = windows[:, round(weights.shape[2] / 2.0), :]
    # B = torch.eye(3).unsqueeze(2).expand(3, 3, windows.shape[0]).permute(2, 0, 1).cuda()

    # lifted: [batches*rows_reduced*cols_reduced, window, N]
    lifted = SphereLog(windows, B)

    # lifted: [batches*rows_reduced*cols_reduced, oc, window, N]
    lifted = lifted.view(-1, lifted.shape[1], lifted.shape[-1])

    # lifted_combination: [batches*rows_reduced*cols_reduced*oc, N]
    # lifted_combination = torch.sum(lifted*weights, dim=2).view(-1, lifted.shape[-1])
    lifted_combination = []
    for i in range(weights.shape[1]):
        lifted_combination.append(torch.matmul(lifted.permute(0,2,1), weights[:,i,:,:]))

    lifted_combination = torch.stack(lifted_combination).permute(1,0,2,3).reshape(-1, lifted.shape[-1])

    B = B.reshape(-1, 1, B.shape[-1]).expand(-1, oc, -1).contiguous().view(-1, B.shape[-1])
    print(lifted_combination.shape)
    projected = SphereExp(lifted_combination, B)
    projected = projected.view(w_s[0], w_s[1], w_s[2], oc, projected.shape[-1])

    return projected.permute(0, 3, 1, 2, 4).contiguous()

# windows: [batches, rows_reduced, cols_reduced, window, N]
# weights: [3, out_channels, in_channels*kern_size**2]
def tangentCombinationVolterra(windows, weights, ReLU=False):
    w_s = windows.shape
    # windows: [batches*rows_reduced*cols_reduced, window, 3, 3]
    windows = windows.view(-1,
                           windows.shape[3],
                           windows.shape[4])

    oc = weights.shape[1]

    # weights: [1, out_channels, in_channels*kern_size**2, 1]
    weights = weights.view(1, 3, weights.shape[1], weights.shape[2], 1)
    
    B = windows[:, round(weights.shape[3] / 2.0), :]
    # B = torch.eye(3).unsqueeze(2).expand(3, 3, windows.shape[0]).permute(2, 0, 1).cuda()

    # lifted: [batches*rows_reduced*cols_reduced, window, N]
    lifted = SphereLog(windows, B)

    # lifted: [batches*rows_reduced*cols_reduced, oc, window, N]
    lifted = lifted.view(-1, lifted.shape[1], lifted.shape[-1])

    # lifted_combination: [batches*rows_reduced*cols_reduced*oc, N]
    # lifted_combination = torch.sum(lifted*weights, dim=2).view(-1, lifted.shape[-1])
    lifted_combination1 = []
    lifted_combination2 = []
    lifted_combination3 = []
    for i in range(weights.shape[2]):
        lifted_combination1.append(torch.matmul(lifted.permute(0,2,1), weights[:,0,i,:,:]))
        lifted_combination2.append(torch.matmul(lifted.permute(0,2,1), weights[:,1,i,:,:]))
        lifted_combination3.append(torch.matmul(lifted.permute(0,2,1), weights[:,2,i,:,:]))

    lifted_combination1 = torch.stack(lifted_combination1).permute(1,0,2,3).reshape(-1, lifted.shape[-1])
    lifted_combination2 = torch.stack(lifted_combination2).permute(1,0,2,3).reshape(-1, lifted.shape[-1])
    lifted_combination3 = torch.stack(lifted_combination3).permute(1,0,2,3).reshape(-1, lifted.shape[-1])
    
    lifted_combination = lifted_combination1+lifted_combination2*lifted_combination3

    if ReLU:
        lifted_combination = torch.nn.functional.relu(lifted_combination)

    B = B.reshape(-1, 1, B.shape[-1]).expand(-1, oc, -1).contiguous().view(-1, B.shape[-1])
    projected = SphereExp(lifted_combination, B)
    projected = projected.view(w_s[0], w_s[1], w_s[2], oc, projected.shape[-1])
    response = projected.permute(0, 3, 1, 2, 4).contiguous()

    return response
