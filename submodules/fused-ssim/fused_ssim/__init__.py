from typing import NamedTuple
import torch.nn as nn
import torch
from fused_ssim_cuda import fusedssim, fusedssim_backward

allowed_padding = ["same", "valid"]

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, window_size=11, padding="same", train=True):
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(C1, C2, img1, img2, window_size, train)

        if padding == "valid":
            pad = window_size // 2
            ssim_map = ssim_map[:, :, pad:-pad, pad:-pad]

        ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding
        ctx.window_size = window_size

        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        C1, C2, padding = ctx.C1, ctx.C2, ctx.padding
        window_size = ctx.window_size

        dL_dmap = opt_grad
        if padding == "valid":
            dL_dmap = torch.zeros_like(img1)
            pad = window_size // 2
            dL_dmap[:, :, pad:-pad, pad:-pad] = opt_grad
        grad = fusedssim_backward(C1, C2, img1, img2, window_size, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        return None, None, grad, None, None, None, None

def fused_ssim(img1, img2, window_size=11, padding="same", train=True, return_mean=True):
    assert window_size in [11, 9, 7, 5, 3], "window_size must be in [11, 9, 7, 5, 3]"

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2, window_size, padding, train)
    if return_mean:
        return ssim_map.mean()
    else:
        return ssim_map
