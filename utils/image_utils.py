#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import ImageFilter

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def mae(normal1, normal2):
    # normalization
    normal1 = torch.nn.functional.normalize(normal1, p=2, dim=0)
    normal2 = torch.nn.functional.normalize(normal2, p=2, dim=0)
    
    # compute mean angular error
    dot = torch.sum(normal1 * normal2, dim=0)
    dot = torch.clamp(dot, -1, 1)
    
    return torch.mean(torch.arccos(dot) * 180 / torch.pi) 

def get_edges(image):
    image_pil = transforms.ToPILImage()(image)
    image_gray = image_pil.convert('L')
    image_edges = image_gray.filter(ImageFilter.FIND_EDGES)
    image_edges_tensor = transforms.ToTensor()(image_edges)
    
    return image_edges_tensor.squeeze(0)

def gradient_map(image):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    
    grad_x = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])])
    grad_y = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])])
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = magnitude.norm(dim=0, keepdim=True)

    return magnitude

def colormap(map, cmap="turbo"):
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    map = (map - map.min()) / (map.max() - map.min())
    map = (map * 255).round().long().squeeze()
    map = colors[map].permute(2,0,1)
    return map

def render_net_image(render_pkg, render_items, render_mode, camera):
    output = render_items[render_mode].lower()
    if output == 'alpha':
        net_image = render_pkg["rend_alpha"]
    elif output == 'normal':
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
    elif output == 'depth':
        net_image = render_pkg["surf_depth"]
    elif output == 'edge':
        net_image = gradient_map(render_pkg["render"])
    elif output == 'curvature':
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
        net_image = gradient_map(net_image)
    else:
        net_image = render_pkg["render"]

    if net_image.shape[0]==1:
        net_image = colormap(net_image)
    return net_image


def read_exr(exr_path: str, with_alpha: bool = True) -> np.ndarray:
    """
    Read an OpenEXR file and return its contents as a numpy array.

    Args:
    exr_path: Path to the EXR file
    with_alpha: If True, also read the alpha channel if available

    Returns:
    numpy array with shape (height, width, channels)
    """
    import OpenEXR
    import Imath

    file = OpenEXR.InputFile(exr_path)
    dw = file.header()["dataWindow"]
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # Check if alpha channel exists
    header = file.header()
    channels_in_file = header['channels'].keys()
    has_alpha = 'A' in channels_in_file

    try:
        if with_alpha and has_alpha:
            channel_names = ["R", "G", "B", "A"]
            channels = file.channels(channel_names, FLOAT)
            pixels = np.zeros((size[1], size[0], 4))
        else:
            channel_names = ["R", "G", "B"]
            channels = file.channels(channel_names, FLOAT)
            pixels = np.zeros((size[1], size[0], 3))
    except:
        channel_names = ["V"]
        channels = file.channels(channel_names, FLOAT)
        pixels = np.zeros((size[1], size[0], 1))

    for i, channel in enumerate(channels):
        pixels[:, :, i] = np.frombuffer(channel, dtype=np.float32).reshape(
            size[1], size[0]
        )
    return pixels.astype(np.float32)

def write_exr(output_path: str, image: np.ndarray) -> None:
    """
    Writes a NumPy array to an OpenEXR file.

    Args:
        output_path: Path to the output EXR file.
        image: NumPy array to write, with shape (height, width, channels). It should be float32.
    """
    import OpenEXR
    import Imath
    
    height, width, channels = image.shape

    header = OpenEXR.Header(width, height)
    header['channels'] = {}

    if channels == 1:
        header['channels']['V'] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) # Single channel
        pixels = image[:, :, 0].astype(np.float32).tobytes()
        out = OpenEXR.OutputFile(output_path, header)
        out.writePixels({'V': pixels})
    elif channels == 3:
        header['channels']['R'] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        header['channels']['G'] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        header['channels']['B'] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))

        R = image[:, :, 0].astype(np.float32).tobytes()
        G = image[:, :, 1].astype(np.float32).tobytes()
        B = image[:, :, 2].astype(np.float32).tobytes()

        out = OpenEXR.OutputFile(output_path, header)
        out.writePixels({'R': R, 'G': G, 'B': B})
    elif channels == 4:
        header['channels']['R'] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        header['channels']['G'] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        header['channels']['B'] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        header['channels']['A'] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))

        R = image[:, :, 0].astype(np.float32).tobytes()
        G = image[:, :, 1].astype(np.float32).tobytes()
        B = image[:, :, 2].astype(np.float32).tobytes()
        A = image[:, :, 3].astype(np.float32).tobytes()

        out = OpenEXR.OutputFile(output_path, header)
        out.writePixels({'R': R, 'G': G, 'B': B, 'A': A})
    else:
        raise ValueError("Unsupported number of channels.  Must be 1, 3, or 4.")

    out.close()

# from https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/box_filter.py
def diff_x(x, r):
    assert x.dim() == 3

    left   = x[:,         r:2 * r + 1, :]
    middle = x[:, 2 * r + 1:         , :] - x[:,           :-2 * r - 1, :]
    right  = x[:,        -1:         , :] - x[:, -2 * r - 1:    -r - 1, :]

    output = torch.cat([left, middle, right], dim=-2)

    return output

def diff_y(x, r):
    assert x.dim() == 3

    left   = x[:, :,         r:2 * r + 1]
    middle = x[:, :, 2 * r + 1:         ] - x[:, :,           :-2 * r - 1]
    right  = x[:, :,        -1:         ] - x[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=-1)

    return output

def box_filter(image, kernel_size:int = 3):
    # image: [C, H, W]
    r = kernel_size // 2
    return diff_y(diff_x(image.cumsum(dim=-2), r).cumsum(dim=-1), r) / kernel_size**2

def box_filter_sr(image, kernel_size:int = 3):

    h, w = image.shape[-2:]
    size = min(h, w)
    sv_size = kernel_size * 12
    scale_factor = sv_size / size

    # surper resolution
    sv_image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode= 'bilinear'
    )[0]
    filter_sv_image = box_filter(sv_image, kernel_size)
    
    output = torch.nn.functional.interpolate(
        filter_sv_image[None], size=(h, w) , mode= 'bilinear'
    )[0]

    # direct_output = box_filter(image, kernel_size)

    # import matplotlib.pyplot as plt
    # plt.subplot(1, 4, 1)
    # plt.imshow(image[0].detach().cpu().numpy())
    # plt.subplot(1, 4, 2)
    # plt.imshow(filter_sv_image[0].detach().cpu().numpy())
    # plt.subplot(1, 4, 3)
    # plt.imshow(output[0].detach().cpu().numpy())
    # plt.subplot(1, 4, 4)
    # plt.imshow(direct_output[0].detach().cpu().numpy())
    # plt.show()
    # exit(-1)

    return output
