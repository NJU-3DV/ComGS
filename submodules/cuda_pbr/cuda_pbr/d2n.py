


import torch
from . import _C
from torch import Tensor
import torch.nn.functional as F
from jaxtyping import Float, jaxtyped
from typeguard import typechecked as typechecker


@jaxtyped
@typechecker
def depth2normal(
    intr: Float[Tensor, "4"],
    Rc2w: Float[Tensor, "3 3"],
    depth: Float[Tensor, "H W"],
    cuda_backend: bool=True
):

    if cuda_backend:
        normal = _Depth2Normal.apply(intr, Rc2w, depth)
    else:
        normal = _d2n_pytorch_impl(intr, Rc2w, depth)

    return normal

class _Depth2Normal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, intr, Rc2w, depth):

        normal, sxyz = _C.depth2normal(intr, Rc2w, depth)

        ctx.save_for_backward(intr, Rc2w, depth, normal, sxyz)

        return normal

    @staticmethod
    def backward(ctx, grad_normal, *_):

        intr, Rc2w, depth, normal, sxyz = ctx.saved_tensors

        grad_depth = _C.depth2normal_backward(intr, Rc2w, depth, sxyz, normal, grad_normal)

        grads = (
            None,
            None,
            grad_depth
        )
        return grads

EPS = 1e-10

def depth2xyz(intr, depths):
    """
    Computes the surface XYZ coordinates from depth values and camera intrinsics.
    
    Args:
        intr (torch.Tensor): A 1D tensor of camera intrinsic parameters [fx, fy, cx, cy].
        depths (torch.Tensor): A 2D tensor of depth values with shape (H, W).
        W (int): The width of the image.
        H (int): The height of the image.
        is_camera_perspective (bool): Whether the camera is perspective or orthographic.
        
    Returns:
        torch.Tensor: A 3D tensor with shape (3, H, W) representing the surface XYZ coordinates.
    """

    # Create a grid of pixel coordinates
    H, W = depths.shape
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=intr.device), 
        torch.arange(W, dtype=torch.float32, device=intr.device), 
        indexing='ij')
    
    # Initialize the surface_xyz
    surface_xyz = []

    x_component = (x_coords + 0.5 - intr[2]) / intr[0]
    y_component = (y_coords + 0.5 - intr[3]) / intr[1]

    # Compute the surface XYZ coordinates
    surface_xyz.append(x_component * depths)
    surface_xyz.append(y_component * depths)

    surface_xyz.append(depths)
    surface_xyz = torch.stack(surface_xyz, dim=0)

    return surface_xyz

def xyz2normal(surface_xyz):
    """
    Computes pseudo normals from the surface XYZ coordinates and camera intrinsics.
    
    Args:
        surface_xyz (torch.Tensor): A 3D tensor of surface XYZ coordinates with shape (3, H, W).
        W (int): The width of the image.
        H (int): The height of the image.
        
    Returns:
        torch.Tensor: A 3D tensor with shape (3, H, W) representing the normals.
    """
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], 
                            device=surface_xyz.device) / 8.0
    
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], 
                            device=surface_xyz.device) / 8.0
    
    # Reshape filters to match the convolution requirements
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
    
    # Manually pad the input using 'replicate' padding mode
    surface_xyz_padded = F.pad(surface_xyz.unsqueeze(0), (1, 1, 1, 1), mode='replicate')

    # Compute gradients along x and y directions for each channel
    gradient_a = F.conv2d(surface_xyz_padded, sobel_x, groups=3).squeeze(0)
    gradient_b = F.conv2d(surface_xyz_padded, sobel_y, groups=3).squeeze(0)

    # Compute the cross product of gradients to get the normal vector
    normals = torch.cross(gradient_a, gradient_b, dim=0)

    # Normalize the normals
    norm = torch.norm(normals, p=2, dim=0, keepdim=True)

    normals = torch.where(norm >= EPS, -normals / torch.clamp(norm, EPS), torch.zeros_like(normals))

    return normals

def _d2n_pytorch_impl(intr, Rc2w, depth):
    sxyz = depth2xyz(intr, depth)
    normal_cam = xyz2normal(sxyz)
    _, H, W = normal_cam.shape
    normal = (Rc2w @ normal_cam.reshape(3, -1)).reshape(3, H, W)

    return normal
    