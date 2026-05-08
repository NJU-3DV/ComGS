import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math

def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

def save_ply(filename, points, colors=None, normals=None):

    from plyfile import PlyData, PlyElement

    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy().astype(np.float32)
    if colors is not None and isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy().astype(np.float32)
    if normals is not None and isinstance(normals, torch.Tensor):
        normals = normals.detach().cpu().numpy().astype(np.float32)

    num_points = points.shape[0]

    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

    vertex_elements = []
    for i in range(num_points):
        vertex_elements.append((points[i, 0], points[i, 1], points[i, 2]))

    if colors is not None:
        vertex_dtype.extend([('red', 'f4'), ('green', 'f4'), ('blue', 'f4')])
        for i in range(num_points):
           vertex_elements[i] = vertex_elements[i] + (colors[i,0], colors[i,1],colors[i,2])

    if normals is not None:
        vertex_dtype.extend([('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])

        for i in range(num_points):
            vertex_elements[i] = vertex_elements[i] + (normals[i, 0], normals[i, 1], normals[i, 2])

    vertex = np.array(vertex_elements, dtype=vertex_dtype)

    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(filename)

def load_ply(ply_file):
    from plyfile import PlyData, PlyElement

    plydata = PlyData.read(ply_file)
    vertex = plydata['vertex']

    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T
    normals = np.vstack([vertex['nx'], vertex['ny'], vertex['nz']]).T

    colors = colors.astype(np.float32) / 255.

    return points, colors, normals