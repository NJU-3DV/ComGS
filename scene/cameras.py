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
from torch import nn
import numpy as np
from utils.image_utils import get_edges
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(
        self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
        image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
        data_device = "cuda", depth=None, normal=None, albedo = None,
        metallic=None, roughness=None, gt_normal=None, gt_albedo= None,
        is_linear_rgb=False
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.is_linear_rgb = is_linear_rgb

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.gt_alpha_mask = None
        
        self.depth = depth
        if self.depth is not None:
            # depth [H, W]
            self.depth = depth.to(self.data_device)

        self.normal = normal
        if self.normal is not None:
            # normal [3, H, W]
            self.normal = normal.to(self.data_device)

        self.albedo = albedo
        if self.albedo is not None:
            # albedo [3, H, W]
            self.albedo = albedo.to(self.data_device)

        self.metallic = metallic
        if self.metallic is not None:
            # metallic [H, W]
            self.metallic = metallic.to(self.data_device)

        self.roughness = roughness
        if self.roughness is not None:
            # roughness [H, W]
            self.roughness = roughness.to(self.data_device)
        
        self.gt_normal = gt_normal
        if self.gt_normal is not None:
            # gt_normal [3, H, W]
            self.gt_normal = gt_normal.to(self.data_device)
        
        self.gt_albedo = gt_albedo
        if self.gt_albedo is not None:
            # gt_albedo [3, H, W]
            self.gt_albedo = gt_albedo.to(self.data_device)
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        tan_fovx = np.tan(self.FoVx / 2.0)
        tan_fovy = np.tan(self.FoVy / 2.0)
        self.fx = self.image_width / (2.0 * tan_fovx)
        self.fy = self.image_height / (2.0 * tan_fovy)
        self.cx = float(self.image_width) / 2.0
        self.cy = float(self.image_height) / 2.0

        self.intrinsics = torch.tensor([self.fx, self.fy, self.cx, self.cy]).cuda().float()
    
    @property
    def c2w(self):    
        return self.world_view_transform.transpose(0, 1).inverse()
    
    @property
    def w2c(self):
        return self.world_view_transform.transpose(0, 1)

    @property
    def Rc2w(self):
        return self.c2w[:3, :3]
    
    @property
    def Rw2c(self):
        return self.w2c[:3, :3]
    
    @property
    def tw2c(self):
        return self.w2c[:3, 3]
    
    def get_gt_rgb(self, background: torch.Tensor):
        background = background.to(self.data_device)
        if len(background.shape) == 1:
            background = background.unsqueeze(-1).unsqueeze(-1)
            return self.original_image * self.gt_alpha_mask + background * (1.0 - self.gt_alpha_mask)
        else:
            # NOTE: if learnable background, we return the original_image
            return self.original_image

    def get_cam_rays(self, world_frame=False, step=1):
        # [H, W, 4]
        v, u = torch.meshgrid(
            torch.arange(0, self.image_height, step=step, device='cuda'),
            torch.arange(0, self.image_width, step=step, device='cuda'), indexing="ij")

        directions = torch.stack([(u + 0.5 - self.cx) / self.fx, 
                                  (v + 0.5 - self.cy) / self.fy,
                                  torch.ones_like(u)], dim=-1)
        
        length = torch.norm(directions, p=2, dim=-1, keepdim=True)
        directions /= length

        if world_frame:
            # [Rc2W]: 3 * 3
            # directions: H * W * 3
            directions = torch.sum(self.Rc2w[None, None] * directions[:, :, None], dim=-1)

        return torch.cat([directions, length], dim=-1)
        

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

