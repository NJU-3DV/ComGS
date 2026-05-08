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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import build_scaling_rotation
import warnings
import json

# composition
from utils.general_utils import build_rotation, rotation_to_quaternion, quaternion_multiply
from utils.sh_utils import rotate_sh



# def reddish_to_golden(colors, golden_hue="balanced"):
#     """
#     Transform reddish colors in a [N, 3] RGB tensor to a golden hue.
    
#     Args:
#         colors (torch.Tensor): A tensor of shape [N, 3] with RGB values in range [0, 1].
#         golden_hue (str): Type of gold shade, options: "balanced", "warm", "amber".
    
#     Returns:
#         torch.Tensor: Transformed tensor with reddish colors shifted to golden.
#     """
#     # Define different shades of gold (normalized RGB)
#     golden_shades = {
#         "balanced": torch.tensor([1.0, 0.75, 0.1], device="cuda"),  # Less yellow, slight warmth
#         "warm": torch.tensor([1.0, 0.65, 0.05], device="cuda"),     # Warmer, less intense yellow
#         "amber": torch.tensor([1.0, 0.7, 0.2], device="cuda")       # A bit more orange-amber tone
#     }
#     golden_color = golden_shades.get(golden_hue, golden_shades["balanced"])
    
#     # Criteria for "reddish" colors: R is dominant and significantly higher than G and B
#     redness_mask = (colors[:, 0] > 0.6) & (colors[:, 0] > colors[:, 1] * 1.2) & (colors[:, 0] > colors[:, 2] * 1.5)
    
#     # Compute a blending factor based on how "red" the color is
#     redness_strength = torch.clamp((colors[:, 0] - colors[:, 1]) * 2.0, 0.0, 1.0)
#     redness_strength = redness_strength * redness_mask.float()
    
#     # Reshape for broadcasting
#     redness_strength = redness_strength.unsqueeze(1)  # Shape [N, 1]
#     golden_color = golden_color.unsqueeze(0)  # Shape [1, 3]
    
#     # Blend the original color with the golden color based on redness strength
#     transformed_colors = (1.0 - redness_strength) * colors + redness_strength * golden_color
    
#     return transformed_colors

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        # for pbr
        self.albedo_activation = torch.sigmoid
        self.inverse_albedo_activation = inverse_sigmoid
        self.metallic_activation = torch.sigmoid
        self.inverse_metallic_activation = inverse_sigmoid
        self.roughness_activation = torch.sigmoid
        self.inverse_roughness_activation = inverse_sigmoid


    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        # for pbr
        self._albedo = torch.empty(0)
        self._metallic = torch.empty(0)
        self._roughness = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.radii = None

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._albedo,
            self._metallic,
            self._roughness,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args=None):
        (
            self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self._albedo,
            self._metallic,
            self._roughness,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale
        ) = model_args

        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)

        # NOTE: from irgs
        # (self.active_sh_degree, 
        # self._xyz, 
        # self._metallic, 
        # self._roughness, 
        # self._albedo, 
        # self._features_dc, 
        # self._features_rest,
        # self._scaling, 
        # self._rotation, 
        # self._opacity,
        # self.max_radii2D, 
        # xyz_gradient_accum, 
        # denom,
        # opt_dict, 
        # env_dict,
        # self.spatial_lr_scale) = model_args

        # if training_args is not None:
        #     self.training_setup(training_args)
        #     self.xyz_gradient_accum = xyz_gradient_accum
        #     self.denom = denom

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_albedo(self):
        return self.albedo_activation(self._albedo)
    
    @property
    def get_metallic(self):
        return self.metallic_activation(self._metallic)
    
    @property
    def get_roughness(self):
        return self.roughness_activation(self._roughness)
    
    @property
    def get_normal(self):
        normals = torch.zeros_like(self._xyz)
        normals[..., -1] = 1
        RS = build_scaling_rotation(self.get_scaling, self.get_rotation)

        normals = torch.bmm(RS, normals[..., None]).squeeze(-1)
        return normals

    @property
    def get_exposure(self):
        return self._exposure
    
    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    @property
    def get_material(self):
        albedo = self.get_albedo
        metallic = self.get_metallic
        roughness = self.get_roughness
        return torch.cat((albedo, metallic, roughness), dim=1)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):

        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        fused_normal = torch.tensor(np.asarray(pcd.normals)).float().cuda()

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # for exposure
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

        # for pbr 
        # NOTE: test this init
        albedo = self.inverse_albedo_activation(torch.zeros((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda") + 0.5) # [0, 1]
        metallic = self.inverse_albedo_activation(torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda") + 0.2) # [1e-9, 0.2]
        roughness = self.inverse_albedo_activation(torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda") + 0.6) # [0.3, 0.7]
        
        self._albedo = nn.Parameter(albedo.requires_grad_(True))
        self._metallic = nn.Parameter(metallic.requires_grad_(True))
        self._roughness = nn.Parameter(roughness.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / training_args.sh_div, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._albedo], 'lr': training_args.albedo_lr, "name": "albedo"},
            {'params': [self._metallic], 'lr': training_args.metallic_lr, "name": "metallic"},
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"}
        ]
        
        # self.exposure_optimizer = torch.optim.Adam([self._exposure])
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)
        
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    
    def training_pbr_setup(self, training_args):

        # pbr_scale = 0.1
        pbr_scale = 0.1
        self._features_dc.grad = None
        self._features_rest.grad = None
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale * pbr_scale, "name": "xyz"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr * pbr_scale, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * pbr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr * pbr_scale, "name": "rotation"},
            {'params': [self._albedo], 'lr': training_args.albedo_lr, "name": "albedo"},
            {'params': [self._metallic], 'lr': training_args.metallic_lr, "name": "metallic"},
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"}
        ]
        
        self.exposure_optimizer = torch.optim.Adam([self._exposure])
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)
        
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        if self.optimizer is None:
            return 
        
        # if self.pretrained_exposures is None:
        #     for param_group in self.exposure_optimizer.param_groups:
        #         param_group['lr'] = self.exposure_scheduler_args(iteration)
        
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        
        # pbr
        for i in range(self._albedo.shape[1]):
            l.append('albedo_{}'.format(i))
        l.append('metallic')
        l.append('roughness')
        
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = self.get_normal.detach().cpu().numpy()

        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # pbr
        albedo = self._albedo.detach().cpu().numpy()
        metallic = self._metallic.detach().cpu().numpy()
        roughness = self._roughness.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, albedo, metallic, roughness), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        # for pbr, show compatibility with 2DGS
        # NOTE: this init shoud be test!!!
        albedo_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("albedo_")]
        albedo_names = sorted(albedo_names, key = lambda x: int(x.split('_')[-1]))
        if albedo_names:
            albedo = np.zeros((xyz.shape[0], len(albedo_names)))
            for idx, attr_name in enumerate(albedo_names):
                albedo[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            warnings.warn("Albedo names do not exist. Random albedo values have been generated.", UserWarning)
            # Generate random albedo if albedo_names do not exist
            albedo = self.inverse_albedo_activation(torch.zeros((self._xyz.shape[0], 3), dtype=torch.float, device="cuda") + 0.5).cpu().numpy()

        if "metallic" in plydata.elements[0]:
            metallic = np.asarray(plydata.elements[0]["metallic"])[..., np.newaxis]
        else:
            warnings.warn("Metallic attribute does not exist. Random metallic values have been generated.", UserWarning)
            metallic = self.inverse_albedo_activation(torch.zeros((self._xyz.shape[0], 1), dtype=torch.float, device="cuda") + 0.2).cpu().numpy()
        
        if "roughness" in plydata.elements[0]:  
            roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]
        else:
            # Generate random roughness
            warnings.warn("Roughness attribute does not exist. Random roughness values have been generated.", UserWarning)
            roughness = self.inverse_albedo_activation(torch.zeros((self._xyz.shape[0], 1), dtype=torch.float, device="cuda") + 0.6).cpu().numpy()
        
        self._albedo = nn.Parameter(torch.tensor(albedo, dtype=torch.float, device="cuda").requires_grad_(True))
        self._metallic = nn.Parameter(torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(True))
        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # pbr
        self._albedo = optimizable_tensors["albedo"]
        self._metallic = optimizable_tensors["metallic"]
        self._roughness = optimizable_tensors["roughness"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.radii is not None:
            self.radii = self.radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
            self, new_xyz, new_features_dc, new_features_rest, 
            new_opacities, new_scaling, new_rotation,
            new_albedo, new_metallic, new_roughness, new_radii=None
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling" : new_scaling,
            "rotation" : new_rotation,
            # for pbr
            "albedo": new_albedo,
            "metallic": new_metallic,
            "roughness": new_roughness
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # pbr
        self._albedo = optimizable_tensors["albedo"]
        self._metallic = optimizable_tensors["metallic"]
        self._roughness = optimizable_tensors["roughness"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.radii is not None:
            self.radii = torch.concat([self.radii, new_radii], dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, mask=None, N=2):

        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        
        if mask is not None:
            padded_mask = torch.zeros((n_init_points), dtype=torch.bool, device='cuda')
            padded_mask[:grads.shape[0]] = mask
            selected_pts_mask = torch.logical_or(selected_pts_mask, padded_mask)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        # pbr
        new_albedo = self._albedo[selected_pts_mask].repeat(N, 1)
        new_metallic = self._metallic[selected_pts_mask].repeat(N, 1)
        new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)

        new_radii = None
        if self.radii is not None:
            new_radii = self.radii[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, 
            new_opacity, new_scaling, new_rotation,
            new_albedo, new_metallic, new_roughness, new_radii
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        # pbr
        new_albedo = self._albedo[selected_pts_mask]
        new_metallic = self._metallic[selected_pts_mask]
        new_roughness = self._roughness[selected_pts_mask]

        new_radii = None
        if self.radii is not None:
            new_radii = self.radii[selected_pts_mask]

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, 
            new_opacities, new_scaling, new_rotation,
            new_albedo, new_metallic, new_roughness, new_radii
        )


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, mask_blur=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent, mask_blur)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    # composition
    @property
    def attribute_names(self):
        attribute_names = ['xyz', 'features_dc', 'features_rest', 'scaling', 'rotation', 'opacity',
                           'albedo', 'roughness', 'metallic']
        return attribute_names
    
    @classmethod
    def create_from_gaussians(cls, gaussians_list):
        assert len(gaussians_list) > 0
        sh_degree = max(g.max_sh_degree for g in gaussians_list)
        gaussians = GaussianModel(sh_degree=sh_degree)
        attribute_names = gaussians.attribute_names
        for attribute_name in attribute_names:
            setattr(gaussians, "_" + attribute_name,
                    nn.Parameter(torch.cat([getattr(g, "_" + attribute_name).data for g in gaussians_list],
                                           dim=0).requires_grad_(True)))

        return gaussians
    
    def update_albedo(self, scale) -> None:
        albedo_new = torch.clamp(self.get_albedo * scale, 1e-6, 1.0 - 1e-6)
        self._albedo = inverse_sigmoid(albedo_new)
    
    @torch.no_grad()
    def set_transform(self, rotation=None, center=None, scale=None, offset=None, transform=None):
        if transform is not None:
            scale = transform[:3, :3].norm(dim=-1)

            if scale[0] != scale[1] and scale[0] != scale[2]:
                assert "Currently, only the same scales on all axes are supported..."

            self._scaling.data = self.scaling_inverse_activation(self.get_scaling * scale[:2])
            xyz_homo = torch.cat([self._xyz.data, torch.ones_like(self._xyz[:, :1])], dim=-1)
            self._xyz.data = (xyz_homo @ transform.T)[:, :3]
            rotation = transform[:3, :3] / scale[:, None]
            rotation_q = rotation_to_quaternion(rotation[None])
            self._rotation.data = quaternion_multiply(rotation_q, self._rotation.data)
            sh_data = torch.cat((self._features_dc.transpose(1, 2), self._features_rest.transpose(1, 2)), dim=-1)
            sh_data = rotate_sh(sh_data, rotation)
            self._features_dc.data = sh_data[..., 0].unsqueeze(-1).transpose(1, 2)
            self._features_rest.data = sh_data[..., 1:].transpose(1, 2)

            return

        if center is not None:
            self._xyz.data = self._xyz.data - center
        if rotation is not None:
            self._xyz.data = (self._xyz.data @ rotation.T)
            rotation_q = rotation_to_quaternion(rotation[None])
            self._rotation.data = quaternion_multiply(rotation_q, self._rotation.data)
            sh_data = torch.cat((self._features_dc.transpose(1, 2), self._features_rest.transpose(1, 2)), dim=-1)
            sh_data = rotate_sh(sh_data, rotation)
            self._features_dc.data = sh_data[..., 0].unsqueeze(-1).transpose(1, 2)
            self._features_rest.data = sh_data[..., 1:].transpose(1, 2)
        if scale is not None:
            self._xyz.data = self._xyz.data * scale
            self._scaling.data = self.scaling_inverse_activation(self.get_scaling * scale)
        if offset is not None:
            self._xyz.data = self._xyz.data + offset
    

def crop(input_gaussian, center, radius):

    sh_degree = input_gaussian.max_sh_degree
    gaussians = GaussianModel(sh_degree=sh_degree)
    attribute_names = gaussians.attribute_names

    xyz = input_gaussian.get_xyz
    # mask = torch.sum((xyz - center[None,])**2, dim=-1) < radius * radius

    rectangle_min = center - radius
    rectangle_max = center + radius
    x_condition = (xyz[:, 0] >= rectangle_min[0]) & (xyz[:, 0] <= rectangle_max[0])
    y_condition = (xyz[:, 1] >= rectangle_min[1]) & (xyz[:, 1] <= rectangle_max[1])
    z_condition = (xyz[:, 2] >= rectangle_min[2]) & (xyz[:, 2] <= rectangle_max[2])
    mask = x_condition & y_condition & z_condition

    for attribute_name in attribute_names:
        setattr(gaussians, "_" + attribute_name, nn.Parameter(
            getattr(input_gaussian, "_" + attribute_name).data[mask]))

    return gaussians
