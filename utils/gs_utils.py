#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import os
import math
import torch
from tqdm import tqdm
from functools import partial
from utils.pbr_utils import rgb_to_srgb
from utils.render_utils import save_img_f32, save_img_u8

class GaussianExtractor(object):
    def __init__(self, gaussians, render, pbr_shading, pipe, bg_func, envmap, sops, num_rays, render_type):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        self.gaussians = gaussians

        self.requires_render = "rf" in render_type
        self.requires_geometry = "geo" in render_type or "pbr" in render_type
        self.requires_material = "mat" in render_type or "pbr" in render_type
        self.requires_light = "lgt" in render_type
        self.requires_pbr = "pbr" in render_type
        
        self.render = partial(
            render, pipe=pipe, bg_func=bg_func, envmap=envmap, 
            requires_render=self.requires_render,
            requires_geometry=self.requires_geometry, 
            requires_material=self.requires_material
        )

        self.render_type = render_type
        self.pbr_shading = None

        if self.requires_pbr:
            self.pbr_shading = partial(
                pbr_shading, sops=sops, envmap=envmap,
                num_rays=num_rays, return_details=self.requires_light
            )

        self.clean()

    @torch.no_grad()
    def clean(self):
        # render
        self.rgbmaps = []
        self.bgmaps = []

        # geometric buffers
        self.alphamaps = []
        self.depthmaps = []
        self.normalmaps = []
        self.pnormals = []
        
        # material buffers
        self.albedomaps = []
        self.metallicmaps = []
        self.roughnessmaps = []

        # pbr
        self.pbrmaps = []

        # light
        self.aomaps = []
        self.localmaps = []
        self.globalmaps = []

        self.viewpoint_stack = []

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack, with_effect=False):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):

            if with_effect:
                scale_modifer = 0.5 * (1 - math.cos(i * math.pi / len(self.viewpoint_stack))) * (1 - 0.01) + 0.01
            else:
                scale_modifer = 1.0

            render_pkg = self.render(viewpoint_cam, self.gaussians, scaling_modifier=scale_modifer)

            if self.requires_render:
                rgb = render_pkg['render']
                rgb = rgb_to_srgb(rgb) # NOTE that we build a Radiance Field in linear RGB space...
                self.rgbmaps.append(rgb.cpu())
            
            if self.requires_geometry:
                alpha = render_pkg['rend_alpha']
                depth = render_pkg['rend_depth']
                normal = render_pkg['rend_normal']
                pnormal = render_pkg['surf_normal']

                self.alphamaps.append(alpha.cpu())
                self.depthmaps.append(depth.cpu())
                self.normalmaps.append(normal.cpu())
                self.pnormals.append(pnormal.cpu())
            
            if self.requires_material:
                albedo = render_pkg['rend_albedo']
                metallic = render_pkg['rend_metallic']
                roughness = render_pkg['rend_roughness']

                self.albedomaps.append(albedo.cpu())
                self.metallicmaps.append(metallic.cpu())
                self.roughnessmaps.append(roughness.cpu())
            
            background = render_pkg['background']
            self.bgmaps.append(background.cpu())
            if self.requires_pbr:
                pbr_pkg = self.pbr_shading(
                    render_pkg, viewpoint_cam, background=background
                )

                pbr_image = rgb_to_srgb(pbr_pkg["pbr"])
                self.pbrmaps.append(pbr_image.cpu())

                if self.requires_light:
                    ao_map = pbr_pkg["ao"]
                    local_map = rgb_to_srgb(pbr_pkg["local"])
                    global_map = rgb_to_srgb(pbr_pkg["global"])

                    self.aomaps.append(ao_map.cpu())
                    self.localmaps.append(local_map.cpu())
                    self.globalmaps.append(global_map.cpu())
        
        if self.requires_render:
            self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        if self.requires_geometry:
            self.alphamaps = torch.stack(self.alphamaps, dim=0)
            self.depthmaps = torch.stack(self.depthmaps, dim=0)
            self.normalmaps = torch.stack(self.normalmaps, dim=0)
            self.pnormals = torch.stack(self.pnormals, dim=0)
        if self.requires_material:
            self.albedomaps = torch.stack(self.albedomaps, dim=0)
            self.metallicmaps = torch.stack(self.metallicmaps, dim=0)
            self.roughnessmaps = torch.stack(self.roughnessmaps, dim=0)

        if self.requires_pbr:
            self.pbrmaps = torch.stack(self.pbrmaps, dim=0)
            
            if self.requires_light:
                self.aomaps = torch.stack(self.aomaps, dim=0)
                self.localmaps = torch.stack(self.localmaps, dim=0)
                self.globalmaps = torch.stack(self.globalmaps, dim=0)

    @torch.no_grad()
    def export_image(self, path):
        gts_path = os.path.join(path, "gt")
        masks_path = os.path.join(path, "masks")
        gts_normal_path = os.path.join(path, "gt_normal")
        gts_albedo_path = os.path.join(path, "gt_albedo")

        render_path = os.path.join(path, "renders")
        alpha_path = os.path.join(path, "alpha")
        depth_path = os.path.join(path, "depth")
        normal_path = os.path.join(path, "normal")
        pnormal_path = os.path.join(path, "pnormal")

        albedo_path = os.path.join(path, "albedo")
        metallic_path = os.path.join(path, "metallic")
        roughness_path = os.path.join(path, "roughness")

        pbr_path = os.path.join(path, "pbr")
        ao_path = os.path.join(path, "ao")
        local_path = os.path.join(path, "local")
        global_path = os.path.join(path, "global")

        os.makedirs(gts_path, exist_ok=True)
        os.makedirs(gts_normal_path, exist_ok=True)
        os.makedirs(gts_albedo_path, exist_ok=True)
        os.makedirs(masks_path, exist_ok=True)

        if self.requires_render:
            os.makedirs(render_path, exist_ok=True)
        
        if self.requires_geometry:
            os.makedirs(alpha_path, exist_ok=True)
            os.makedirs(depth_path, exist_ok=True)
            os.makedirs(normal_path, exist_ok=True)
            os.makedirs(pnormal_path, exist_ok=True)
        
        if self.requires_material:    
            os.makedirs(albedo_path, exist_ok=True)
            os.makedirs(metallic_path, exist_ok=True)
            os.makedirs(roughness_path, exist_ok=True)

        if self.requires_pbr:
            os.makedirs(pbr_path, exist_ok=True)
            if self.requires_light:
                os.makedirs(ao_path, exist_ok=True)
                os.makedirs(local_path, exist_ok=True)
                os.makedirs(global_path, exist_ok=True)

        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):

            image_name = viewpoint_cam.image_name

            gt = viewpoint_cam.get_gt_rgb(self.bgmaps[idx]).cuda()
            if viewpoint_cam.is_linear_rgb:
                gt = rgb_to_srgb(gt)
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, image_name + ".png"))

            if viewpoint_cam.gt_normal is not None:
                gt_normal = viewpoint_cam.gt_normal.cuda()
                save_img_u8(gt_normal.permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(gts_normal_path, image_name + ".png"))
            
            if viewpoint_cam.gt_albedo is not None:
                gt_albedo = viewpoint_cam.gt_albedo.cuda()
                save_img_u8(gt_albedo.permute(1,2,0).cpu().numpy(), os.path.join(gts_albedo_path, image_name + ".png"))

            mask = viewpoint_cam.gt_alpha_mask.cuda()            
            save_img_u8(mask[0].cpu().numpy(), os.path.join(masks_path, image_name + ".png"))

            if self.requires_render:
                save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, image_name + ".png"))
            if self.requires_geometry:
                save_img_u8(self.alphamaps[idx][0].cpu().numpy(), os.path.join(alpha_path, 'alpha_{0:04d}'.format(idx) + ".png"))
                save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(depth_path, image_name + ".tiff"))
                save_img_u8(self.normalmaps[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(normal_path, image_name + ".png"))
                save_img_u8(self.pnormals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(pnormal_path, image_name + ".png"))
            if self.requires_material:
                save_img_u8(self.albedomaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(albedo_path, image_name + ".png"))
                save_img_u8(self.metallicmaps[idx].squeeze(0).cpu().numpy(), os.path.join(metallic_path, image_name + ".png"))
                save_img_u8(self.roughnessmaps[idx].squeeze(0).cpu().numpy(), os.path.join(roughness_path, image_name + ".png"))
            
            if self.requires_pbr:
                save_img_u8(self.pbrmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(pbr_path, image_name + ".png"))
                if self.requires_light:
                    save_img_u8(self.aomaps[idx].squeeze(0).cpu().numpy(), os.path.join(ao_path, image_name + ".png"))
                    save_img_u8(self.localmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(local_path, image_name + ".png"))
                    save_img_u8(self.globalmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(global_path, image_name + ".png"))