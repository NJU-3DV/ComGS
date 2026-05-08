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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        bg,
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        transMat_precomp,
        materials,
        raster_settings
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            transMat_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            dc,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            materials,
            raster_settings.prefiltered,
            raster_settings.flag_max_count,
            raster_settings.requires_render,
            raster_settings.requires_geometry,
            raster_settings.requires_material,
            raster_settings.debug           
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                (
                    num_rendered, 
                    num_buckets, 
                    out_render,
                    out_geometry,
                    out_material, 
                    radii, 
                    accum_max_count,
                    geomBuffer, 
                    binningBuffer, 
                    imgBuffer, 
                    sampleBuffer
                ) = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            (
                num_rendered, 
                num_buckets, 
                out_render,
                out_geometry,
                out_material, 
                radii, 
                accum_max_count,
                geomBuffer, 
                binningBuffer, 
                imgBuffer, 
                sampleBuffer
            ) = _C.rasterize_gaussians(*args)
        
        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.num_buckets = num_buckets
        ctx.save_for_backward(
            bg, 
            colors_precomp, 
            means3D, scales, 
            rotations, 
            transMat_precomp, 
            radii, 
            dc, 
            sh, 
            materials, 
            geomBuffer, 
            binningBuffer, 
            imgBuffer, 
            sampleBuffer
        )

        return out_render, out_geometry, out_material, radii, accum_max_count

    @staticmethod
    def backward(ctx, grad_out_render, grad_out_geometry, grad_out_material, *_):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        num_buckets = ctx.num_buckets
        raster_settings = ctx.raster_settings
        (
            bg, 
            colors_precomp, 
            means3D, 
            scales, 
            rotations, 
            transMat_precomp, 
            radii, 
            dc, 
            sh, 
            materials, 
            geomBuffer, 
            binningBuffer, 
            imgBuffer, 
            sampleBuffer
         ) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            bg,
            means3D, 
            radii, 
            colors_precomp, 
            scales, 
            rotations, 
            raster_settings.scale_modifier, 
            transMat_precomp, 
            raster_settings.viewmatrix, 
            raster_settings.projmatrix, 
            raster_settings.tanfovx, 
            raster_settings.tanfovy, 
            grad_out_render,
            grad_out_geometry,
            grad_out_material, 
            dc,
            sh, 
            raster_settings.sh_degree, 
            raster_settings.campos,
            materials,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            num_buckets,
            sampleBuffer,
            raster_settings.debug,
            raster_settings.requires_render,
            raster_settings.requires_geometry,
            raster_settings.requires_material
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                (
                    grad_bg, 
                    grad_means2D, 
                    grad_colors_precomp, 
                    grad_opacities, 
                    grad_means3D, 
                    grad_transMat_precomp, 
                    grad_dc, 
                    grad_sh, 
                    grad_scales, 
                    grad_rotations, 
                    grad_materials
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            (
                grad_bg, 
                grad_means2D, 
                grad_colors_precomp, 
                grad_opacities, 
                grad_means3D, 
                grad_transMat_precomp, 
                grad_dc, 
                grad_sh, 
                grad_scales, 
                grad_rotations, 
                grad_materials
            ) = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_bg,
            grad_means3D,
            grad_means2D,
            grad_dc,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_transMat_precomp,
            grad_materials,
            None
        )

        return grads
    
class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx : float
    tanfovy : float
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    flag_max_count: bool
    requires_render: bool
    requires_geometry: bool
    requires_material: bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, bg, dc = None, shs = None, colors_precomp = None, 
                scales = None, rotations = None, transMat_precomp = None, materials = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and transMat_precomp is None) or ((scales is not None or rotations is not None) and transMat_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed transMat!')
        
        if dc is None:
            dc = torch.Tensor([])
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if transMat_precomp is None:
            transMat_precomp = torch.Tensor([])
        
        if materials is None:
            materials = torch.Tensor([])
        
        # Invoke C++/CUDA rasterization routine
        return _RasterizeGaussians.apply(
                bg,
                means3D,
                means2D,
                dc,
                shs,
                colors_precomp,
                opacities,
                scales,
                rotations,
                transMat_precomp,
                materials,
                raster_settings,
            )