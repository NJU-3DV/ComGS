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
import math
import torch.nn.functional as F
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
# from utils.point_utils import depth_to_normal
from scene.envmap import EnvMapBase
from cuda_pbr.d2n import depth2normal
# from scene.cameras import Camera

# def render(
#     viewpoint_camera, pc : GaussianModel, pipe, 
#     bg_func, envmap: EnvMapBase=None, scaling_modifier = 1.0, 
#     override_color = None, **kwargs
# ):
#     """
#     Render the scene. 
    
#     Background tensor (bg_color) must be on GPU!
#     """
#     from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
 
#     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
#     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     bg_color = torch.ones(3, dtype=torch.float32, device=pc.get_xyz.device)

#     # Set up rasterization configuration
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         sh_degree=pc.active_sh_degree,
#         campos=viewpoint_camera.camera_center,
#         prefiltered=False,
#         debug=False,
#         # pipe.debug
#     )

#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

#     means3D = pc.get_xyz
#     means2D = screenspace_points
#     opacity = pc.get_opacity

#     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
#     # scaling / rotation by the rasterizer.
#     scales = None
#     rotations = None
#     cov3D_precomp = None
#     if pipe.compute_cov3D_python:
#         # currently don't support normal consistency loss if use precomputed covariance
#         splat2world = pc.get_covariance(scaling_modifier)
#         W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
#         near, far = viewpoint_camera.znear, viewpoint_camera.zfar
#         ndc2pix = torch.tensor([
#             [W / 2, 0, 0, (W-1) / 2],
#             [0, H / 2, 0, (H-1) / 2],
#             [0, 0, far-near, near],
#             [0, 0, 0, 1]]).float().cuda().T
#         world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
#         cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
#     else:
#         scales = pc.get_scaling
#         rotations = pc.get_rotation
    
#     # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
#     # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
#     pipe.convert_SHs_python = False
#     shs = None
#     colors_precomp = None
#     if override_color is None:
#         if pipe.convert_SHs_python:
#             shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
#             dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
#             dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
#             sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
#             colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
#         else:
#             shs = pc.get_features
#     else:
#         colors_precomp = override_color
    
#     rendered_image, radii, allmap = rasterizer(
#         means3D = means3D,
#         means2D = means2D,
#         shs = shs,
#         colors_precomp = colors_precomp,
#         opacities = opacity,
#         scales = scales,
#         rotations = rotations,
#         cov3D_precomp = cov3D_precomp
#     )
    
#     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#     # They will be excluded from value updates used in the splitting criteria.
#     rets =  {"render": rendered_image,
#              "background": bg_color,
#             "viewspace_points": means2D,
#             "visibility_filter" : radii > 0,
#             "radii": radii,
#     }


#     # additional regularizations
#     render_alpha = allmap[1:2]

#     # get normal map
#     # transform normal from view space to world space
#     render_normal = allmap[2:5]
#     render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
#     # get median depth map
#     render_depth_median = allmap[5:6]
#     render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

#     # get expected depth map
#     render_depth_expected = allmap[0:1]
#     render_depth_expected = (render_depth_expected / render_alpha)
#     render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
#     # get depth distortion map
#     render_dist = allmap[6:7]

#     # psedo surface attributes
#     # surf depth is either median or expected by setting depth_ratio to 1 or 0
#     # for bounded scene, use median depth, i.e., depth_ratio = 1; 
#     # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
#     surf_depth = render_depth_expected
    
#     # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
#     from cuda_pbr.d2n import depth2normal
#     surf_normal = depth2normal(
#         viewpoint_camera.intrinsics, 
#         viewpoint_camera.Rc2w, 
#         surf_depth[0], cuda_backend=True)
#     # remember to multiply with accum_alpha since render_normal is unnormalized.
#     surf_normal = surf_normal * (render_alpha).detach()

#     rets.update({
#             'rend_alpha': render_alpha,
#             'rend_normal': render_normal,
#             'rend_dist': render_dist,
#             'rend_depth': surf_depth,
#             'surf_normal': surf_normal,
#             'rend_albedo': torch.zeros_like(surf_normal),
#             'rend_metallic': torch.zeros_like(surf_depth),
#             'rend_roughness': torch.zeros_like(surf_depth)
#     })

#     return rets

def render(viewpoint_camera, pc: GaussianModel, pipe, 
           bg_func, envmap: EnvMapBase=None, scaling_modifier=1.0, 
           override_color=None, flag_max_count=False, requires_render=True,
           requires_geometry=False, requires_material=False, use_trained_exp=False):
    
    from comgs_rasterizer import GaussianRasterizationSettings, GaussianRasterizer

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        flag_max_count=flag_max_count,
        requires_render=requires_render,
        requires_geometry=requires_geometry,
        requires_material=requires_material
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    transMat_precomp = None
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = pc.get_features_dc, pc.get_features_rest
    else:
        colors_precomp = override_color

    materials = None
    if requires_material:
        materials = pc.get_material
    
    vdir = None
    if envmap is not None:
        vdir = viewpoint_camera.get_cam_rays(world_frame=True).permute(2, 0, 1)[:3]
    bg_color = bg_func(envmap, vdir)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    (
        out_render,
        out_geometry,
        out_material, 
        radii,
        accum_max_count
    ) = rasterizer(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        transMat_precomp = transMat_precomp,
        materials = materials,
        bg = bg_color
    )
    
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        out_render = torch.matmul(out_render.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {
        "background": bg_color,
        "viewspace_points": means2D,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "area_max": accum_max_count
    }

    if requires_render:
        rets["render"] = out_render

    if requires_geometry:
        # additional regularizations
        rend_alpha = out_geometry[0:1]

        # get depth map
        rend_depth = out_geometry[1:2]
        rend_depth = rend_depth / rend_alpha
        rend_depth = torch.nan_to_num(rend_depth, 0, 0, 0)

        # get normal map
        # transform normal from view space to world space
        rend_normal = out_geometry[2:5]
        rend_normal = (rend_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
        
        surf_normal = depth2normal(
            viewpoint_camera.intrinsics, 
            viewpoint_camera.Rc2w, 
            rend_depth[0], cuda_backend=True)
        surf_normal = surf_normal * (rend_alpha).detach()
        
        rets.update({
            'rend_alpha': rend_alpha,
            'rend_normal': rend_normal,
            'rend_depth': rend_depth,
            'surf_normal': surf_normal
        })
    
    if requires_material:
        render_albedo = out_material[0:3]
        render_metallic = out_material[3:4]
        render_roughness = out_material[4:5]
        
        rets.update({
            'rend_albedo': render_albedo,
            'rend_metallic': render_metallic,
            'rend_roughness': render_roughness
        })

    return rets

# def pbr_render_full(view: Camera, gbuffers: dict, pc : GaussianModel, global_lighting:OctahedralEnvMap, num_ray = 32, update_bvh=False):
    
#     from cuda_pbr.sample import uniform_sample_hemisphere
#     from cuda_pbr.pbr import physically_based_rendering

#     # trace lights, alpha
#     if not hasattr(pc.gaussian_tracer, 'gs_idxs'):
#         pc.build_bvh()
    
#     if update_bvh:
#         pc.update_bvh()
    
#     # points from Gbuffers in 3D space
#     rgb_buffer = gbuffers["render"]
#     alpha_buffer = gbuffers["rend_alpha"]
#     depth_buffer = gbuffers["rend_depth"]
#     normal_buffer = gbuffers["rend_normal"]

#     albedo_buffer = gbuffers["rend_albedo"]
#     metallic_buffer = gbuffers["rend_metallic"]
#     roughness_buffer = gbuffers["rend_roughness"]

#     depths = depth_buffer.reshape(-1)
#     indices = depths > 1e-7

#     cam_rays = view.get_cam_rays()
#     cam_xyz = cam_rays[indices, :3] * cam_rays[indices, -1:] * depths[indices, None]

#     pts = torch.bmm(view.c2w[:3, :3].unsqueeze(0).repeat(cam_xyz.shape[0], 1, 1), 
#                     cam_xyz.unsqueeze(2)) + view.c2w[:3, -1:].unsqueeze(0)
#     pts = pts.squeeze(-1) # [P, 3]

#     normal = normal_buffer.permute(1, 2, 0).reshape(-1, 3)[indices]

#     albedo = albedo_buffer.permute(1, 2, 0).reshape(-1, 3)[indices]
#     metallic = metallic_buffer.permute(1, 2, 0).reshape(-1, 1)[indices]
#     roughness = roughness_buffer.permute(1, 2, 0).reshape(-1, 1)[indices]

#     # sampling rays [P, N, 3]
#     # print(normal[2])
#     rays, invpdf = uniform_sample_hemisphere(normal=normal, num_ray=num_ray)

#     # trace lighting (radiance, alpha)
#     rays_o = pts.unsqueeze(1).repeat(1, num_ray, 1).reshape(-1, 3)
#     rays_d = rays.reshape(-1, 3)
#     trace_pkg = pc.trace(rays_o + rays_d * 0.05, rays_d)
#     # trace_pkg = pc.trace(rays_o, rays_d)

#     trace_rgb = trace_pkg["render"].reshape(-1, num_ray, 3) # [P, N, 3]
#     trace_alpha = trace_pkg["alpha"].reshape(-1, num_ray, 1) # [P, N, 1]

#     # alpha_blending traced RF with global lighting
#     sample_rays = rays.permute(2, 0, 1) # [3, P, N]
#     light_global = global_lighting.sample(sample_rays).permute(1, 2, 0)
#     light = trace_rgb + (1 - trace_alpha) * light_global
#     # light = light_global
    
#     # camera direction [N, 3]
#     world_view_dir = - (view.c2w[:3, :3] @ cam_rays[..., :3].T).T
#     vdir = world_view_dir[indices, :]
#     # PBR rendering [P, 3]
#     pbr = physically_based_rendering(
#         normal, albedo, metallic, roughness, rays, light, invpdf, vdir, cuda_backend=True
#     )

#     # [3, H, W]
#     # sample envmap
#     image = torch.zeros_like(rgb_buffer)
#     valid_mask = indices.reshape(*image.shape[1:]) 
#     image[:3, valid_mask] = pbr.T

#     return image


# def pbr_render_index(view: Camera, gbuffers: dict, pc : GaussianModel, global_lighting:OctahedralEnvMap, select_mask, num_ray = 32, update_bvh=False):

#     from cuda_pbr.sample import uniform_sample_hemisphere, cosine_sample_hemisphere, fibonacci_sampling_hemisphere
#     from cuda_pbr.pbr import physically_based_rendering

#     # trace lights, alpha
#     if not hasattr(pc.gaussian_tracer, 'gs_idxs'):
#         pc.build_bvh()
    
#     if update_bvh:
#         pc.update_bvh()
    
#     indices = torch.where(select_mask.reshape(-1))[0]
    
#     # points from Gbuffers in 3D space
#     depth_buffer = gbuffers["rend_depth"]
#     normal_buffer = gbuffers["rend_normal"]
#     albedo_buffer = gbuffers["rend_albedo"]
#     metallic_buffer = gbuffers["rend_metallic"]
#     roughness_buffer = gbuffers["rend_roughness"]

#     depths = depth_buffer.reshape(-1)

#     cam_rays = view.get_cam_rays()
#     cam_xyz = cam_rays[indices, :3] * cam_rays[indices, -1:] * depths[indices, None]

#     pts = torch.bmm(view.c2w[:3, :3].unsqueeze(0).repeat(cam_xyz.shape[0], 1, 1), 
#                     cam_xyz.unsqueeze(2)) + view.c2w[:3, -1:].unsqueeze(0)
#     pts = pts.squeeze(-1) # [P, 3]

#     normal = normal_buffer.permute(1, 2, 0).reshape(-1, 3)[indices]
#     albedo = albedo_buffer.permute(1, 2, 0).reshape(-1, 3)[indices]
#     metallic = metallic_buffer.permute(1, 2, 0).reshape(-1, 1)[indices]
#     roughness = roughness_buffer.permute(1, 2, 0).reshape(-1, 1)[indices]

#     # sampling rays [P, N, 3]
#     rays, invpdf = uniform_sample_hemisphere(normal=normal, num_ray=num_ray, cuda_backend=True)

#     # trace lighting (radiance, alpha)
#     rays_o = pts.unsqueeze(1).repeat(1, num_ray, 1).reshape(-1, 3)
#     rays_d = rays.reshape(-1, 3)
#     # trace_pkg = pc.trace(rays_o + rays_d * 0.05, rays_d)
#     trace_pkg = pc.trace(rays_o + rays_d * 0.225, rays_d)
#     # trace_pkg = pc.trace(rays_o + rays_d, rays_d)

#     trace_rgb = trace_pkg["render"].reshape(-1, num_ray, 3) # [P, N, 3]
#     trace_alpha = trace_pkg["alpha"].reshape(-1, num_ray, 1) # [P, N, 1]

#     # alpha_blending traced RF with global lighting
#     sample_rays = rays.permute(2, 0, 1) # [3, P, N]
#     light_global = global_lighting.sample(sample_rays).permute(1, 2, 0)
#     light = light_global * (1 - trace_alpha) + trace_rgb

#     # camera direction [N, 3]
#     world_view_dir = - (view.c2w[:3, :3] @ cam_rays[..., :3].T).T
#     vdir = world_view_dir[indices, :]
#     # PBR rendering [P, 3]
#     pbr = physically_based_rendering(
#         normal, albedo, metallic, roughness, rays, light, invpdf, vdir, cuda_backend=True
#     )

#     return pbr.T