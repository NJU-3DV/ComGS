

import torch
import torch.nn.functional as F
from scene.envmap import OctahedralEnvMap
from scene.sops import SurfaceOctahedralProbes, sample_from_octahedral_envmaps
from cuda_pbr.sample import uniform_sample_hemisphere, octahedral_importance_sample_sphere

def pbr_shading(
    render_pkg, viewpoint_cam, sops:SurfaceOctahedralProbes, 
    envmap:OctahedralEnvMap, background, num_rays = 128, 
    return_details=False, ignore_local=False, is_training=False,
    important_sampling=False
):
   
    # Deferred PBR shading
    # Get xyz

    # (1) Get SV-Envmaps...
    # Applies max pooling to the depth map. Then projects them into world 
    # coordinates to obtain candinate points. These points will be used to capture SV-EnvMaps.
    # Max pooling might introduce some noisy points, but it is crucial to avoid losing SV-EnvMaps 
    # along depth edges. Prioritizes completeness over absolute accuracy.
    
    depth_buffer = render_pkg["rend_depth"].squeeze(0)
    normal_buffer = render_pkg["rend_normal"].permute(1, 2, 0)       # [H, W, 3]
    alpha_buffer = render_pkg["rend_alpha"].permute(1, 2, 0)       # [H, W, 1]
    normal_buffer = normal_buffer / alpha_buffer.clamp_min(1e-6)
    normal_buffer = F.normalize(normal_buffer, dim=-1) # [H, W, 3]

    candinate_mask = depth_buffer > 1e-7
    candinate_depth = depth_buffer[candinate_mask]
    candinate_normal = normal_buffer[candinate_mask] # [N, 3]

    # rays_with_length is [H, W, 4] (unnormalized ray x3 and length x1)
    rays_with_length = viewpoint_cam.get_cam_rays(world_frame=True)
    candinate_rays_with_length = rays_with_length[candinate_mask] # [N, 4]
    candinate_range_rays = candinate_rays_with_length[..., :3] * candinate_rays_with_length[..., -1:]
    # candinate_range_rays [N, 3] candinate_depth [N] view.camera_center [3]
    candinate_xyz = candinate_range_rays * candinate_depth[..., None] + viewpoint_cam.camera_center[None]

    # [N, res, res, 4]
    probe_envmaps = sops.query_light_knn(candinate_xyz, candinate_normal).cuda()

    # (2) Get the shading points
    mask = depth_buffer > 1e-7 # [H, W, 1]
    albedo_buffer = render_pkg["rend_albedo"].permute(1, 2, 0)       # [H, W, 3]
    metallic_buffer = render_pkg["rend_metallic"].permute(1, 2, 0)   # [H, W, 1]
    roughness_buffer = render_pkg["rend_roughness"].permute(1, 2, 0) # [H, W, 1]

    # # get position buffer from depth buffer
    rays_with_length = viewpoint_cam.get_cam_rays(world_frame=True)
    range_rays = rays_with_length[..., :3] * rays_with_length[..., -1:]

    # # get valid shading points
    shade_normal = normal_buffer[mask]
    shade_albedo = albedo_buffer[mask]
    shade_metallic = metallic_buffer[mask]
    shade_roughness = roughness_buffer[mask]

    # (3) Sample rays 
    # rays: [S, num_rays, 3], inv_pdf: [S, num_rays, 1]
    if important_sampling:

        oct_map = envmap.export_as_octahedral()
        rays, inv_pdf = octahedral_importance_sample_sphere(
            oct_map, num_rays, random_offset=False
        )

        P = shade_normal.shape[0]
        rays = rays.unsqueeze(0).repeat(P, 1, 1)
        inv_pdf = inv_pdf.unsqueeze(0).repeat(P, 1, 1)

    else:
        rays, inv_pdf = uniform_sample_hemisphere(
            shade_normal, num_rays, cuda_backend=True, 
            random_rotate=is_training
        )
    
    # (4) Sample radiance from SV-EnvMaps
    # probe_envmaps: [S, res, res, 4] 
    # rays: [S, num_rays, 3]
    # -> sampled_probe_radiance: [S, num_rays, 4]
    sampled_probe_radiance = sample_from_octahedral_envmaps(
        probe_envmaps, sops.world_to_local_dirs(rays)
    )

    # rays: [S, num_rays, 3]
    # sampled_global_radiance: [S, num_rays, 3]
    sampled_global_radiance = envmap.sample(rays.permute(2, 0, 1)).permute(1, 2, 0)
    # NOTE: sampled_probe_radiance[..., -1:] is occusion 
    if ignore_local:
        sampled_radiance = sampled_probe_radiance[..., -1:] * sampled_global_radiance
    else:
        sampled_radiance = sampled_probe_radiance[..., :3] + sampled_probe_radiance[..., -1:] * sampled_global_radiance
    # sampled_radiance = sampled_global_radiance
    
    # # (5) PBR shading
    vdir = - range_rays[mask]
    from cuda_pbr.pbr import physically_based_rendering
    shading_result = physically_based_rendering(
        shade_normal, shade_albedo, shade_metallic,
        shade_roughness, rays, sampled_radiance, inv_pdf,
        vdir, cuda_backend=True
    )

    # (6) PBR * render_alpha + background * (1 - render_alpha)
    pbr_image = torch.zeros_like(render_pkg["rend_albedo"].permute(1, 2, 0))
    pbr_image[mask] = shading_result
    pbr_image = pbr_image.permute(2, 0, 1)

    if len(background.shape) == 1:
        background = background[:, None, None]
    pbr_image = pbr_image * render_pkg["rend_alpha"] +  background * (1 - render_pkg["rend_alpha"])

    rets = {"pbr": pbr_image}

    # show ao, local light and global light
    if return_details:
        ao_map = torch.zeros_like(render_pkg["rend_alpha"].permute(1, 2, 0))
        visibility = sampled_probe_radiance[..., -1:].mean(dim=1) # [S, num_rays, 1] -> [S, 1]
        ao_map[mask] = visibility
        ao_map = ao_map.permute(2, 0, 1) # [H, W, 1]
        ao_map = ao_map * render_pkg["rend_alpha"]

        # incident light
        local_map = torch.zeros_like(render_pkg["rend_albedo"].permute(1, 2, 0))
        local_map[mask] = sampled_probe_radiance[..., :3].mean(dim=1)
        local_map = local_map.permute(2, 0, 1)
        local_map = local_map * render_pkg["rend_alpha"]

        # global_light
        global_map = torch.zeros_like(render_pkg["rend_albedo"].permute(1, 2, 0))
        global_map[mask] = (sampled_probe_radiance[..., -1:] * sampled_global_radiance).mean(dim=1)
        global_map = global_map.permute(2, 0, 1)
        global_map = global_map * render_pkg["rend_alpha"]

        rets["ao"] = ao_map
        rets["local"] = local_map
        rets["global"] = global_map
    
    return rets
