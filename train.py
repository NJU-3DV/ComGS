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

import os
# # NOTE: set some environment variables to avoid increasing reserved memory
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync,max_split_size_mb:512,expandable_segments:True"

import torch
from random import randint
from utils.loss_utils import l1_loss, entropy_loss, first_order_edge_aware_loss
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.pbr_utils import rgb_to_srgb
from scene.envmap import OctahedralEnvMap
from scene.sops import SurfaceOctahedralProbes
from utils.image_utils import write_exr
from scene.reconstruction import Reconstruction
from gaussian_renderer.pbr import pbr_shading
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import math
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid


def show_sops(tex_radiance, tex_alpha):
    
    M = tex_radiance.shape[0]
    N = int(math.sqrt(M))
    res = tex_radiance.shape[1]

    radiance = tex_radiance[:N*N]
    radiance = radiance.view(N, N, res, res, 3)
    radiance_show = radiance.permute(0, 2, 1, 3, 4).reshape(N * res, N * res, 3)

    alpha = tex_alpha[:N*N]
    alpha = alpha.view(N, N, res, res, 1)
    alpha_show = alpha.permute(0, 2, 1, 3, 4).reshape(N * res, N * res, 1)

    radiance_show = radiance_show.detach().cpu().numpy()
    alpha_show = alpha_show.detach().cpu().numpy()

    write_exr("sops_radiance.exr", radiance_show)
    write_exr("sops_alpha.exr", alpha_show)

    plt.subplot(1, 2, 1)
    plt.imshow(rgb_to_srgb(radiance_show))
    plt.subplot(1, 2, 2)
    plt.imshow(alpha_show)
    plt.show()


def training(
    dataset, opt, pipe, testing_iterations, vis_iterations, 
    saving_iterations, checkpoint_iterations, checkpoint, 
    warmup_scales, warmup_iterations, split_blur, rebuttal_scale
):
    first_iter = 0
    # gaussian always init as rf mode, not pbr.
    is_init_as_pbr = False 

    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, resolution_scales=warmup_scales + [2])
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

        print(f"[INFO] restore gaussians from {checkpoint}..., first iter:{first_iter}")
    
    # set background
    if dataset.background == "b":
        # Black background: return a tensor of zeros
        background_func = lambda x, y: torch.zeros(3, device="cuda")  # or device="cpu" if you have cpu
    elif dataset.background == "w":
        # White background: return a tensor of ones
        background_func = lambda x, y: torch.ones(3, device="cuda")  # or device="cpu"
    elif dataset.background == "rand":
        # Random background: return a tensor of random values
        background_func = lambda x, y: torch.rand(3, device="cuda")  # or device="cpu"
    elif dataset.background == "learn":
        # Learned background: assume envmap.sample(vdir) is the correct way to sample
        # The lambda function now correctly accepts two parameters.
        background_func = lambda envmap, vdir: envmap.sample(vdir)
    else:
        raise NotImplementedError(f"Background type '{dataset.background}' not implemented.")
    
    envmap = None
    if dataset.background == "learn" or opt.pbr_from_iter < opt.iterations:
        envmap = OctahedralEnvMap(
            resolution = dataset.envmap_resolution,
            init_value = dataset.envmap_init_value
        )

        envmap.training_setup(opt)
        if checkpoint:
            envmap_chpt_file = checkpoint.replace(".pth", "_envmap.pth")
            try:
                (model_params, first_iter) = torch.load(envmap_chpt_file, weights_only=False)
                envmap.restore(model_params, opt)

                print(f"[INFO] restore envmap from {envmap_chpt_file}, first iter:{first_iter}")
            except:
                print(f"[WARN] Failed to load envmap from checkpoint {envmap_chpt_file}...")            
    
    # surface octahedral probes...
    sops = None
    if opt.pbr_from_iter < opt.iterations:
        sops = SurfaceOctahedralProbes(
            num_probes=dataset.sops_num,
            resolution=dataset.sops_resolution
        )
        sops.training_setup(opt)

        if checkpoint:
            sops_chpt_file = checkpoint.replace(".pth", "_sops.pth")
            try:
                (model_params, first_iter) = torch.load(sops_chpt_file, weights_only=False)
                sops.restore(model_params)

                print(f"[INFO] restore sops from {sops_chpt_file}, first iter:{first_iter}")
            except:
                print(f"[WARN] Failed to load sops from checkpoint {sops_chpt_file}...")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    train_cameras = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(initial=first_iter, total=opt.iterations, desc="Training progress", dynamic_ncols=True)
    first_iter += 1

    mask_blur = None
    if split_blur:
        mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')

    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0 and (len(warmup_iterations) > 0 and iteration > warmup_iterations[-1]):
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            train_cameras = scene.getTrainCameras_warmup(iteration, warmup_scales, warmup_iterations)
            viewpoint_stack = train_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        # Differentiable Muti-Target Rendering Configuration
        use_geometry_prior = any([
            viewpoint_cam.normal is not None, 
            viewpoint_cam.depth is not None
        ])
        use_material_prior = any([
            viewpoint_cam.albedo is not None, 
            viewpoint_cam.metallic is not None, 
            viewpoint_cam.roughness is not None
        ])

        if not use_geometry_prior:
            opt.prior_geometry_loss_from_iter = opt.iterations + 1
        if not use_material_prior:
            opt.prior_material_loss_from_iter = opt.iterations + 1
        
        run_pbr = iteration > opt.pbr_from_iter
        requires_geometry = iteration > min([
            opt.mask_loss_from_iter, 
            opt.d2n_loss_from_iter, 
            opt.prior_geometry_loss_from_iter,
            opt.pbr_from_iter
        ])
        requires_material = iteration > min([
            opt.prior_material_loss_from_iter,
            opt.pbr_from_iter
        ])

        requires_geometry = requires_geometry or requires_material
        requires_geometry = requires_geometry or run_pbr
        requires_material = requires_material or run_pbr
        
        # splatting
        render_pkg = render(
            viewpoint_cam, gaussians, pipe, background_func, envmap, 
            requires_geometry=requires_geometry, 
            requires_material=requires_material,
            requires_render=not run_pbr,
            use_trained_exp=dataset.train_test_exp,
            flag_max_count= (split_blur and iteration < opt.densify_until_iter + 1)
        )
        
        background = render_pkg["background"]
        gt_image = viewpoint_cam.get_gt_rgb(background).cuda()

        # PBR
        pbr_image = None
        if iteration > opt.pbr_from_iter:

            if not is_init_as_pbr:
                gaussians.training_pbr_setup(opt)
            
            if sops.is_empty():
                # NOTE: xyz, rgb, normals in CPU!!!!
                # Reconstruction from multi-view images, depths and normals....
                reconstruct_pipe = Reconstruction(scene.getTrainCameras(scale=1), render, pipe, step=2)
                xyz, rgb, normal = reconstruct_pipe.run(gaussians) # xyz, rgb , normal
                del reconstruct_pipe

                sops.build(xyz, rgb, normal, rebuttal_scale=rebuttal_scale)
                sops.initialize_textures_by_tracing(gaussians)
                sops.training_setup(opt)
                # sops.vis(xyz)
            
            # NOTE: In the pbr stage, we only care about the foreground, not the background
            pbr_background = background.detach()
            pbr_pkg = pbr_shading(
                render_pkg, viewpoint_cam, sops, envmap, pbr_background, 
                num_rays=dataset.num_rays, return_details=False, is_training=True
            )
            pbr_image = pbr_pkg["pbr"]
        
        """
        ############################ Calculate Losses. ########################################
        """
        # render loss
        Ll1, ssim_val = 0, 0
        image = pbr_image if pbr_image is not None else render_pkg["render"]
        prefix = "pbr." if pbr_image is not None else ""
        
        if viewpoint_cam.is_linear_rgb:
            # linear L1
            Ll1 = l1_loss(image, gt_image)
            # Convert to sRGB if the input is linear RGB.
            # NOTE: SSIM is better in sRGB as human vision is non-linear.
            srgb_image = rgb_to_srgb(image)
            srgb_gt = rgb_to_srgb(gt_image)
            ssim_val = fast_ssim(srgb_image[None], srgb_gt[None])
        else:
            # srgb L1
            srgb_image = rgb_to_srgb(image)
            Ll1 = l1_loss(srgb_image, gt_image)
            # srgb ssim and srgb psnr
            ssim_val = fast_ssim(srgb_image[None], gt_image[None])

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
        loss_items = { f"{prefix}l1.loss": Ll1.item(), f"{prefix}dssim.loss": 1.0 - ssim_val.item() }
        
        # Geometric Regularization
        if iteration > opt.d2n_loss_from_iter:
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']

            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            loss_normal = (normal_error).mean()

            loss += opt.lambda_d2n_normal * loss_normal
            loss_items["d2n.loss"] = loss_normal.item()

            if opt.lambda_depth_smooth > 0:
                rend_depth = render_pkg['rend_depth']
                if viewpoint_cam.gt_alpha_mask is not None:
                    image_mask = viewpoint_cam.gt_alpha_mask.float().cuda()
                    loss_depth_smooth = first_order_edge_aware_loss(rend_depth * image_mask, gt_image)
                else:
                    loss_depth_smooth = first_order_edge_aware_loss(rend_depth, gt_image)
                loss_items["loss.geo.depth_smooth"] = loss_depth_smooth.item()
                loss = loss + opt.lambda_depth_smooth * loss_depth_smooth

            if opt.lambda_normal_smooth > 0:
                rend_normal = render_pkg["rend_normal"]
                if viewpoint_cam.gt_alpha_mask is not None:
                    image_mask = viewpoint_cam.gt_alpha_mask.float().cuda()
                    loss_normal_smooth = first_order_edge_aware_loss(rend_normal * image_mask, gt_image)
                else:
                    loss_normal_smooth = first_order_edge_aware_loss(rend_normal, gt_image)
                loss_items["loss.geo.normal_smooth"] = loss_normal_smooth.item()
                loss = loss + opt.lambda_normal_smooth * loss_normal_smooth
            
        if iteration > opt.mask_loss_from_iter:
            rend_alpha = render_pkg['rend_alpha']
            mask_loss = entropy_loss(rend_alpha, viewpoint_cam.gt_alpha_mask.cuda())
            
            loss += opt.lambda_mask * mask_loss
            loss_items["mask.loss"] = mask_loss.item()
        
        # Geometric Prior
        if iteration > opt.prior_geometry_loss_from_iter:

            t = (iteration - opt.prior_geometry_loss_from_iter) / (opt.iterations - opt.prior_geometry_loss_from_iter)
            t = max(0.0, min(1.0, t))  # Clamp t to [0, 1]

            # Linearly interpolate between init and final values
            lambda_prior_normal = opt.lambda_prior_normal_init * (1.0 - t) + opt.lambda_prior_normal_final * t
            # print(lambda_prior_normal)

            if viewpoint_cam.normal is not None:
                rend_normal  = render_pkg['rend_normal'] * 0.5 + 0.5
                prior_normal = viewpoint_cam.normal.cuda() * 0.5 + 0.5
                loss_prior_normal = (1.0 - fast_ssim(rend_normal[None], prior_normal[None]).mean())
                
                loss += lambda_prior_normal * loss_prior_normal
                loss_items["loss.geo.prior_normal"] = loss_prior_normal.item()
            
            if viewpoint_cam.normal is not None:
                surf_normal  = render_pkg['surf_normal'] * 0.5 + 0.5
                prior_d2n_normal = viewpoint_cam.normal.cuda() * 0.5 + 0.5
                loss_prior_d2n_normal = (1.0 - fast_ssim(surf_normal[None], prior_d2n_normal[None]).mean())
                
                loss += lambda_prior_normal * loss_prior_d2n_normal
                loss_items["loss.geo.prior_d2n_normal"] = loss_prior_normal.item()
        
        if iteration > opt.pbr_from_iter:
            lamb_loss = (1.0 - render_pkg['rend_roughness']).mean() + render_pkg['rend_metallic'].mean()
            loss = loss + opt.lambda_lamb * lamb_loss
            loss_items["loss.mat.lamb"] = lamb_loss.item()

            if opt.lambda_albedo_smooth > 0:
                rend_albedo = render_pkg["rend_albedo"]
                if viewpoint_cam.gt_alpha_mask is not None:
                    image_mask = viewpoint_cam.gt_alpha_mask.float().cuda()
                    loss_albedo_smooth = first_order_edge_aware_loss(rend_albedo * image_mask, gt_image)
                else:
                    loss_albedo_smooth = first_order_edge_aware_loss(rend_albedo, gt_image)
                loss_items["loss.mat.albedo_smooth"] = loss_albedo_smooth.item()
                loss = loss + opt.lambda_albedo_smooth * loss_albedo_smooth
            
            if opt.lambda_metallic_smooth > 0:
                rendered_metallic = render_pkg["rend_metallic"]
                if viewpoint_cam.gt_alpha_mask is not None:
                    image_mask = viewpoint_cam.gt_alpha_mask.float().cuda()
                    loss_metallic_smooth = first_order_edge_aware_loss(rendered_metallic * image_mask, gt_image)
                else:
                    loss_metallic_smooth = first_order_edge_aware_loss(rendered_metallic, gt_image)
                loss_items["loss.mat.metallic_smooth"] = loss_metallic_smooth.item()
                loss = loss + opt.lambda_metallic_smooth * loss_metallic_smooth
            
            if opt.lambda_roughness_smooth > 0:
                rendered_roughness = render_pkg["rend_roughness"]
                if viewpoint_cam.gt_alpha_mask is not None:
                    image_mask = viewpoint_cam.gt_alpha_mask.float().cuda()
                    loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness * image_mask, gt_image)
                else:
                    loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness, gt_image)
                loss_items["loss_roughness_smooth"] = loss_roughness_smooth.item()
                loss = loss + opt.lambda_roughness_smooth * loss_roughness_smooth
        
        # Material Prior
        if iteration > opt.prior_material_loss_from_iter:

            if viewpoint_cam.albedo is not None:
                # alignment and then loss
                rend_albedo  = render_pkg['rend_albedo']
                prior_albedo = viewpoint_cam.albedo.cuda()
                # loss_prior_albedo =l1_loss(scaled_albedo, prior_albedo)
                loss_prior_albedo = (1.0 - fast_ssim(rend_albedo[None], prior_albedo[None]).mean())

                loss = loss + opt.lambda_prior_albedo * loss_prior_albedo
                loss_items["loss.mat.prior_albedo"] = loss_prior_albedo.item()
            
            if viewpoint_cam.metallic is not None:
                rend_metallic  = render_pkg['rend_metallic']
                # loss_prior_metallic = l1_loss(rend_metallic, viewpoint_cam.metallic.cuda())
                loss_prior_metallic = (1.0 - fast_ssim(rend_metallic[None], viewpoint_cam.metallic.cuda()[None]).mean())
                
                loss = loss + opt.lambda_prior_metallic * loss_prior_metallic
                loss_items["loss.mat.prior_metallic"] = loss_prior_metallic.item()
            
            if viewpoint_cam.roughness is not None:
                rend_roughness  = render_pkg['rend_roughness']
                # loss_prior_roughness = l1_loss(rend_roughness, viewpoint_cam.roughness.cuda())
                loss_prior_roughness = (1.0 - fast_ssim(rend_roughness[None], viewpoint_cam.roughness.cuda()[None]).mean())
                
                loss = loss + opt.lambda_prior_roughness * loss_prior_roughness
                loss_items["loss.mat.prior_roughness"] = loss_prior_roughness.item()
        
        # SOPs Surpervision
        if iteration > opt.pbr_from_iter:
            # NOTE: only part of light are traced due to limited observation 
            # we need to mask the loss.
            loss_sops_radiance = l1_loss(sops.trace_radiance, sops.tex_radiance)
            loss = loss + opt.lambda_sops * loss_sops_radiance
            loss_items["loss.sops_radiance"] = loss_sops_radiance.item()

            loss_sops_alpha = l1_loss(sops.trace_alpha, sops.tex_alpha)
            loss = loss + opt.lambda_sops * loss_sops_alpha
            loss_items["loss.sops_alpha"] = loss_sops_alpha.item()

        # Light regularization
        if iteration > opt.pbr_from_iter:
            light = envmap.get_data()
            mean_light = light.mean(0, keepdim=True)
            # print(light.shape, mean_light.shape)
            loss_light = l1_loss(light, mean_light)
            
            loss = loss + opt.lambda_light * loss_light
            loss_items["loss.light.lamb"] = loss_light.item()

        # backward
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, loss_items, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, pipe, sops, envmap, dataset.num_rays, train_test_exp=dataset.train_test_exp)
            training_vis(dataset.model_path, iteration, vis_iterations, scene, render, pipe,
                         background_func, sops, envmap, dataset.num_rays, train_test_exp=dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                
                if envmap is not None:
                    oct_envmap_rgb = envmap.export_as_octahedral()
                    oct_envmap_rgb = oct_envmap_rgb.permute(1, 2, 0).detach().cpu().numpy()

                    envmap_file = os.path.join(dataset.model_path, "envmap/iteration_{}/envmap.exr".format(iteration))
                    os.makedirs(os.path.dirname(envmap_file), exist_ok=True)
                    write_exr(envmap_file, oct_envmap_rgb)

                    sph_envmap_rgb = envmap.export_as_spherical()
                    sph_envmap_rgb = sph_envmap_rgb.permute(1, 2, 0).detach().cpu().numpy()
                    sph_envmap_file = os.path.join(dataset.model_path, "envmap/iteration_{}/sph_envmap.exr".format(iteration))
                    write_exr(sph_envmap_file, sph_envmap_rgb)

            
            # Densification
            # print(radii.shape)
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            if iteration < opt.densify_until_iter:

                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                gaussians.radii = radii.unsqueeze(-1)

                if split_blur:
                    area_max = render_pkg["area_max"]
                    # print(area_max.shape, mask_blur.shape)
                    mask_blur = torch.logical_or(mask_blur, area_max>(image.shape[1]*image.shape[2]/5000))

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_prune_threshold, scene.cameras_extent, size_threshold, mask_blur)
                    
                    if split_blur:
                        mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')

                if iteration % opt.opacity_reset_interval == 0 or (dataset.background == "w" and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            
            # Optimizer step
            if iteration < opt.iterations:
                # if gaussians.exposure_optimizer is not None and dataset.train_test_exp:
                #     gaussians.exposure_optimizer.step()
                #     gaussians.exposure_optimizer.zero_grad(set_to_none = True)

                if gaussians.optimizer is not None:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    
                if envmap is not None:
                    envmap.optimizer.step()
                    envmap.optimizer.zero_grad(set_to_none = True)
                
                if sops is not None and not sops.is_empty():
                    if hasattr(sops, "optimizer") and sops.optimizer is not None:
                        sops.optimizer.step()
                        sops.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                if gaussians.optimizer is not None:
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    gaussian_ckpt_file = scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                    torch.save((gaussians.capture(), iteration), gaussian_ckpt_file)
                
                if envmap is not None:
                    envmap_ckpt_file = gaussian_ckpt_file.replace(".pth", "_envmap.pth")
                    torch.save((envmap.capture(), iteration), envmap_ckpt_file)
                
                    if sops is not None and not sops.is_empty():
                        sops_ckpt_file = gaussian_ckpt_file.replace(".pth", "_sops.pth")
                        torch.save((sops.capture(), iteration), sops_ckpt_file)


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def visualize_map(render_map, min_ = None, max_ = None, invalid_ = None):
    valid_map = render_map.clone()
    if invalid_ is not None:
        valid_map = valid_map[valid_map != invalid_]

    min_valid = torch.quantile(valid_map, 0.1) if min_ is None else min_
    max_valid = torch.quantile(valid_map, 0.9) if max_ is None else max_
    normalized_map = (render_map - min_valid) / (max_valid - min_valid)
    normalized_map = normalized_map.clamp(0, 1)

    if normalized_map.shape[0] == 1:
        normalized_map = normalized_map.repeat(3, 1, 1)

    return normalized_map

@torch.no_grad()
def training_vis(
    model_path, iteration, vis_iterations,
    scene : Scene, renderFunc, pipe, background_func, 
    sops:SurfaceOctahedralProbes, envmap:OctahedralEnvMap, num_rays,
    train_test_exp
):

    if iteration in vis_iterations:
        torch.cuda.empty_cache()

        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(0, 5, 1)]}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})   
        
        vis_render_result = os.path.join(model_path, f"vis/iter_{iteration:05}/rgb")
        vis_geo_result = os.path.join(model_path, f"vis/iter_{iteration:05}/geo")
        vis_mat_result = os.path.join(model_path, f"vis/iter_{iteration:05}/mat")
        vis_light_result = os.path.join(model_path, f"vis/iter_{iteration:05}/light")

        os.makedirs(vis_render_result, exist_ok=True)
        os.makedirs(vis_geo_result, exist_ok=True)
        os.makedirs(vis_mat_result, exist_ok=True)
        os.makedirs(vis_light_result, exist_ok=True)

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                for idx, viewpoint in enumerate(config['cameras']):
                    gt_image = viewpoint.original_image.cuda()

                    # render
                    render_pkg = renderFunc(
                        viewpoint, scene.gaussians, pipe, background_func,
                        envmap, requires_geometry=True, requires_material=True,
                        use_trained_exp=train_test_exp
                    )
                    image = render_pkg["render"]
                    background = render_pkg["background"]

                    if sops is not None and not sops.is_empty():
                        pbr_pkg = pbr_shading(
                            render_pkg, viewpoint, sops, envmap, background, 
                            num_rays=num_rays, return_details=True, is_training=False
                        )
                        image = pbr_pkg["pbr"]
                        ao_map = pbr_pkg["ao"]
                        local_map = pbr_pkg["local"]
                        global_map = pbr_pkg["global"]
                    else:
                        ao_map = torch.zeros_like(gt_image[0:1])
                        local_map = torch.zeros_like(gt_image)
                        global_map = torch.zeros_like(gt_image)

                    image = rgb_to_srgb(image) 
                    local_map = rgb_to_srgb(local_map)
                    global_map = rgb_to_srgb(global_map)

                    gt_image = viewpoint.get_gt_rgb(background).cuda()
                    if viewpoint.is_linear_rgb:                        
                        gt_image = rgb_to_srgb(gt_image)
                    
                    render_grid = make_grid([gt_image, image], nrow=2)
                    save_image(render_grid, f"{vis_render_result}/{config['name']}_{viewpoint.image_name}.jpg")

                    # geometric buffers
                    gt_alpha = visualize_map(viewpoint.gt_alpha_mask.cuda(), 0, 1)
                    rend_alpha = visualize_map(render_pkg['rend_alpha'], 0, 1)

                    inv_depth = (1.0 / render_pkg['rend_depth'])
                    inv_depth = torch.nan_to_num(inv_depth, 0.0, 0.0, 0.0)
                    rend_depth = visualize_map(inv_depth)
                    rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                    surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5

                    depth_prior = visualize_map(viewpoint.depth, invalid_=0) if viewpoint.depth is not None else torch.zeros_like(rend_depth)
                    normal_prior = viewpoint.normal * 0.5 + 0.5 if viewpoint.normal is not None else torch.zeros_like(rend_normal)

                    render_grid = make_grid([rend_alpha, rend_depth, rend_normal, surf_normal, 
                                             gt_alpha, depth_prior, normal_prior, torch.zeros_like(surf_normal)], nrow=4)
                    save_image(render_grid, f"{vis_geo_result}/{config['name']}_{viewpoint.image_name}.jpg")

                    # material buffers
                    rend_albedo = render_pkg["rend_albedo"]
                    rend_metallic = visualize_map(render_pkg["rend_metallic"], 0, 1)
                    rend_roughness = visualize_map(render_pkg["rend_roughness"], 0, 1)

                    albedo_prior = viewpoint.albedo if viewpoint.albedo is not None else torch.zeros_like(rend_albedo)
                    metallic_prior = visualize_map(viewpoint.metallic, 0, 1) if viewpoint.metallic is not None else torch.zeros_like(rend_metallic)
                    roughness_prior = visualize_map(viewpoint.roughness, 0, 1) if viewpoint.roughness is not None else torch.zeros_like(rend_roughness)

                    render_grid = make_grid([rend_albedo, rend_metallic, rend_roughness, 
                                             albedo_prior, metallic_prior, roughness_prior], nrow=3)
                    save_image(render_grid, f"{vis_mat_result}/{config['name']}_{viewpoint.image_name}.jpg")

                    # light buffers
                    rend_ao = visualize_map(ao_map, 0, 1)
                    rend_local = local_map
                    rend_global = global_map

                    render_grid = make_grid([rend_ao, rend_local, rend_global], nrow=3)
                    save_image(render_grid, f"{vis_light_result}/{config['name']}_{viewpoint.image_name}.jpg")

        torch.cuda.empty_cache()


@torch.no_grad()
def training_report(
    tb_writer, iteration, loss_items, elapsed, test_iterations, 
    scene : Scene, renderFunc, pipe, 
    sops:SurfaceOctahedralProbes, envmap:OctahedralEnvMap, num_rays,
    train_test_exp
):
    if tb_writer:
        for key, value in loss_items.items():
            if "loss" in key:
                tb_writer.add_scalar('train_loss_patches/' + key, value, iteration)
            else:
                tb_writer.add_scalar(key, value, iteration)
        
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
    
    background_func = lambda x, y: torch.zeros(3, device="cuda")

    # Report test and samples of training set
    if iteration in test_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})
        # validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(0, 5, 1)]}, 
        #                       {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test,psnr_test, ssim_test = 0.0, 0.0, 0.0
                pbr_l1_test, pbr_psnr_test, pbr_ssim_test = 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):

                    render_pkg = renderFunc(
                        viewpoint, scene.gaussians, pipe, background_func, 
                        envmap, requires_geometry=True, requires_material=True,
                        use_trained_exp=train_test_exp
                    )
                    image = render_pkg["render"]
                    background = render_pkg["background"]

                    if sops is not None and not sops.is_empty():
                        pbr_pkg = pbr_shading(
                            render_pkg, viewpoint, sops, envmap, background, 
                            num_rays=num_rays, return_details=False, is_training=False
                        )
                        image = pbr_pkg["pbr"]

                    image = rgb_to_srgb(image)
                    gt_image = viewpoint.get_gt_rgb(background).cuda()
                    if viewpoint.is_linear_rgb:                        
                        gt_image = rgb_to_srgb(gt_image)
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += fast_ssim(image[None], gt_image[None]).mean().double()

                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
        
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 7000, 15000, 30000, 32000])
    parser.add_argument("--vis_iterations", nargs="+", type=int, default=[1000, 7000, 15000, 30000, 30001, 32000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 15000, 30000, 32000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7000, 30000, 32000])
    parser.add_argument("--warmup_scales", nargs="+", type=int, default=[1])
    parser.add_argument("--warmup_iterations", nargs="+", type=int, default = [])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--split_blur", action="store_true")
    parser.add_argument("--rebuttal_scale", type=float, default=2.0)
    args = parser.parse_args(sys.argv[1:])

    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)

    args.save_iterations.append(opt.pbr_from_iter)
    args.save_iterations.append(opt.iterations)
    args.test_iterations.append(opt.iterations)
    args.checkpoint_iterations.append(opt.pbr_from_iter)
    args.checkpoint_iterations.append(opt.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        dataset, opt, pipe, args.test_iterations, args.vis_iterations, 
        args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, 
        args.warmup_scales, args.warmup_iterations, args.split_blur, args.rebuttal_scale
    )
    
    # All done
    print("\nTraining complete.")
