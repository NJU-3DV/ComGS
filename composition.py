
import os
# # NOTE: set some environment variables to avoid increasing reserved memory
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync,max_split_size_mb:512,expandable_segments:True"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

import sys
import torch
import numpy as np
from gaussian_renderer import render
from scene.envmap import SphericalEnvMap
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from scipy.spatial.transform import Rotation as R
import logging
import json
from gaussian_renderer.trace import Tracer
import glob
from utils.pbr_utils import rgb_to_srgb
from cuda_pbr.sample import octahedral_importance_sample_sphere
from cuda_pbr.pbr import physically_based_rendering
from utils.image_utils import box_filter
from utils.image_utils import read_exr
import matplotlib.pyplot as plt
import time
from scene.reconstruction import Reconstruction
from scene.sops import SurfaceOctahedralProbes, sample_from_octahedral_envmaps
import imageio.v2 as imageio
from utils.pbr_utils import rgb_to_srgb
from utils.comp_utils import parse_composition_name, get_object_transform
from gaussian_renderer.trace import trace_scene_360
from pano_light.inference import inference
from pano_light.config import load_config
from pano_light.miscs import save_exr
from utils.render_utils import generate_path, generate_constrained_path
from utils.seed_utils import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(module)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def analyze_similarity_transform(transform: np.ndarray):
    linear = transform[:3, :3]
    axis_scales = np.linalg.norm(linear, axis=1)
    scale = float(axis_scales.mean())
    uniform = np.allclose(axis_scales, scale, atol=1e-5, rtol=1e-4)
    return scale, axis_scales, uniform


def setup_scene(args):
    """
    Set up scene and object data.
    
    Args:
        args: Command line arguments
        dataset: Dataset parameters
        
    Returns:
        tuple: Tuple containing scene dictionary, scene name, and list of object names
    """
    logger.info("Setting up scene data...")
    
    # 00. Sparse scene and object name from composition name
    composition_gt_path = args.source_path
    composition_name = os.path.basename(composition_gt_path)
    scene_name, object_names = parse_composition_name(composition_name)

    logger.debug(f"Composition path: {composition_gt_path}, name: {composition_name}")
    logger.info(f"Composition scene: {composition_name}")
    logger.info(f"Background scene: {scene_name}")
    logger.info(f"Object list: {object_names}")
    
    scene_root = os.path.join(args.workspace, "scenes")
    object_root = os.path.join(args.workspace, "objects")
    
    # 01. Prepare scene data
    scene_ply = os.path.join(scene_root, scene_name, "point_cloud", "iteration_30000", "point_cloud.ply")
    logger.info(f"Using scene point cloud: {scene_ply}")

    scene_dict = {
        "name": composition_name,
        "scene": {
            "name": scene_name,
            "path": scene_ply,
            "transform": np.eye(4).flatten(),
            "albedo_scale": torch.tensor([1.0, 1.0, 1.0], device="cuda", dtype=torch.float32),
        }
    }

    # 02. Prepare transformation data
    transform_file = os.path.join(composition_gt_path, 'transform.json')
    with open(transform_file, 'r') as f:
        transform_data = json.load(f)
        logger.info(f"Loaded transformation data from {transform_file}")
    
    # 03. Prepare object data
    for obj in object_names:
        # (1) Load relightable Gaussian (ply file)
        obj_dir = os.path.join(object_root, obj)
        obj_path = os.path.join(obj_dir, "point_cloud", "iteration_32000", "point_cloud.ply")

        # (2) Load transformation for the object
        obj_transform = get_object_transform(obj, transform_data)
        logging.info(f"Loaded transformation for {obj} as \n{obj_transform}")
        scale, axis_scales, uniform_scale = analyze_similarity_transform(obj_transform)
        if not uniform_scale:
            logger.warning(
                f"Object [{obj}] transform is not uniform-scale. "
                f"SOPS queries assume similarity transforms; axis scales={axis_scales.tolist()}"
            )

        # (3) Load albedo scale 
        # NOTE: Estimating albedo and environment lighting inherently involves scale ambiguity.
        # To achieve realistic object-scene composition, an additional albedo alignment
        # ratio is required.

        # `albedo_ratio.json`: For datasets without ground-truth albedo, this ratio should 
        # be manually specified.

        # `albedo_ratio.pth`: For datasets with ground-truth albedo, this ratio can be 
        # automatically computed from the ground truth and stored.
        albedo_scale_json = os.path.join(obj_dir, "albedo_ratio.json")
        albedo_scale_pth = os.path.join(obj_dir, "albedo_ratio.pth")
        if os.path.exists(albedo_scale_json):
            with open(albedo_scale_json, 'r') as f:
                albedo_scale = torch.tensor(json.load(f), device="cuda", dtype=torch.float32)
            logger.warning(f"Loaded albedo scale {albedo_scale}.")
        elif os.path.exists(albedo_scale_pth):
            albedo_scale = torch.load(albedo_scale_pth).cuda()
            logger.warning(f"Loaded albedo scale {albedo_scale}.")
        else:
            logger.warning(f"No albedo scale found for {obj}, using [1, 1, 1]")
            albedo_scale = torch.tensor([1.0, 1.0, 1.0], device="cuda", dtype=torch.float32)

        # (4) Load envmap and lightmap if available
        envmap_path = os.path.join(args.workspace, "envmap", f"{composition_name}/{obj}.exr")
        lgtmap_path = envmap_path.replace(".exr", "_lgt.exr")

        # get sops file path
        sops_path = os.path.join(obj_dir, "chkpnt32000_sops.pth")

        # (4) Add the object to the scene dictionary
        scene_dict[obj] = {
            "name": obj,
            "path": obj_path,
            "sops_path": sops_path,
            "transform": obj_transform.flatten(),
            "transform_scale": scale,
            "albedo_scale": albedo_scale,
            "envmap": envmap_path,
        }

    logger.info(f"Successfully loaded...")
    
    return scene_dict, scene_name, object_names, composition_name


def load_models(scene_dict, object_names, dataset, args):
    """
    Load Gaussian models and lightings.
    
    Args:
        scene_dict: Scene dictionary
        object_names: List of object names
        dataset: Dataset parameters
        args: Command line arguments
        
    Returns:
        tuple: Tuple containing loaded models and light source information
    """
    logger.info("Loading Gaussian models and detecting light sources...")
    
    # load all composited gaussians
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False, composition=True)
    
    # load only scene points
    logger.info(f"Loading scene: {scene_dict['scene']['path']}")
    gaussians.load_ply(scene_dict["scene"]["path"])
    
    # Load Gaussian models
    object_gaussians_list = []
    object_envmap_list = []
    object_lgtmap_list = []
    object_sops_list = [] if args.with_ao else None
    
    for obj in object_names:
        if obj not in scene_dict:
            logger.warning(f"Object {obj} does not exist in scene dictionary, skipping")
            continue
            
        logger.info(f"Processing object: {obj}")
        object_gaussians = GaussianModel(dataset.sh_degree)
        object_gaussians.load_ply(scene_dict[obj]["path"])
        
        # Apply transformation and albedo scaling
        torch_transform = torch.tensor(scene_dict[obj]["transform"], device="cuda", dtype=torch.float32).reshape(4, 4)
        object_gaussians.set_transform(transform=torch_transform)

        object_gaussians.update_albedo(scene_dict[obj]["albedo_scale"])
        object_gaussians_list.append(object_gaussians)

        # Calculate object center and radius
        center = object_gaussians.get_xyz.mean(dim=0)
        logger.info(f"center of object [{obj}]: {center}")

        # load environment map
        logger.info(f"Loading environment map for {obj} from {scene_dict[obj]['envmap']}")
        envmap_file = scene_dict[obj]["envmap"]
        lgtmap_file = envmap_file.replace(".exr", "_lgt.exr")

        envmap = SphericalEnvMap(act_fn="none")
        lgtmap = SphericalEnvMap(act_fn="none")
        if not os.path.exists(envmap_file):

            logger.info(f"No environment map found; inferring it from scratch...")
            pano_image, _, pano_alpha = trace_scene_360(gaussians, center, resolution=512)
            logger.info(f"Tracing finished...")

            pano_image = pano_image.permute(2, 0, 1) # [3, H, W]
            pano_image = rgb_to_srgb(pano_image)
            pano_mask = pano_alpha.unsqueeze(0) # [1, H, W]

            # inference lighting
            config = load_config(args.config)
            results, runtime = inference(pano_image, pano_mask, args.label, config, args.seed)

            logger.info(f"Edit1 time: {runtime:.2f} seconds")

            os.makedirs(os.path.dirname(envmap_file), exist_ok=True)
            save_exr(lgtmap_file, results['lgt_detected'])
            save_exr(envmap_file, results['hdr_synthetic'])

            show_image = pano_image.permute(1, 2, 0).detach().cpu().numpy()
            show_mask = pano_mask[0].detach().cpu().numpy()
            imageio.imwrite(envmap_file.replace(".exr", ".png"), (show_image * 255).astype(np.uint8))
            imageio.imwrite(envmap_file.replace(".exr", "_mask.png"), (show_mask * 255).astype(np.uint8))

            from torchvision.utils import make_grid, save_image
            from torchvision import transforms
            
            to_tensor = transforms.ToTensor()
            imgs_to_save = [
                results['image_in'][0].cpu(), to_tensor(results['exposure_dicts'][0]), 
                results['image_in'][0].cpu(), to_tensor(results['exposure_dicts'][-2.5]),
                results['image_in'][0].cpu(), to_tensor(results['exposure_dicts'][-5]),
            ]
            grid = make_grid(imgs_to_save, nrow=2)
            save_image(grid, envmap_file.replace(".exr", "_grid.png"))

            if args.vis:
                plt.subplot(1, 2, 1)
                plt.title("image 360")
                plt.imshow(pano_image.permute(1, 2, 0).detach().cpu().numpy())
                plt.subplot(1, 2, 2)
                plt.title("alpha 360")
                plt.imshow(pano_mask.permute(1, 2, 0).detach().cpu().numpy())
                plt.show()
        
        # load environment map and light map
        sph_data = read_exr(envmap_file)[..., :3]
        sph_data = torch.from_numpy(sph_data).to("cuda", dtype=torch.float32)

        lgt_data = read_exr(lgtmap_file)[..., :3]
        lgt_data = torch.from_numpy(lgt_data).to("cuda", dtype=torch.float32)
        
        # NOTE: `sph_data` is the synthesized HDR environment map used as the base illumination.
        # It is built by tracing a 360-degree panorama from the object center in the GS scene,
        # followed by multi-exposure prediction and HDR merging. This provides stable
        # full-environment lighting but tends to attenuate compact high-intensity sources:
        # GS+SH radiance is biased toward smooth low-frequency appearance, and diffusion-based
        # HDR prediction can further smooth or under-estimate point lights, lamps, windows,
        # and the sun.
        #
        # `lgt_data` is therefore treated as an explicit high-frequency light component, not
        # a standalone environment map. It compensates for compact directional lights that
        # are critical for specular response and cast-shadow formation.
        #
        # We keep `sph_data` as the base for both branches and inject `lgt_data` with
        # task-dependent modulation. Relighting uses softer injection to avoid double-counting
        # and over-exposed highlights; shadow estimation uses stronger injection because cast
        # shadows depend on directional high-intensity lights, while the base `sph_data` still
        # provides ambient/indirect fill to avoid unrealistically black hard shadows.
        #
        # Tuning: raise relight_light_mod for flatter relighting, lower it for excessive
        # highlights; raise shadow_light_mod for weak shadows, lower it for overly dark or
        # hard shadows.
        relighting_data = sph_data + args.relight_light_mod * lgt_data
        shadow_data = sph_data + args.shadow_light_mod * lgt_data

        envmap.set_data(relighting_data.permute(2, 0, 1))
        lgtmap.set_data(shadow_data.permute(2, 0, 1))

        if args.vis:
            envmap_show = envmap.export_as_spherical()
            lgtmap_show = lgtmap.export_as_spherical()
            plt.subplot(1, 2, 1)
            plt.imshow(envmap_show.permute(1, 2, 0).detach().cpu().numpy())
            plt.subplot(1, 2, 2)
            plt.imshow(lgtmap_show.permute(1, 2, 0).detach().cpu().numpy())
            plt.show()
        
        object_envmap_list.append(envmap)
        object_lgtmap_list.append(lgtmap)

        if args.with_ao:
            sops_path = scene_dict[obj]["sops_path"]
            if not os.path.exists(sops_path):
                raise FileNotFoundError(f"Object AO is enabled but SOPs checkpoint was not found: {sops_path}")

            sops = SurfaceOctahedralProbes()
            logger.info(f"Loading object AO SOPs from {sops_path}")
            model_params, _ = torch.load(sops_path)
            sops.restore(model_params)
            torch_transform = torch.tensor(scene_dict[obj]["transform"], device="cuda", dtype=torch.float32).reshape(4, 4)
            sops.set_query_pose(transform=torch_transform)
            object_sops_list.append(sops)

    return (gaussians, scene, object_gaussians_list, object_envmap_list, object_lgtmap_list, object_sops_list)



def search_shadow_on_plane(obj_gaussians, viewpoint, scene_gbuffer, obj_sampled_pkg, sops):

    depth_buffer = scene_gbuffer["rend_depth"].permute(1, 2, 0)         # [H, W, 1]
    normal_buffer = scene_gbuffer["rend_normal"].permute(1, 2, 0)       # [H, W, 3]
    alpha_buffer = scene_gbuffer["rend_alpha"].permute(1, 2, 0)       # [H, W, 1]
    normal_buffer = normal_buffer / alpha_buffer.clamp_min(1e-6)
    normal_buffer = torch.nn.functional.normalize(normal_buffer, dim=-1) # [H, W, 3]

    rays_with_length = viewpoint.get_cam_rays(world_frame=True) # [H, W, 4]
    range_rays_step = rays_with_length[..., :3] * rays_with_length[..., -1:] # [H, W, 3]
    xyz = range_rays_step * depth_buffer + viewpoint.camera_center[None] # [H, W, 3]

    # (1) search shadow only on the plane (sops)
    bbx_min, bbx_max = sops.get_bbox_world()
    bbox_pad = 10 * sops.voxel_size * sops.query_scale
    bbx_min[-1] = bbx_min[-1] - bbox_pad
    bbx_max[-1] = bbx_max[-1] + bbox_pad

    condition = torch.all((xyz > bbx_min[None, None]) & (xyz < bbx_max[None, None]), dim=-1)

    # (2) the vector from the shadow-casting position to the object should 
    # form a non-obtuse angle with the normal at that position
    objrct_xyz = obj_gaussians.get_xyz
    max_xyz = objrct_xyz.max(dim=0)[0]
    min_xyz = objrct_xyz.min(dim=0)[0]
    center = (max_xyz + min_xyz) / 2

    vector= center[None, None] - xyz
    vector = torch.nn.functional.normalize(vector, dim=-1)
    condition_dir = torch.sum(normal_buffer * vector, dim=-1) >= -1e-8

    condition = torch.logical_and(condition, condition_dir)

    # get latent shadow points
    shade_xyz = xyz[condition] # [S, 3]
    shade_normal = normal_buffer[condition] # [S, 3]

    if shade_xyz.shape[0] <= 0:
        logger.warning(f"No valid points found for {viewpoint.image_name}, skipping shadow calculation")
        return torch.ones_like(scene_gbuffer["rend_depth"].squeeze(0)) # [H, W]
    
    # get global lighting
    rays = obj_sampled_pkg["lgt_rays"].unsqueeze(0).repeat(shade_normal.shape[0], 1, 1) # [P, num_rays, 3]
    inv_pdf = obj_sampled_pkg["lgt_inv_pdf"].unsqueeze(0).repeat(shade_normal.shape[0], 1, 1) # [P, num_rays, 1]
    radiance = obj_sampled_pkg["lgt_radiance"].unsqueeze(0).repeat(shade_normal.shape[0], 1, 1) # [P, num_rays, 3]

    shade_normal = shade_normal.unsqueeze(1) # [P, 1 3]
    n_dot_l = (shade_normal * rays).sum(dim=-1, keepdim=True).clamp(0)
    transport = radiance * inv_pdf * n_dot_l # [S, num_rays, 3]

    texture_alpha = sops.query_visibility_knn(shade_xyz, shade_normal.squeeze(1), k=4) # [S, res, res, 1]
    visibility = sample_from_octahedral_envmaps(texture_alpha, sops.world_to_local_dirs(rays)) # [S, num_rays, 1]

    if args.vis:
        # Visualize the first texture alpha map and its statistics
        plt.figure(figsize=(12, 4))
        # Plot the texture alpha map
        plt.subplot(1, 2, 1)
        plt.title("Texture Alpha [0]")
        tex_alpha_0 = texture_alpha[0].squeeze(-1).detach().cpu().numpy()
        plt.imshow(tex_alpha_0)
        plt.colorbar()
        # Plot histogram and statistics
        plt.subplot(1, 2, 2)
        plt.title("Value Distribution")
        plt.hist(tex_alpha_0.flatten(), bins=50)
        plt.grid(True, alpha=0.3)
        # Calculate and display statistics
        min_val = tex_alpha_0.min()
        max_val = tex_alpha_0.max()
        mean_val = tex_alpha_0.mean()
        median_val = np.median(tex_alpha_0)
        std_val = tex_alpha_0.std()
        stats_text = f"Min: {min_val:.4f}\nMax: {max_val:.4f}\nMean: {mean_val:.4f}\n" \
                    f"Median: {median_val:.4f}\nStd: {std_val:.4f}"
        plt.text(0.02, 0.6, stats_text, transform=plt.gca().transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.show()
        # Print statistics to console as well
        logger.info(f"Texture Alpha [0] stats - Min: {min_val:.4f}, Max: {max_val:.4f}, " 
                f"Mean: {mean_val:.4f}, Median: {median_val:.4f}, Std: {std_val:.4f}")
    
    before_transport = transport.mean(dim=-2) # [S, num_rays, 3] -> [S, 3]
    after_transport = (transport * visibility).mean(dim=-2) # [S, num_rays, 3] -> [S, 3]

    shadow_map = torch.ones_like(scene_gbuffer["rend_normal"]) # [3, H, W]

    # condition [H, W], after_transport [S, 3], before_transport [S, 3], shadow_map [3, H, W]
    shadow_map = shadow_map.permute(1, 2, 0)
    shadow_map[condition] = after_transport / before_transport.clamp_min(1e-6) # [H, W, 3]
    shadow_map = shadow_map.permute(2, 0, 1) # [3, H, W]

    if args.vis:
        plt.imshow(shadow_map.permute(1, 2, 0).detach().cpu().numpy())
        plt.show()

    return shadow_map # [H, W, 3]


def crop_points(xyz, rgb, normals, object_gaussians, scale=1):
    """Crop points based on the bounding box of the object Gaussian.

    Args:
        xyz (torch.Tensor): Point cloud coordinates.
        rgb (torch.Tensor): Point cloud colors.
        normals (torch.Tensor): Point cloud normals.
        object_gaussians: Object Gaussian model.
        scale (float): Scale factor for the bounding box.

    Returns:
        tuple: Cropped point cloud coordinates, colors, and normals.
    """
    objrct_xyz = object_gaussians.get_xyz
    max_xyz = objrct_xyz.max(dim=0)[0]
    min_xyz = objrct_xyz.min(dim=0)[0]

    logging.info(f"xyz extent: {min_xyz} - {max_xyz}")

    center = (max_xyz + min_xyz) / 2
    radius = (max_xyz - min_xyz) / 2 * scale

    bbox = torch.cat([center - radius, center + radius], dim=0).cpu().numpy()

    mask = torch.logical_and(
        torch.logical_and(xyz[:, 0] > bbox[0], xyz[:, 0] < bbox[3]),
        torch.logical_and(xyz[:, 1] > bbox[1], xyz[:, 1] < bbox[4]),
    )
    mask = torch.logical_and(
        mask, 
        torch.logical_and(xyz[:, 2] > bbox[2], xyz[:, 2] < bbox[5])
    )
    
    vector= center[None] - xyz
    vector = torch.nn.functional.normalize(vector, dim=-1)
    mask_dir = torch.sum(normals * vector, dim=-1) >= -1e-8

    mask = torch.logical_and(mask, mask_dir)
    
    return xyz[mask], rgb[mask], normals[mask]


def trace_shadow_on_plane(viewpoint, obj_gaussians, scene_gbuffer, obj_sampled_pkg, tracer, args):
    """Generate shadow using ray tracing method.
    
    Args:
        viewpoint: Camera viewpoint
        obj_gaussians: Object Gaussian model
        scene_gbuffer: G-buffer of the scene
        obj_sampled_pkg: Sampled package for the object
        tracer: Tracer object
        args: Command line arguments
        
    Returns:
        torch.Tensor: Shadow map tensor
    """
    # Extract buffers from G-buffer
    depth_buffer = scene_gbuffer["rend_depth"].permute(1, 2, 0)         # [H, W, 1]
    normal_buffer = scene_gbuffer["rend_normal"].permute(1, 2, 0)       # [H, W, 3]
    alpha_buffer = scene_gbuffer["rend_alpha"].permute(1, 2, 0)         # [H, W, 1]
    normal_buffer = normal_buffer / alpha_buffer.clamp_min(1e-6)
    normal_buffer = torch.nn.functional.normalize(normal_buffer, dim=-1) # [H, W, 3]

    # Get camera rays and world coordinates
    rays_with_length = viewpoint.get_cam_rays(world_frame=True) # [H, W, 4]
    range_rays_step = rays_with_length[..., :3] * rays_with_length[..., -1:] # [H, W, 3]
    xyz = range_rays_step * depth_buffer + viewpoint.camera_center[None] # [H, W, 3]

    # Get object bounding box for shadow area determination
    object_xyz = tracer.gaussians.get_xyz
    max_xyz = object_xyz.max(dim=0)[0]
    min_xyz = object_xyz.min(dim=0)[0]
    center = (max_xyz + min_xyz) / 2
    radius = (max_xyz - min_xyz) / 2 * args.latent_scale

    # (1) search shadow only on the plane using bounding box
    bbx_min, bbx_max = center - 1.3 * radius, center + 1.3 * radius
    bbx_min[-1] = bbx_min[-1] + radius[-1]
    bbx_max[-1] = bbx_max[-1] - radius[-1]

    condition = torch.all((xyz > bbx_min[None, None]) & (xyz < bbx_max[None, None]), dim=-1)

    # (2) Filter points where the normal and direction to object make non-obtuse angle
    objrct_xyz = obj_gaussians.get_xyz
    max_xyz = objrct_xyz.max(dim=0)[0]
    min_xyz = objrct_xyz.min(dim=0)[0]
    center = (max_xyz + min_xyz) / 2

    vector = center[None, None] - xyz
    vector = torch.nn.functional.normalize(vector, dim=-1)
    condition_dir = torch.sum(normal_buffer * vector, dim=-1) >= -1e-8

    # Combine both conditions
    condition = torch.logical_and(condition, condition_dir)

    shade_xyz = xyz[condition] # [S, 3]
    shade_normal = normal_buffer[condition] # [S, 3]

    if shade_xyz.shape[0] <= 0:
        logger.warning(f"No valid points found for {viewpoint.image_name}, skipping shadow calculation")
        return torch.ones_like(scene_gbuffer["rend_normal"]) # [3, H, W]
    
    # Get global lighting
    rays = obj_sampled_pkg["lgt_rays"].unsqueeze(0).repeat(shade_normal.shape[0], 1, 1) # [P, num_rays, 3]
    inv_pdf = obj_sampled_pkg["lgt_inv_pdf"].unsqueeze(0).repeat(shade_normal.shape[0], 1, 1) # [P, num_rays, 1]
    radiance = obj_sampled_pkg["lgt_radiance"].unsqueeze(0).repeat(shade_normal.shape[0], 1, 1) # [P, num_rays, 3]

    shade_normal = shade_normal.unsqueeze(1) # [P, 1, 3]
    n_dot_l = (shade_normal * rays).sum(dim=-1, keepdim=True).clamp(0)
    transport = radiance * inv_pdf * n_dot_l # [S, num_rays, 3]

    # Trace visibility
    trace_pkg = tracer.trace(shade_xyz.unsqueeze(1).expand_as(rays), rays) # [S, num_rays, 1]
    visibility = 1 - trace_pkg["alpha"] # [S, num_rays, 1]

    before_transport = transport.mean(dim=-2) # [S, 3]
    after_transport = (transport * visibility).mean(dim=-2) # [S, 3]

    # Create shadow map
    shadow_map = torch.ones_like(scene_gbuffer["rend_normal"]) # [3, H, W]
    shadow_map = shadow_map.permute(1, 2, 0)
    shadow_map[condition] = after_transport / before_transport.clamp_min(1e-6) # [H, W, 3]
    shadow_map = shadow_map.permute(2, 0, 1) # [3, H, W]

    if args.vis:
        plt.imshow(shadow_map.permute(1, 2, 0).detach().cpu().numpy())
        plt.show()

    return shadow_map # [3, H, W]


def pbr_shading(render_pkg, viewpoint_cam, obj_sampled_pkg:dict, background, obj_sops=None, ao_relaxation=0.8):

    # create PBR_image
    pbr_image = torch.zeros_like(render_pkg["rend_normal"]) # [3, H, W]

    # Deferred PBR shading
    depth_buffer = render_pkg["rend_depth"].squeeze(0)
    # (1) Get the shading points
    mask = depth_buffer > 1e-7 # [H, W, 1]

    if mask.sum() <= 0:
        return pbr_image
    
    # normalize normal buffers
    normal_buffer = render_pkg["rend_normal"].permute(1, 2, 0)       # [H, W, 3]
    alpha_buffer = render_pkg["rend_alpha"].permute(1, 2, 0)       # [H, W, 1]
    normal_buffer = normal_buffer / alpha_buffer.clamp_min(1e-6)
    normal_buffer = torch.nn.functional.normalize(normal_buffer, dim=-1) # [H, W, 3]

    albedo_buffer = render_pkg["rend_albedo"].permute(1, 2, 0)       # [H, W, 3]
    metallic_buffer = render_pkg["rend_metallic"].permute(1, 2, 0)   # [H, W, 1]
    roughness_buffer = render_pkg["rend_roughness"].permute(1, 2, 0) # [H, W, 1]

    shade_normal = normal_buffer[mask]
    shade_albedo = albedo_buffer[mask]
    shade_metallic = metallic_buffer[mask]
    shade_roughness = roughness_buffer[mask]

    # (2) Get important sampled rays
    rays = obj_sampled_pkg["rays"].unsqueeze(0).repeat(shade_normal.shape[0], 1, 1) # [P, num_rays, 3]
    inv_pdf = obj_sampled_pkg["inv_pdf"].unsqueeze(0).repeat(shade_normal.shape[0], 1, 1) # [P, num_rays, 1]
    radiance = obj_sampled_pkg["radiance"].unsqueeze(0).repeat(shade_normal.shape[0], 1, 1) # [P, num_rays, 3]

    rays_with_length = viewpoint_cam.get_cam_rays(world_frame=True)
    range_rays = rays_with_length[..., :3] * rays_with_length[..., -1:]
    vdir = - range_rays[mask]

    if obj_sops is not None:
        shade_depth = depth_buffer[mask]
        shade_range_rays = range_rays[mask] # [P, 3]
        shade_xyz = shade_range_rays * shade_depth[:, None] + viewpoint_cam.camera_center[None] # [P, 3]

        probe_visibility = obj_sops.query_visibility_knn(
            shade_xyz, shade_normal
        )

        sampled_probe_visibility = sample_from_octahedral_envmaps(
            probe_visibility, obj_sops.world_to_local_dirs(rays)
        ) # [P, num_rays, 1]

        # NOTE: Raw self-visibility only describes how the object occludes direct incoming
        # radiance from each direction; it does not include residual indirect or ambient
        # fill in self-occluded regions. We therefore interpolate the raw visibility with
        # full visibility and retain only `ao_relaxation` of the self-occlusion strength.
        #
        # Tuning: increase ao_relaxation for stronger self-AO/contact darkening; decrease
        # it when self-occluded regions become too dark, dirty, or overly hard.
        sampled_probe_visibility = 1 - (1 - sampled_probe_visibility) * ao_relaxation
        radiance = radiance * sampled_probe_visibility # [P, num_rays, 3]

    # (3) PBR shading
    shading_result = physically_based_rendering(
        shade_normal, shade_albedo, shade_metallic,
        shade_roughness, rays, radiance, inv_pdf,
        vdir, cuda_backend=True
    )

    # (4) PBR * render_alpha + background * (1 - render_alpha)
    pbr_image = pbr_image.permute(1, 2, 0) # [H, W, 3]
    pbr_image[mask] = shading_result # mask [H, W]
    pbr_image = pbr_image.permute(2, 0, 1) # [3， H， W]
    pbr_image = pbr_image * render_pkg["rend_alpha"] + background[:, None, None] * (1 - render_pkg["rend_alpha"])

    return pbr_image


def layer_composition(layer_images, layer_depths, layer_alphas):
    """Fuses a set of images based on their depth and alpha values using PyTorch.

    Args:
        layer_images (list of torch.Tensor): A list of N images, each with shape [C, H, W].
        layer_depths (list of torch.Tensor): A list of N depth maps, each with shape [1, H, W].
        layer_alphas (list of torch.Tensor): A list of N alpha maps, each with shape [1, H, W].

    Returns:
        torch.Tensor: The fused image with shape [C, H, W].
    """
    num_layers = len(layer_images)

    # Initialize the fused image and depth map with the first layer
    fused_image = layer_images[0].clone()
    fused_depth = layer_depths[0].clone()

    for i in range(1, num_layers):
        image = layer_images[i]

        depth = layer_depths[i]
        alpha = layer_alphas[i]

        tmp_depth = depth / alpha
        tmp_depth = torch.nan_to_num(tmp_depth, nan=0, posinf=0, neginf=0)  # Handle NaN and Inf values

        tmp_fused_depth = fused_depth / alpha
        tmp_fused_depth = torch.nan_to_num(tmp_fused_depth, nan=0, posinf=0, neginf=0)  # Handle NaN and Inf values

        occlusion = (tmp_depth < tmp_fused_depth).float()
        occlusion = box_filter(occlusion.float(), 3)
        alpha = occlusion * alpha

        fused_image = image * alpha + fused_image * (1 - alpha)
        fused_depth = depth * alpha + fused_depth * (1 - alpha)  # Update depth

    return fused_image, fused_depth


def render_scene_with_shadows(
    cameras, gaussians, object_gaussians_list, 
    object_envmap_list, object_lgtmap_list, 
    pipe, args, object_sops_list=None
):
    logger.info("Starting scene rendering...")
    logger.info(f"Found {len(cameras)} camera viewpoints")
    
    # Prepare output directories
    render_time = []
    image_frames = []
    
    edit_start_time = time.time()

    sops = None
    if not args.with_trace:
        # Prepare resources for shadow generation
        # SOPs is always used as the default method
        logger.info("Preparing SOPs resources for shadow generation...")
        
        # Reconstruct scene point cloud
        # NOTE: if number of cameras is more than 100, we use uniform sampling of cameras
        selected_cameras = cameras.copy()
        if len(selected_cameras) > 100:
            logger.info("More than 100 cameras, using uniform sampling...")
            sampled_indices = np.linspace(0, len(selected_cameras) - 1, 100).astype(int)
            selected_cameras = [selected_cameras[i] for i in sampled_indices]

        reconstruct_pipe = Reconstruction(selected_cameras, render, pipe, step=4)
        xyz, rgb, normals = reconstruct_pipe.run(gaussians)
        xyz, rgb, normals = xyz.cuda(), rgb.cuda(), normals.cuda()
        del reconstruct_pipe

        # Create SOPs
        sops = SurfaceOctahedralProbes(
            num_probes=dataset.sops_num, 
            resolution=dataset.sops_resolution
        )
        
        print(f"total points: {xyz.shape[0]}")
        # Get potential shadow area
        roi_xyz, roi_rgb, roi_normals = crop_points(
            xyz, rgb, normals, 
            object_gaussians_list[0], 
            scale=args.latent_scale
        )
        print(f"roi points: {roi_xyz.shape[0]}")

        # Build and initialize SOPs
        sops.build(roi_xyz, roi_rgb, roi_normals)
        sops.initialize_textures_by_tracing(object_gaussians_list[0])

        if args.vis:
            # Visualize SOPs
            roi_rgb = torch.where(
                roi_rgb > 0.0031308, 
                torch.pow(torch.max(roi_rgb, torch.tensor(0.0031308)), 1.0 / 2.4) * 1.055 - 0.055, 12.92 * roi_rgb
            )
            roi_rgb = roi_rgb.clamp(0, 1)
            sops.vis(roi_xyz, roi_rgb)
        
        # Clean up unused resources
        del sops.trace_radiance, sops.trace_alpha
        del xyz, rgb, normals, roi_xyz, roi_rgb, roi_normals

    # important sampling on the environment maps
    # rays: [S, num_rays, 3], inv_pdf: [S, num_rays, 1]
    obj_sampled_list = []
    for envmap, lgtmap in zip(object_envmap_list, object_lgtmap_list):

        # sample envmap
        octmap = envmap.export_as_octahedral()
        rays, inv_pdf = octahedral_importance_sample_sphere(
            octmap, dataset.num_rays, random_offset=False
        )
        radiance = envmap.sample(rays.permute(1, 0)).permute(1, 0)
        
        # sample lgtmap
        oct_lgtmap = lgtmap.export_as_octahedral()
        lgt_rays, lgt_inv_pdf = octahedral_importance_sample_sphere(
            oct_lgtmap, int(dataset.num_rays / 4), random_offset=False
        )
        lgt_radiance = lgtmap.sample(lgt_rays.permute(1, 0)).permute(1, 0)

        sampled_pkg = {
            "rays": rays,
            "inv_pdf": inv_pdf,
            "radiance": radiance,
            "lgt_rays": lgt_rays,
            "lgt_inv_pdf": lgt_inv_pdf,
            "lgt_radiance": lgt_radiance,
        }
        obj_sampled_list.append(sampled_pkg)

    edit_end_time = time.time()
    edit_time = edit_end_time - edit_start_time
    logger.info(f"Edit2 time: {edit_time:.2f} seconds")
    
    # Initialize resources for shadow processing
    shadow_resources = {}
    
    # Add SOPs resource if available
    if sops is not None:
        shadow_resources["sops"] = sops
        logger.info("Using SOPs for shadow generation")
    
    # Initialize tracer for trace-based shadows if needed
    tracer = None
    if args.with_trace:
        logger.info("Initializing tracer for shadow generation...")
        tracer = Tracer(object_gaussians_list[0])
        shadow_resources["tracer"] = tracer

    for view_idx, viewpoint in enumerate(cameras):

        start_time = time.time()

        logger.debug(f"Rendering viewpoint {view_idx+1}/{len(cameras)}...")
        
        # Prepare image, depth, and alpha buffers for each layer
        layer_images, layer_depths, layer_alphas = [], [], []
        
        # Render background scene
        background_func = lambda x, y: torch.zeros(3, device="cuda", dtype=torch.float32)
        logger.debug("Rendering background scene...")
        render_pkg1 = render(
            viewpoint, gaussians, pipe, 
            background_func, 
            requires_geometry=True, 
            requires_material=False
        )
        
        layer_images.append(render_pkg1["render"])
        layer_depths.append(render_pkg1["rend_depth"])
        layer_alphas.append(render_pkg1["rend_alpha"])

        # for empty frames
        image = render_pkg1["render"]
        image = rgb_to_srgb(image)

        # imageio.imwrite(f"temp_{view_idx:03d}_scene.png", (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        
        # Render individual objects
        for obj_idx, (obj_gaussians, obj_sampled_pkg) in enumerate(zip(object_gaussians_list, obj_sampled_list)):
            obj_sops = object_sops_list[obj_idx] if object_sops_list is not None else None
            
            logger.debug(f"Rendering object {obj_idx+1}/{len(object_gaussians_list)}...")

            # object PBR
            render_pkg2 = render(
                viewpoint, obj_gaussians, pipe, 
                background_func, 
                requires_render=False,
                requires_geometry=True, 
                requires_material=True
            )
            background = render_pkg2["background"]

            pbr_image = pbr_shading(
                render_pkg2, viewpoint, obj_sampled_pkg, background,
                obj_sops=obj_sops, ao_relaxation=args.ao_relaxation
            )
            layer_images.append(pbr_image)
            layer_depths.append(render_pkg2["rend_depth"])
            layer_alphas.append(render_pkg2["rend_alpha"])
            
            # Also add trace-based shadow if enabled
            if args.with_trace:
                logger.debug("Adding trace-based shadow...")
                shadow_shading = trace_shadow_on_plane(viewpoint, obj_gaussians, render_pkg1, obj_sampled_pkg, shadow_resources["tracer"], args)
            else:
                # Generate shadow [H, W, 3]
                # Using SOPs method as default
                logger.debug("Generating shadow using SOPs method...")
                shadow_shading = search_shadow_on_plane(obj_gaussians, viewpoint, render_pkg1, obj_sampled_pkg, shadow_resources["sops"])
            
            # Apply shadow to background layer
            layer_images[0] = layer_images[0] * shadow_shading

        logger.debug("Depth composition...")
        image, depth = layer_composition(layer_images, layer_depths, layer_alphas)
        image = rgb_to_srgb(image)

        end = time.time()

        if view_idx > 5:
            render_time.append(end - start_time)

        if args.vis:
            plt.subplot(3, 3, 1)
            plt.title("scene render")
            plt.imshow(rgb_to_srgb(render_pkg1["render"]).permute(1, 2, 0).cpu().numpy())
            plt.subplot(3, 3, 2)
            plt.title("scene render with shadow")
            plt.imshow(rgb_to_srgb(layer_images[0]).permute(1, 2, 0).cpu().numpy())
            plt.subplot(3, 3, 3)
            plt.title("scene depth")

            show_scene_depth = layer_depths[0].permute(1, 2, 0).cpu().numpy()
            show_scene_depth[show_scene_depth <= 0] = np.nan
            valid_value = show_scene_depth[~np.isnan(show_scene_depth)]
            robust_min = np.percentile(valid_value, 0.1)
            robust_max = np.percentile(valid_value, 99.9)
            show_scene_depth = (show_scene_depth - robust_min) / (robust_max - robust_min)
            show_scene_depth = np.clip(show_scene_depth, 0, 1)

            plt.imshow(show_scene_depth)

            plt.subplot(3, 3, 4)
            plt.title("scene alpha")
            plt.imshow(layer_alphas[0].permute(1, 2, 0).cpu().numpy())
            plt.subplot(3, 3, 5)
            plt.title("object render")
            plt.imshow(rgb_to_srgb(layer_images[1]).permute(1, 2, 0).cpu().numpy())
            plt.subplot(3, 3, 6)
            plt.title("object depth")
            show_object_depth = layer_depths[1].permute(1, 2, 0).cpu().numpy()
            show_object_depth[show_object_depth <= 0] = np.nan
            
            valid_value = show_object_depth[~np.isnan(show_object_depth)]
            robust_min = np.percentile(valid_value, 0.1)
            robust_max = np.percentile(valid_value, 99.9)
            show_object_depth = (show_object_depth - robust_min) / (robust_max - robust_min)
            show_object_depth = np.clip(show_object_depth, 0, 1)

            plt.imshow(show_object_depth)
            plt.subplot(3, 3, 7)
            plt.title("object alpha")
            plt.imshow(layer_alphas[1].permute(1, 2, 0).cpu().numpy())
            plt.subplot(3, 3, 8)
            plt.title("full image")
            plt.imshow(image.permute(1, 2, 0).cpu().numpy())
            plt.subplot(3, 3, 9)
            plt.title("full depth")
            plt.imshow(depth.permute(1, 2, 0).cpu().numpy())
            plt.show()

        # Post-process and save image
        image = image.clamp(0, 1).detach().permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image_frames.append(image)

    print(f"Average render time: {np.mean(render_time)}, Average FPS: {1/np.mean(render_time)}")
    
    logger.info(f"Rendering complete, generated {len(image_frames)} image frames")
    
    return image_frames


def save_output(image_frames, output_filedir, fps=5):
    """
    Save rendering results to video and individual image frames.
    
    Args:
        image_frames: List of rendered image frames
        output_filedir: Output video filename, output_filedir/video.mp4, output_filedir/frames/{idx:05d}.jpg
    """
    logger.info("Saving rendering results...")

    os.makedirs(output_filedir, exist_ok=True)
    
    # Save as video
    output_filename = os.path.join(output_filedir, "video.mp4")
    imageio.mimwrite(output_filename, image_frames, fps=fps, codec='libx264')
    logger.info(f"Video saved to {output_filename}")
    
    # Save individual frames
    frames_dir = os.path.join(output_filedir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    for idx, image in enumerate(image_frames):
        frame_filename = f"{frames_dir}/{idx:05d}.png"
        imageio.imwrite(frame_filename, image)

    logger.info(f"Individual frames saved to {frames_dir} directory")


if __name__ == "__main__":

    parser = ArgumentParser(description="Composition")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--workspace", type=str, default=r"./exp/syncom", help="Workspace directory for reconstruction")
    parser.add_argument("--vis", action="store_true", default=False, help="Visualize intermediate results")
    parser.add_argument("--latent_scale", type=float, default=4.0, help="Scale for latent shadow space")
    parser.add_argument("--relight_light_mod", type=float, default=0.3, help="Modulation strength of detected light for object relighting")
    parser.add_argument("--shadow_light_mod", type=float, default=1.0, help="Modulation strength of detected light for shadow estimation")
    parser.add_argument("--ao_relaxation", type=float, default=0.8, help="Retained strength of object self-AO visibility attenuation in [0, 1]")
    
    # Shadow method selection
    parser.add_argument("--with_trace", action="store_true", default=False, help="Enable trace-based shadow rendering along with SOPs")
    parser.add_argument("--with_ao", action="store_true", default=False, help="Enable object-level SOPs ambient occlusion in PBR shading")

    # pano_light
    parser.add_argument("--config", type=str, default='pano_light/configs/inference.yaml', required=False, help="Path to config file")
    parser.add_argument("--label", type=str, required=True, choices=["indoor", "outdoor"], help="Label for scene type")

    # camera
    parser.add_argument("--generate_cameras", action="store_true", default=False, help="Generate camera, otherwise load from file")
    parser.add_argument("--generate_constrained_cameras", action="store_true", default=False, help="Generate constrained camera paths")
    parser.add_argument("--num_frames", type=int,default=480, help="Number of generated render frames.")

    # output_fps
    parser.add_argument("--fps", type=int, default=5, help="Output video fps")
    parser.add_argument("--output_dir", type=str, default="", help="Optional explicit output directory for rendered composition")
    parser.add_argument("--seed", type=int, default=48, help="Unified global seed for envmap generation")

    args = parser.parse_args(sys.argv[1:])
    if not 0.0 <= args.ao_relaxation <= 1.0:
        raise ValueError("--ao_relaxation must be in [0, 1].")

    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    
    logger.info("Starting object-scene composition...")

    set_global_seed(args.seed)
    logger.info("Random seed set to {}".format(args.seed))

    with torch.no_grad():

        # Step 1: Set up scene data
        scene_dict, scene_name, object_names, composition_name = setup_scene(args)

        # Step 2: Load models and detect light sources
        (
            gaussians, scene, object_gaussians_list, 
            envmap_list, lgtmap_list, object_sops_list
        ) = load_models(
            scene_dict, object_names, dataset, args
        )

        cameras = scene.getTrainCameras()
        # NOTE: Camera path selection:
        # - `generate_cameras` is used when the input capture sufficiently surrounds the scene.
        #   In this case, the observed cameras provide enough coverage to synthesize a smooth
        #   orbit-style rendering path around the scene.
        # - `generate_constrained_cameras` is used when the capture is limited, sparse, or only
        #   covers a restricted viewing range. Instead of forcing a full orbit, it constrains
        #   the rendering path to the available or manually selected views, avoiding poorly
        #   observed viewpoints and usually producing more reliable renderings.
        #   If `camera_selection.json` exists, only the selected cameras are used as anchors;
        #   otherwise all available cameras are used.
        # If neither flag is enabled, the original training cameras are rendered directly.
        # If both flags are enabled, `generate_cameras` takes precedence due to the if/elif order.
        if args.generate_cameras:
            logger.info("Generating cameras for 360° orbit path...")
            cameras = generate_path(cameras, n_frames=args.num_frames)
            logger.info(f"Generated {len(cameras)} cameras for full orbit")
        elif args.generate_constrained_cameras:
            selection_file = os.path.join(dataset.source_path, "camera_selection.json")
            if os.path.exists(selection_file):
                logger.info(f"Found camera selection file: {selection_file}")
                with open(selection_file, 'r') as f:
                    selection_data = json.load(f)
                    indices = selection_data.get('selected_camera_indices', [])
                    if indices:
                        logger.info(f"Using {len(indices)} selected cameras from indices: {indices}")
                        selected_cameras = [cameras[i] for i in indices if i < len(cameras)]
                        cameras = generate_constrained_path(selected_cameras, n_frames=args.num_frames)
                        logger.info(f"Generated {len(cameras)} cameras for constrained path")
                    else:
                        logger.warning("No camera indices specified in selection file, using all cameras")
                        cameras = generate_constrained_path(cameras, n_frames=args.num_frames)
                        logger.info(f"Generated {len(cameras)} cameras for constrained path using all views")
            else:
                logger.info("No camera selection file found, using all available cameras")
                cameras = generate_constrained_path(cameras, n_frames=args.num_frames)
                logger.info(f"Generated {len(cameras)} cameras for constrained path")

        # Step 3: Render scene with shadows
        image_frames = render_scene_with_shadows(
            cameras, gaussians, object_gaussians_list, 
            envmap_list, lgtmap_list, pipe, args, object_sops_list=object_sops_list
        )

        # Step 3: Save output
        output_filedir = args.output_dir if len(args.output_dir) > 0 else os.path.join(
            args.workspace, "composition" + ("_trace" if args.with_trace else ""), composition_name
        )
        save_output(image_frames, output_filedir, args.fps)
        logger.info("Program execution completed")
