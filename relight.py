import os
# # # NOTE: set some environment variables to avoid increasing reserved memory
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync,max_split_size_mb:512,expandable_segments:True"


import json
import sys
from gaussian_renderer import render
from gaussian_renderer.pbr import pbr_shading
import numpy as np
import torch
from scene import GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.cameras import Camera
from scene.envmap import SphericalEnvMap
from utils.graphics_utils import focal2fov
from utils.pbr_utils import rgb_to_srgb
from utils.system_utils import searchForMaxIteration
from torchvision.utils import save_image
from tqdm import tqdm
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr
from utils.image_utils import read_exr
import imageio.v2 as imageio
from scene.sops import SurfaceOctahedralProbes
import warnings
warnings.filterwarnings("ignore")


def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict


def process_dataset(args, dataset, pipe, gaussians, sops, dataset_type):
    """Process the relighting task for different dataset types
    
    Args:
        args: Command line arguments
        dataset: Dataset parameters
        pipe: Pipeline parameters
        gaussians: Gaussian model
        sops: Surface Octahedral Probes
        dataset_type: Type of dataset ('syncom' or 'tensoir')
    """
    # Setup result directory and output structures
    results_dir = os.path.join(args.model_path, "test_rli" + (f"_{args.extra}" if len(args.extra)>0 else ""))
    sph_map = SphericalEnvMap()
    results_dict = {}
    
    # Set background function
    if dataset.background == "b":
        # Black background: return a tensor of zeros
        background_func = lambda x, y: torch.zeros(3, device="cuda")
    elif dataset.background == "w":
        # White background: return a tensor of ones
        background_func = lambda x, y: torch.ones(3, device="cuda")
    else:
        raise NotImplementedError(f"Background type '{dataset.background}' not implemented.")
    
    # Load dataset-specific information
    if dataset_type == 'syncom':
        # SynCom specific setup
        object_name = os.path.basename(args.model_path)
        relit_root = os.path.dirname(os.path.dirname(args.source_path))
        relit_object_path = os.path.join(relit_root, "object_relit", object_name)
        
        test_transforms_file = os.path.join(relit_object_path, "cameras.json")
        contents = load_json_config(test_transforms_file)
        frames = contents["cameras"].values()
        
        # Define environment maps
        envmap_dir = os.path.join(relit_root, "object_relit", "envmaps")
        env_files = ["billiard.exr", "fireplace.exr", "lakeside.exr", "snowy.exr"]
        use_exr = True
        
        # Print additional info for SynCom
        print("Load albedo scale:", torch.load(os.path.join(args.model_path, "albedo_ratio.pth")))
        print(f"Number of rays: {dataset.num_rays}")
        
        # Function to get the correct frame data and image path
        def get_frame_data(idx, frame, envname):
            image_name = frame["name"]
            cam_width = frame["width"]
            cam_height = frame["height"]
            
            intr = np.array(frame["intr"], dtype=np.float32)
            FovX = focal2fov(intr[0], cam_width)
            FovY = focal2fov(intr[1], cam_height)

            c2w = np.array(frame["extr"], dtype=np.float32).reshape(4, 4)
            c2w[:3, 1:3] *= -1  # blender to opencv
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3]).astype(np.float32)
            T = w2c[:3, 3].astype(np.float32)
            
            image_path = os.path.join(relit_object_path, envname, f"{image_name}.png")
            
            return {
                'fovx': FovX,
                'fovy': FovY,
                'R': R,
                'T': T,
                'image_path': image_path
            }
        
    else:  # tensoir
        # TensOIR specific setup
        test_transforms_file = os.path.join(args.source_path, "transforms_test.json")
        contents = load_json_config(test_transforms_file)
        
        fovx = contents["camera_angle_x"]
        frames = contents["frames"]
        
        # Define environment maps
        envmap_dir = os.path.join(os.path.dirname(args.source_path), "Environment_Maps", "high_res_envmaps_2k")
        env_files = ["bridge.hdr", "city.hdr", "fireplace.hdr", "forest.hdr", "night.hdr"]
        use_exr = False
        
        # Function to get the correct frame data and image path
        def get_frame_data(idx, frame, envname):
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # Change from OpenGL/Blender camera axes to COLMAP
            c2w[:3, 1:3] *= -1
            
            # Get world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            
            image_path = os.path.join(args.source_path, 
                                    f"test_{idx:03}/" + frame["file_path"].split("/")[-1] + "_" + envname + ".png")
            
            return {
                'fovx': fovx,
                'fovy': fovx,
                'R': R,
                'T': T,
                'image_path': image_path
            }

    # Create the task dictionary
    task_dict = {
        f"envmap_{env_file[:-4]}": {
            "envmap_path": os.path.join(envmap_dir, env_file),
        }
        for env_file in env_files
    }
    
    # Process each environment map
    for task_name in task_dict:
        
        results_dict[task_name] = {}
        task_dir = os.path.join(results_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        # Load environment map data according to dataset type
        if use_exr:
            envmap_data = read_exr(task_dict[task_name]["envmap_path"])[..., :3]
            envmap_data = torch.from_numpy(envmap_data).permute(2, 0, 1).float().cuda()
        else:
            envmap_data = imageio.imread(task_dict[task_name]["envmap_path"])
            envmap_data = torch.from_numpy(envmap_data).permute(2, 0, 1).float().cuda()
            
        sph_map.set_data(envmap_data)
        
        # Setup metrics tracking
        psnr_pbr = 0.0
        ssim_pbr = 0.0
        lpips_pbr = 0.0
        
        # Create output directories
        if not args.no_save:
            os.makedirs(os.path.join(task_dir, "render"), exist_ok=True)
            os.makedirs(os.path.join(task_dir, "render_env"), exist_ok=True)
            os.makedirs(os.path.join(task_dir, "gt"), exist_ok=True)
            os.makedirs(os.path.join(task_dir, "gt_env"), exist_ok=True)
        
        # Get environment map name for image paths
        envname = os.path.splitext(os.path.basename(task_dict[task_name]["envmap_path"]))[0]
        
        # Process each frame
        for idx, frame in enumerate(tqdm(frames, leave=False, dynamic_ncols=True, desc=f"Relighting with {envname}")):
            # Get frame-specific data
            frame_data = get_frame_data(idx, frame, envname)
            
            # Load and process the image
            image_rgba = imageio.imread(frame_data['image_path'])
            image_rgba = image_rgba.astype(np.float32) / 255.0
            image = image_rgba[..., :3]
            mask = image_rgba[..., 3:]
            
            gt_image = torch.from_numpy(image).permute(2, 0, 1).float().cuda()
            mask = torch.from_numpy(mask).permute(2, 0, 1).float().cuda()
            gt_image = gt_image * mask
            
            H = image.shape[0]
            W = image.shape[1]
            
            # Create camera
            custom_cam = Camera(
                colmap_id=0, 
                R=frame_data['R'], 
                T=frame_data['T'],
                FoVx=frame_data['fovx'], 
                FoVy=frame_data['fovy'],
                image=torch.zeros(3, H, W), 
                gt_alpha_mask=None, 
                image_name=None, 
                uid=0
            )

            # Render and process
            results = render_and_process(custom_cam, gaussians, pipe, background_func, sph_map, 
                                       sops, dataset.num_rays, mask, gt_image)
            
            image = results["image"]
            image_env = results["image_env"]
            gt_image = results["gt_image"]
            gt_image_env = results["gt_image_env"]

            # Save images
            if not args.no_save:
                save_image(image, os.path.join(task_dir, "render", f"{idx}.png"))
                save_image(image_env, os.path.join(task_dir, "render_env", f"{idx}.png"))
                save_image(gt_image, os.path.join(task_dir, "gt", f"{idx}.png"))
                save_image(gt_image_env, os.path.join(task_dir, "gt_env", f"{idx}.png"))
            
            # Calculate metrics
            with torch.no_grad():
                psnr_pbr += psnr(image, gt_image).mean().double().item()
                ssim_pbr += ssim(image, gt_image).mean().double().item()
                if not args.no_lpips:
                    lpips_pbr += lpips(image, gt_image, net_type='vgg').mean().double().item()
                else:
                    lpips_pbr += 0.0
        
        # Average metrics
        psnr_pbr /= len(frames)
        ssim_pbr /= len(frames)
        lpips_pbr /= len(frames)
        
        # Store results
        results_dict[task_name]["psnr_pbr"] = psnr_pbr
        results_dict[task_name]["ssim_pbr"] = ssim_pbr
        results_dict[task_name]["lpips_pbr"] = lpips_pbr

        print("\nEvaluating {}: PSNR_PBR {: .4f} SSIM_PBR {: .4f} LPIPS_PBR {: .4f}".format(
            task_name, psnr_pbr, ssim_pbr, lpips_pbr))

    # Calculate average metrics
    task_names = list(task_dict.keys())
    results_dict["psnr_pbr_avg"] = np.mean([results_dict[task_name]["psnr_pbr"] for task_name in task_names])
    results_dict["ssim_pbr_avg"] = np.mean([results_dict[task_name]["ssim_pbr"] for task_name in task_names])
    results_dict["lpips_pbr_avg"] = np.mean([results_dict[task_name]["lpips_pbr"] for task_name in task_names])
    print("\nEvaluating AVG: PSNR_PBR {: .4f} SSIM_PBR {: .4f} LPIPS_PBR {: .4f}".format(
        results_dict["psnr_pbr_avg"], results_dict["ssim_pbr_avg"], results_dict["lpips_pbr_avg"]))
    
    # Save results
    with open(os.path.join(results_dir, "relighting_results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)
    print("Results saved to", os.path.join(results_dir, "relighting_results.json"))
    
    return results_dict


def render_and_process(custom_cam, gaussians, pipe, background_func, sph_map, sops, num_rays, mask, gt_image):
    """Unified rendering and post-processing pipeline
    
    Args:
        custom_cam: Camera object with intrinsic/extrinsic parameters
        gaussians: Gaussian model
        pipe: Pipeline parameters
        background_func: Function that returns background color
        sph_map: Spherical environment map
        sops: Surface Octahedral Probes
        num_rays: Number of rays for rendering
        mask: Alpha mask for blending
        gt_image: Ground truth image
        
    Returns:
        Dictionary with rendered images and processed ground truth
    """
    render_pkg = render(
        custom_cam, gaussians, pipe, background_func, sph_map, 
        requires_geometry=True, 
        requires_material=True,
        flag_max_count=False
    )

    background = render_pkg["background"]
    pbr_pkg = pbr_shading(
        render_pkg, custom_cam, sops, sph_map, background, 
        num_rays=num_rays, return_details=False, 
        ignore_local=True, is_training=False, 
        important_sampling=True
    )
    pbr_image = pbr_pkg["pbr"]
    
    # Blend foreground with background
    image = pbr_image * mask + background[:, None, None] * (1 - mask)
    gt_image_processed = gt_image + background[:, None, None] * (1 - mask)

    # Sample environment map for background
    vdir = custom_cam.get_cam_rays(world_frame=True).permute(2, 0, 1)[:3]
    bg_env = sph_map.sample(vdir)

    # Create environment-aware images
    image_env = pbr_image * mask + bg_env * (1 - mask)
    gt_image_env = gt_image_processed + rgb_to_srgb(bg_env) * (1 - mask)

    # Convert to sRGB
    image = rgb_to_srgb(image)
    image_env = rgb_to_srgb(image_env)
    
    return {
        "image": image,
        "image_env": image_env,
        "gt_image": gt_image_processed,
        "gt_image_env": gt_image_env
    }


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Unified Relighting for SynCom-Object and TensoIR datasets")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--no_save", default=False, action='store_true')
    parser.add_argument("--no_lpips", default=False, action='store_true')
    parser.add_argument("-e", "--extra", default='', type=str)
    parser.add_argument("--dataset", type=str, required=True, choices=['syncom', 'tensoir'],
                        help="Type of dataset to process: 'syncom' or 'tensoir'")
    
    # Parse arguments
    args = get_combined_args(parser)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    with torch.no_grad():
        # load gaussians
        gaussians = GaussianModel(3)
        
        if args.iteration < 0:
            loaded_iter = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
        else:
            loaded_iter = args.iteration
        gaussians.load_ply(os.path.join(args.model_path, "point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply")) 
        
        # Load albedo scale
        albedo_scale = torch.load(os.path.join(args.model_path, "albedo_ratio.pth"))
        gaussians.update_albedo(albedo_scale)

        # Setup SOPs
        sops = SurfaceOctahedralProbes()
        sops_file = f"{dataset.model_path}/chkpnt{loaded_iter}_sops.pth"
        try:
            (model_params, first_iter) = torch.load(sops_file)
            sops.restore(model_params)
            print(f"[INFO] restore sops from {sops_file}, first iter:{first_iter}")
        except:
            sops = None
            print(f"[WARN] Failed to load sops from checkpoint {sops_file}...")
        
        # Process dataset based on selected type
        process_dataset(args, dataset, pipe, gaussians, sops, args.dataset)
