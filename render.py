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
from scene import Scene
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.system_utils import searchForMaxIteration
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.gs_utils import GaussianExtractor
from utils.render_utils import generate_path, create_videos
from scene.envmap import OctahedralEnvMap
from scene.sops import SurfaceOctahedralProbes
from gaussian_renderer.pbr import pbr_shading


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--render_type", nargs='+', default=['geo', 'mat', 'lgt', 'pbr'], type=str, help='Render type: rf, geo, mat, lgt, pbr')
    parser.add_argument("--load_gt", action="store_true", help="Load ground truth albedo and normal for evaluation")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    # NOTE: prior is not need when render...

    with torch.no_grad():

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, load_gt=args.load_gt)
        bg_func = lambda x, y: torch.zeros(3, device="cuda")

        envmap = OctahedralEnvMap( 
            resolution = dataset.envmap_resolution,
            init_value = dataset.envmap_init_value
        )
        sops = SurfaceOctahedralProbes(
            num_probes=dataset.sops_num,
            resolution=dataset.sops_resolution
        )

        # load from checkpoint
        loaded_iter = args.iteration
        if loaded_iter == -1:
            loaded_iter = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
        
        envmap_exr_file = f"{dataset.model_path}/envmap/iteration_{loaded_iter}/envmap.exr"
        try:
            envmap.load_from_file(envmap_exr_file)
            print(f"[INFO] load envmap from {envmap_exr_file}")
        except:
            print(f"[WARN] Failed to load envmap from checkpoint {envmap_exr_file}")
        
        sops_file = f"{dataset.model_path}/chkpnt{loaded_iter}_sops.pth"
        try:
            (model_params, first_iter) = torch.load(sops_file)
            sops.restore(model_params)
            print(f"[INFO] restore sops from {sops_file}, first iter:{first_iter}")
        except:
            print(f"[WARN] Failed to load sops from checkpoint {sops_file}...")

        train_dir = os.path.join(args.model_path, 'train', "ours_{}_with_{}_rays".format(scene.loaded_iter, args.num_rays))
        test_dir = os.path.join(args.model_path, 'test', "ours_{}_with_{}_rays".format(scene.loaded_iter, args.num_rays))

        gaussExtractor = GaussianExtractor(
            gaussians, render, pbr_shading, pipe, bg_func=bg_func, envmap=envmap, 
            sops=sops, num_rays=dataset.num_rays, render_type=args.render_type
        )    
        
        if not args.skip_train:
            print("export training images ...")
            os.makedirs(train_dir, exist_ok=True)
            gaussExtractor.reconstruction(scene.getTrainCameras())
            gaussExtractor.export_image(train_dir)
            
        
        if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
            print("export rendered testing images ...")
            os.makedirs(test_dir, exist_ok=True)
            gaussExtractor.reconstruction(scene.getTestCameras())
            gaussExtractor.export_image(test_dir)
        
        
        if args.render_path:
            print("render videos ...")
            traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
            os.makedirs(traj_dir, exist_ok=True)
            n_fames = 240
            cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
            gaussExtractor.reconstruction(cam_traj)
            gaussExtractor.export_image(traj_dir)
            create_videos(
                base_dir=traj_dir,
                input_dir=traj_dir, 
                out_name='render_traj', 
                num_frames=n_fames
            )
