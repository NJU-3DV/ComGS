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

import imageio.v2 as imageio
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr, mae
from argparse import ArgumentParser
import numpy as np

MASK_THRE = 0.5

def readImages(renders_dir, gt_dir, mask_dir):
    renders = []
    gts = []
    masks = []
    image_names = []
    for fname in os.listdir(gt_dir):

        render_file = os.path.join(renders_dir, fname)
        gt_file = os.path.join(gt_dir, fname)
        
        render = imageio.imread(render_file)
        gt = imageio.imread(gt_file)
        
        render = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        gt = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
        
        if mask_dir is not None:
            mask_file = os.path.join(mask_dir, fname)
            mask = imageio.imread(mask_file)
            mask = tf.to_tensor(mask).unsqueeze(0).cuda()
            mask = (mask > MASK_THRE).float()

            render = render * mask + (1 - mask)
            gt = gt * mask + (1 - mask)
        
        renders.append(render)
        gts.append(gt)
        masks.append(mask)
        image_names.append(fname)

    return renders, gts, masks, image_names

def readNormalImage(normal_dir, pnormal_dir, gt_dir, mask_dir):
    normals = []
    pnormals = []
    gts = []
    masks = []
    image_names = []
    for fname in os.listdir(normal_dir):

        normal_file = os.path.join(normal_dir, fname)
        pnormal_file = os.path.join(pnormal_dir, fname)
        gt_path_str = os.path.join(gt_dir, fname) 

        normal = imageio.imread(normal_file)

        try:
            pnormal = imageio.imread(pnormal_file)
        except:
            pnormal = normal.copy()
        
        gt = imageio.imread(gt_path_str)
        
        normal = tf.to_tensor(normal).cuda()[:3, :, :]
        pnormal = tf.to_tensor(pnormal).cuda()[:3, :, :]
        gt = tf.to_tensor(gt).cuda()[:3, :, :]

        normal = normal * 2 - 1
        pnormal = pnormal * 2 - 1
        gt = gt * 2 - 1

        if mask_dir is not None:
            mask_file = os.path.join(mask_dir, fname)
            mask = imageio.imread(mask_file)
            mask = tf.to_tensor(mask).cuda()

            mask = (mask > MASK_THRE).float()
            invalid_val = torch.tensor([0, 0, 1], device=mask.device, dtype=torch.float32)
            normal = normal * mask + (1 - mask) * invalid_val[:, None, None]
            pnormal = pnormal * mask + (1 - mask) * invalid_val[:, None, None]
            gt = gt * mask + (1 - mask) * invalid_val[:, None, None]

        normals.append(normal)
        pnormals.append(pnormal)
        gts.append(gt)
        masks.append(mask)
        image_names.append(fname)
    
    return normals, pnormals, gts, masks, image_names

def readAlbedoImages(albedos_dir, gt_dir, mask_dir, scale=None):
    albedos = []
    gts = []
    masks = []
    image_names = []
    for fname in os.listdir(albedos_dir):

        albedo_file = os.path.join(albedos_dir, fname)
        gt_path_str = os.path.join(gt_dir, fname) 
        
        albedo = imageio.imread(albedo_file)
        gt = imageio.imread(gt_path_str)

        albedo = tf.to_tensor(albedo).unsqueeze(0)[:, :3, :, :].cuda()
        gt = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()

        if scale is not None:
            albedo = albedo * scale[None, :, None, None]
            albedo = torch.clamp(albedo, 0, 1)
        
        if mask_dir is not None:
            mask_file = os.path.join(mask_dir, fname)
            mask = imageio.imread(mask_file)
            mask = tf.to_tensor(mask).unsqueeze(0).cuda()
            mask = (mask > MASK_THRE).float()

            albedo = albedo * mask + (1 - mask)
            gt = gt * mask + (1 - mask)

        albedos.append(albedo)
        gts.append(gt)
        masks.append(mask)
        image_names.append(fname)

    return albedos, gts, masks, image_names

def evaluate(model_paths, eval_type, no_lpips):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}

    for scene_dir in model_paths:
        # try:
        if True:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = os.path.join(scene_dir, "test")
            for method in os.listdir(test_dir):
                if not os.path.isdir(os.path.join(test_dir, method)):
                    continue

                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                if "rf" in eval_type:
                    print("Compute metrics for randiance field...")

                    full_dict[scene_dir][method]["rf"] = {}
                    per_view_dict[scene_dir][method]["rf"] = {}
                    full_dict_polytopeonly[scene_dir][method]["rf"] = {}
                    per_view_dict_polytopeonly[scene_dir][method]["rf"] = {}

                    gt_dir = os.path.join(test_dir, method, "gt")
                    renders_dir = os.path.join(test_dir, method, "renders")
                    mask_dir = os.path.join(test_dir, method, "masks")
                    
                    renders, gts, masks, image_names = readImages(renders_dir, gt_dir, mask_dir)

                    ssims, psnrs, lpipss = [], [], []
                    progress_bar = tqdm(range(len(renders)), desc="RF Metric evaluation progress", dynamic_ncols=True)
                    for idx in progress_bar:
                        # Update progress bar description with current image name
                        progress_bar.set_description(f"RF Metric: {image_names[idx]}")

                        ssims.append(ssim(renders[idx], gts[idx]).double())
                        psnrs.append(psnr(renders[idx], gts[idx]).double())
                        if not no_lpips:
                            lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg').double())
                        else:
                            lpipss.append(torch.tensor(0.0).double())
                    
                    ssims = torch.tensor(ssims)
                    psnrs = torch.tensor(psnrs)
                    lpipss = torch.tensor(lpipss)

                    print("  SSIM : {:>12.7f}".format(ssims.mean(), ".5"))
                    print("  PSNR : {:>12.7f}".format(psnrs.mean(), ".5"))
                    print("  LPIPS: {:>12.7f}".format(lpipss.mean(), ".5"))
                    print("")

                    full_dict[scene_dir][method]["rf"].update({
                        "SSIM": ssims.mean().item(), 
                        "PSNR": psnrs.mean().item(),
                        "LPIPS": lpipss.mean().item()
                    })
                    per_view_dict[scene_dir][method]["rf"].update({
                        "SSIM": {name: ssim for ssim, name in zip(ssims.tolist(), image_names)},
                        "PSNR": {name: psnr for psnr, name in zip(psnrs.tolist(), image_names)},
                        "LPIPS": {name: lp for lp, name in zip(lpipss.tolist(), image_names)}
                    })

                    del renders, gts, masks
                    del ssims, psnrs, lpipss
                    torch.cuda.empty_cache()

                if "normal" in eval_type:
                    print("Compute metrics for normal...")
                    full_dict[scene_dir][method]["normal"] = {}
                    per_view_dict[scene_dir][method]["normal"] = {}
                    full_dict_polytopeonly[scene_dir][method]["normal"] = {}
                    per_view_dict_polytopeonly[scene_dir][method]["normal"] = {}

                    gt_dir = os.path.join(test_dir, method, "gt_normal")
                    normal_dir = os.path.join(test_dir, method, "normal")
                    pnormal_dir = os.path.join(test_dir, method, "pnormal")
                    mask_dir = os.path.join(test_dir, method, "masks")
                    
                    normals, pnormals, gts, masks, image_names = readNormalImage(normal_dir, pnormal_dir, gt_dir, mask_dir)

                    render_maes, pseudo_maes = [], []
                    progress_bar = tqdm(range(len(normals)), desc="Normal Metrics evaluation progress", dynamic_ncols=True)
                    for idx in progress_bar:
                        progress_bar.set_description(f"Normal Metric: {image_names[idx]}")
                        render_maes.append(mae(normals[idx], gts[idx]).double())
                        pseudo_maes.append(mae(pnormals[idx], gts[idx]).double())
                    
                    render_maes = torch.tensor(render_maes)
                    pseudo_maes = torch.tensor(pseudo_maes)

                    print("  MAE(render) : {:>12.7f}".format(render_maes.mean(), ".5"))
                    print("  MAE(pseudo) : {:>12.7f}".format(pseudo_maes.mean(), ".5"))
                    print("")

                    full_dict[scene_dir][method]["normal"].update({
                        "MAE(render)": render_maes.mean().item(), 
                        "MAE(pseudo)": pseudo_maes.mean().item(),
                    })
                    per_view_dict[scene_dir][method]["normal"].update({
                        "MAE(render)": {name: mae for mae, name in zip(render_maes.tolist(), image_names)},
                        "MAE(pseudo)": {name: mae for mae, name in zip(pseudo_maes.tolist(), image_names)}
                    })

                    del normals, pnormals, gts, masks
                    del render_maes, pseudo_maes
                    torch.cuda.empty_cache()
                    
                if "albedo" in eval_type:
                    print("Compute scale for albedo...")
                    # calculate albedo scale first...
                    gt_dir = os.path.join(test_dir, method, "gt_albedo")
                    albedo_dir = os.path.join(test_dir, method, "albedo")
                    mask_dir = os.path.join(test_dir, method, "masks")
                    
                    # NOTE that we have saved albedo map in rgb space...
                    albedos, gts, masks, image_names = readAlbedoImages(albedo_dir, gt_dir, mask_dir)

                    valid_gt = []
                    valid_pred = []
                    progress_bar = tqdm(range(len(albedos)), desc="Compute scales for albedo process", dynamic_ncols=True)
                    for idx in progress_bar:
                        progress_bar.set_description(f"Computing albedo scale: {image_names[idx]}")
                        gt_albedo = gts[idx]            # [1, 3, H, W]
                        pred_albedo = albedos[idx]      # [1, 3, H, W]
                        mask = masks[idx]               # [1, 1, H, W]

                        gt_albedo = gt_albedo.squeeze(0)     # [3, H, W]
                        pred_albedo = pred_albedo.squeeze(0) # [3, H, W]
                        mask = mask.squeeze(0).squeeze(0)    # [H, W]

                        valid_gt.append(gt_albedo[:, mask > 0.5])
                        valid_pred.append(pred_albedo[:, mask > 0.5])
                    
                    valid_gt = torch.concat(valid_gt, dim=-1)
                    valid_pred = torch.concat(valid_pred, dim=-1)

                    scale = (valid_gt / torch.clamp_min(valid_pred, 1e-6)).median(dim=-1).values

                    albedo_scale_path = os.path.join(scene_dir, "albedo_ratio.pth")
                    torch.save(scale, albedo_scale_path)

                    print("albedo scale: ", scale)

                    del albedos, gts, masks
                    del valid_gt, valid_pred
                    torch.cuda.empty_cache()

                    # Compute metrics for albedo
                    print("Compute metrics for albedo...")
                    full_dict[scene_dir][method]["albedo"] = {}
                    per_view_dict[scene_dir][method]["albedo"] = {}
                    full_dict_polytopeonly[scene_dir][method]["albedo"] = {}
                    per_view_dict_polytopeonly[scene_dir][method]["albedo"] = {}

                    gt_dir = os.path.join(test_dir, method, "gt_albedo")
                    albedo_dir = os.path.join(test_dir, method, "albedo")
                    mask_dir = os.path.join(test_dir, method, "masks")

                    # saving scaled albedo map...
                    albedo_scaled_dir = os.path.join(test_dir, method, "albedo_scaled")
                    os.makedirs(albedo_scaled_dir, exist_ok=True)

                    albedos, gts, masks, image_names = readAlbedoImages(albedo_dir, gt_dir, mask_dir, scale=scale)

                    ssims, psnrs, lpipss = [], [], []
                    progress_bar = tqdm(range(len(albedos)), desc="Albedo Metric evaluation progress", dynamic_ncols=True)
                    for idx in progress_bar:
                        progress_bar.set_description(f"Albedo Metric: {image_names[idx]}")
                        ssims.append(ssim(albedos[idx], gts[idx]).double())
                        psnrs.append(psnr(albedos[idx], gts[idx]).double())

                        if not no_lpips:
                            lpipss.append(lpips(albedos[idx], gts[idx], net_type='vgg').double())
                        else:
                            lpipss.append(torch.tensor(0.0).double())

                        saved_albedo = albedos[idx].squeeze(0).permute(1, 2, 0).cpu().numpy()
                        saved_albedo = (saved_albedo * 255).astype(np.uint8)
                        imageio.imwrite(os.path.join(albedo_scaled_dir, image_names[idx]), saved_albedo)
                    
                    ssims = torch.tensor(ssims)
                    psnrs = torch.tensor(psnrs)
                    lpipss = torch.tensor(lpipss)
                    
                    print("  SSIM : {:>12.7f}".format(ssims.mean(), ".5"))
                    print("  PSNR : {:>12.7f}".format(psnrs.mean(), ".5"))
                    print("  LPIPS: {:>12.7f}".format(lpipss.mean(), ".5"))
                    print("")

                    full_dict[scene_dir][method]["albedo"].update({
                        "SSIM": ssims.mean().item(), 
                        "PSNR": psnrs.mean().item(),
                        "LPIPS": lpipss.mean().item()
                    })
                    per_view_dict[scene_dir][method]["albedo"].update({
                        "SSIM": {name: ssim for ssim, name in zip(ssims.tolist(), image_names)},
                        "PSNR": {name: psnr for psnr, name in zip(psnrs.tolist(), image_names)},
                        "LPIPS": {name: lp for lp, name in zip(lpipss.tolist(), image_names)}
                    })

                    del albedos, gts, masks
                    del ssims, psnrs, lpipss
                    torch.cuda.empty_cache()
                    
                if "pbr" in eval_type:
                    print("Compute metrics for pbr...")

                    full_dict[scene_dir][method]["pbr"] = {}
                    per_view_dict[scene_dir][method]["pbr"] = {}
                    full_dict_polytopeonly[scene_dir][method]["pbr"] = {}
                    per_view_dict_polytopeonly[scene_dir][method]["pbr"] = {}

                    gt_dir = os.path.join(test_dir, method, "gt")
                    pbr_dir = os.path.join(test_dir, method, "pbr")
                    mask_dir = os.path.join(test_dir, method, "masks")

                    renders, gts, masks, image_names = readImages(pbr_dir, gt_dir, mask_dir)

                    ssims, psnrs, lpipss = [], [], []
                    progress_bar = tqdm(range(len(renders)), desc="PBR Metric evaluation progress", dynamic_ncols=True)
                    for idx in progress_bar:
                        progress_bar.set_description(f"PBR Metric: {image_names[idx]}")
                        ssims.append(ssim(renders[idx], gts[idx]).double())
                        psnrs.append(psnr(renders[idx], gts[idx]).double())

                        if not no_lpips:
                            lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg').double())
                        else:
                            lpipss.append(torch.tensor(0.0).double())
                    
                    ssims = torch.tensor(ssims)
                    psnrs = torch.tensor(psnrs)
                    lpipss = torch.tensor(lpipss)

                    print("  SSIM : {:>12.7f}".format(ssims.mean(), ".5"))
                    print("  PSNR : {:>12.7f}".format(psnrs.mean(), ".5"))
                    print("  LPIPS: {:>12.7f}".format(lpipss.mean(), ".5"))
                    print("")

                    full_dict[scene_dir][method]["pbr"].update({
                        "SSIM": ssims.mean().item(), 
                        "PSNR": psnrs.mean().item(),
                        "LPIPS": lpipss.mean().item()
                    })
                    per_view_dict[scene_dir][method]["pbr"].update({
                        "SSIM": {name: ssim for ssim, name in zip(ssims.tolist(), image_names)},
                        "PSNR": {name: psnr for psnr, name in zip(psnrs.tolist(), image_names)},
                        "LPIPS": {name: lp for lp, name in zip(lpipss.tolist(), image_names)}
                    })

                    del renders, gts, masks
                    del ssims, psnrs, lpipss
                    torch.cuda.empty_cache()
                    
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=4)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=4)
        # except:
        #     print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--eval_type', '-t', required=False, nargs="+", type=str, default=["normal", "albedo", "pbr"])
    parser.add_argument('--no_lpips', action='store_true', help="Disable LPIPS evaluation")
    args = parser.parse_args()

    with torch.no_grad():
        evaluate(args.model_paths, args.eval_type, args.no_lpips)