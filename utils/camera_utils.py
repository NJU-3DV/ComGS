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

import cv2
import torch
from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w = cam_info.width
    orig_h = cam_info.height

    if args.resolution in [1, 2, 3, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # optional buffer [1, H, W]
    mask = cam_info.mask
    if cam_info.mask is not None:
        mask = cv2.resize(cam_info.mask, resolution, interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask).unsqueeze(0)
    
    # required buffer # [3, H, W]
    image = cam_info.image
    image = cv2.resize(image, resolution, interpolation=cv2.INTER_AREA)
    gt_image = torch.from_numpy(image).permute(2, 0, 1)
    
    # optional buffer [H, W]
    depth = cam_info.depth
    if depth is not None:
        if cam_info.dense_depth:
            depth = cv2.resize(depth, resolution, interpolation=cv2.INTER_AREA)
        else:
            depth = cv2.resize(depth, resolution, interpolation=cv2.INTER_NEAREST)
    
        depth = torch.from_numpy(depth).unsqueeze(0)
        if mask is not None:
            depth = depth * mask
    
    # optional buffer [3, H, W]
    normal = cam_info.normal
    if normal is not None:
        if cam_info.dense_normal:
            normal = cv2.resize(normal, resolution, interpolation=cv2.INTER_AREA)
        else: 
            normal = cv2.resize(normal, resolution, interpolation=cv2.INTER_NEAREST)
        
        normal = torch.from_numpy(normal).permute(2, 0, 1)
        if mask is not None:
            normal = normal * mask

    # optional buffer [3, H, W]
    albedo = cam_info.albedo
    if albedo is not None:
        if cam_info.dense_albedo:
            albedo = cv2.resize(albedo, resolution, interpolation=cv2.INTER_AREA)
        else:
            albedo = cv2.resize(albedo, resolution, interpolation=cv2.INTER_NEAREST)
        
        albedo = torch.from_numpy(albedo).permute(2, 0, 1)
        if mask is not None:
            albedo = albedo * mask
    
    # optional buffer [H, W]
    metallic = cam_info.metallic
    if metallic is not None:
        if cam_info.dense_metallic:
            metallic = cv2.resize(metallic, resolution, interpolation=cv2.INTER_AREA)
        else:
            metallic = cv2.resize(metallic, resolution, interpolation=cv2.INTER_NEAREST)
        
        metallic = torch.from_numpy(metallic).unsqueeze(0)
        if mask is not None:
            metallic = metallic * mask
    
    # optional buffer [1, H, W]
    roughness = cam_info.roughness
    if roughness is not None:
        if cam_info.dense_roughness:
            roughness = cv2.resize(cam_info.roughness, resolution, interpolation=cv2.INTER_AREA)
        else:
            roughness = cv2.resize(cam_info.roughness, resolution, interpolation=cv2.INTER_NEAREST)
        
        roughness = torch.from_numpy(roughness).unsqueeze(0)
        if mask is not None:
            roughness = roughness * mask
    
    # optional buffer [3, H, W]
    gt_normal = cam_info.gt_normal
    if gt_normal is not None:
        gt_normal = cv2.resize(gt_normal, resolution, interpolation=cv2.INTER_AREA)
        gt_normal = torch.from_numpy(gt_normal).permute(2, 0, 1)
    
    # optional buffer [3, H, W]
    gt_albedo = cam_info.gt_albedo
    if gt_albedo is not None:
        gt_albedo = cv2.resize(gt_albedo, resolution, interpolation=cv2.INTER_AREA)
        gt_albedo = torch.from_numpy(gt_albedo).permute(2, 0, 1)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=mask, depth=depth, normal=normal, 
                  albedo=albedo, metallic=metallic, roughness=roughness,
                  gt_normal=gt_normal, gt_albedo=gt_albedo,
                  is_linear_rgb=cam_info.is_linear_rgb,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry