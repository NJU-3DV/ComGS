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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.image_utils import read_exr
from scene.gaussian_model import BasicPointCloud
from utils.pbr_utils import srgb_to_rgb

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array = None
    depth: np.array = None
    normal: np.array = None
    albedo: np.array = None
    metallic: np.array = None
    roughness: np.array = None
    dense_depth: bool = True
    dense_normal: bool = True
    dense_albedo: bool = True
    dense_metallic: bool = True
    dense_roughness: bool = True
    gt_normal: np.array = None
    gt_albedo: np.array = None
    is_linear_rgb: bool = False

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, load_prior=""):

    cam_infos = []
    
    # prior
    dataset_root = os.path.dirname(images_folder)
    priors_dict = dict()

    if "depth" in load_prior:
        depth_prior_file = os.path.join(dataset_root, "prior/depth_anything_v2.npy")
        print("[Prior] Loading depth prior from {}".format(depth_prior_file))
        priors_dict["depth"] = np.load(depth_prior_file, allow_pickle=True).item()
    if "normal" in load_prior:
        normal_prior_file = os.path.join(dataset_root, "prior/normal_stable.npy")
        print("[Prior] Loading normal prior from {}".format(normal_prior_file))
        priors_dict["normal"] = np.load(normal_prior_file, allow_pickle=True).item()
    if "albedo" in load_prior:
        albedo_prior_file = os.path.join(dataset_root, "prior/albedo_IDArb.npy")
        print("[Prior] Loading albedo prior from {}".format(albedo_prior_file))
        priors_dict["albedo"] = np.load(albedo_prior_file, allow_pickle=True).item()
    if "metallic" in load_prior:
        metalic_prior_file = os.path.join(dataset_root, "prior/metallic_IDArb.npy")
        print("[Prior] Loading metallic prior from {}".format(metalic_prior_file))
        priors_dict["metallic"] = np.load(metalic_prior_file, allow_pickle=True).item()
    if "roughness" in load_prior:
        roughness_prior_file = os.path.join(dataset_root, "prior/roughness_IDArb.npy")
        print("[Prior] Loading roughness prior from {}".format(roughness_prior_file))
        priors_dict["roughness"] = np.load(roughness_prior_file, allow_pickle=True).item()

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        org_height = intr.height
        org_width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec)).astype(np.float32)
        T = np.array(extr.tvec).astype(np.float32)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, org_height)
            FovX = focal2fov(focal_length_x, org_width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, org_height)
            FovX = focal2fov(focal_length_x, org_width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        image = np.array(image, dtype=np.float32)[..., :3] / 255.0
        height = image.shape[0]
        width = image.shape[1]

        mask = np.ones_like(image[:, :, 0], dtype=np.float32)
        if "mask" in load_prior and os.path.exists(os.path.join(dataset_root, "masks")):
            mask_path = os.path.join(dataset_root, "masks", image_name + ".png")
            mask = Image.open(mask_path)
            mask = np.array(mask).astype(np.float32) / 255.0 > 0
            mask = mask.astype(np.float32)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            
            image = image * mask[..., None]
        
        depth = None
        if "depth" in load_prior:
            depth = priors_dict["depth"][image_name]

        normal = None
        if "normal" in load_prior:
            normal = priors_dict["normal"][image_name]
            normal_width = normal.shape[1]
            normal_height = normal.shape[0]
            # to world space
            normal = np.matmul(R, normal.reshape(-1, 3).T)
            normal = normal.T.reshape(normal_height, normal_width, 3)

        albedo = None
        if "albedo" in load_prior:
            albedo = priors_dict["albedo"][image_name]

        metallic = None
        if "metallic" in load_prior:
            metallic = priors_dict["metallic"][image_name]

        roughness = None
        if "roughness" in load_prior:
            roughness = priors_dict["roughness"][image_name]

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, 
                              width=width, height=height, mask=mask, 
                              depth=depth, normal=normal, albedo=albedo,
                              metallic=metallic, roughness=roughness)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    if "nx" not in vertices or "ny" not in vertices or "nz" not in vertices:
        colors = np.random.rand(positions.shape[0], 3)
    else:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0

    if "nx" not in vertices or "ny" not in vertices or "nz" not in vertices:
        # If normals are not present, we create random normals
        normals = np.random.rand(positions.shape[0], 3)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    else:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb, normals=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    if normals is None:
        # just like *torch.rand* and normalized...
        normals = np.random.rand(xyz.shape[0], 3)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, load_prior="", llffhold=8, composition=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, 
        cam_intrinsics=cam_intrinsics, 
        images_folder=os.path.join(path, reading_dir),
        load_prior=load_prior
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval and not composition:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = cam_infos # all train images as test images

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")

    try:
        xyz, rgb, _ = read_points3D_binary(bin_path)
    except:
        xyz, rgb, _ = read_points3D_text(txt_path)
    
    storePly(ply_path, xyz, rgb)
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, extension=".png", load_prior=""):
    cam_infos = []

    priors_dict = dict()
    if "depth" in load_prior:
        depth_prior_file = os.path.join(path, "prior/depth_anything_v2.npy")
        print("[Prior] Loading depth prior from {}".format(depth_prior_file))
        priors_dict["depth"] = np.load(depth_prior_file, allow_pickle=True).item()
    if "normal" in load_prior:
        normal_prior_file = os.path.join(path, "prior/normal_stable.npy")
        print("[Prior] Loading normal prior from {}".format(normal_prior_file))
        priors_dict["normal"] = np.load(normal_prior_file, allow_pickle=True).item()
    
    if "albedo" in load_prior:
        albedo_prior_file = os.path.join(path, "prior/albedo_IDArb.npy")
        print("[Prior] Loading albedo prior from {}".format(albedo_prior_file))
        priors_dict["albedo"] = np.load(albedo_prior_file, allow_pickle=True).item()
    if "metallic" in load_prior:
        metalic_prior_file = os.path.join(path, "prior/metallic_IDArb.npy")
        print("[Prior] Loading metallic prior from {}".format(metalic_prior_file))
        priors_dict["metallic"] = np.load(metalic_prior_file, allow_pickle=True).item()
    if "roughness" in load_prior:
        roughness_prior_file = os.path.join(path, "prior/roughness_IDArb.npy")
        print("[Prior] Loading roughness prior from {}".format(roughness_prior_file))
        priors_dict["roughness"] = np.load(roughness_prior_file, allow_pickle=True).item()
    
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3]).astype(np.float32)  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3].astype(np.float32)

            image_path = os.path.join(path, cam_name)
            image_name = os.path.basename(image_path).split(".")[0]
            image = Image.open(image_path)
            
            im_data = np.array(image.convert("RGBA")).astype(np.float32)
            norm_data = im_data / 255
            mask = norm_data[:, :, 3]
            image = norm_data[:,:,:3] 

            width = image.shape[1]
            height = image.shape[0]

            fovy = focal2fov(fov2focal(fovx, width), height)
            FovY = fovy 
            FovX = fovx

            # load prior
            depth = None
            if priors_dict.get("depth") is not None:
                depth = priors_dict["depth"][image_name]
            
            normal = None
            if priors_dict.get("normal") is not None:
                normal = priors_dict["normal"][image_name]
                height_normal, width_normal = normal.shape[0], normal.shape[1]
                # to world space
                normal = np.matmul(R, normal.reshape(-1, 3).T)
                normal = normal.T.reshape(height_normal, width_normal, 3)
            
            albedo = None
            if priors_dict.get("albedo") is not None:
                albedo = priors_dict["albedo"][image_name]
            
            metallic = None
            if priors_dict.get("metallic") is not None:
                metallic = priors_dict["metallic"][image_name]
            
            roughness = None
            if priors_dict.get("roughness") is not None:
                roughness = priors_dict["roughness"][image_name]
            
                
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=width, 
                            height=height, mask=mask, depth=depth, normal=normal,
                            albedo=albedo,metallic=metallic,roughness=roughness))
            
    return cam_infos

def readNerfSyntheticInfo(path, eval, extension=".png", load_prior=""):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", extension, load_prior)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTensoIRTransforms(path, transformsfile, extension=".png", load_prior="", load_gt=False):
    cam_infos = []

    priors_dict = dict()
    if "depth" in load_prior:
        depth_prior_file = os.path.join(path, "prior/depth_anything_v2.npy")
        print("[Prior] Loading depth prior from {}".format(depth_prior_file))
        priors_dict["depth"] = np.load(depth_prior_file, allow_pickle=True).item()
    if "normal" in load_prior:
        normal_prior_file = os.path.join(path, "prior/normal_stable.npy")
        print("[Prior] Loading normal prior from {}".format(normal_prior_file))
        priors_dict["normal"] = np.load(normal_prior_file, allow_pickle=True).item()
    
    if "albedo" in load_prior:
        albedo_prior_file = os.path.join(path, "prior/albedo_IDArb.npy")
        print("[Prior] Loading albedo prior from {}".format(albedo_prior_file))
        priors_dict["albedo"] = np.load(albedo_prior_file, allow_pickle=True).item()
    if "metallic" in load_prior:
        metalic_prior_file = os.path.join(path, "prior/metallic_IDArb.npy")
        print("[Prior] Loading metallic prior from {}".format(metalic_prior_file))
        priors_dict["metallic"] = np.load(metalic_prior_file, allow_pickle=True).item()
    if "roughness" in load_prior:
        roughness_prior_file = os.path.join(path, "prior/roughness_IDArb.npy")
        print("[Prior] Loading roughness prior from {}".format(roughness_prior_file))
        priors_dict["roughness"] = np.load(roughness_prior_file, allow_pickle=True).item()
    
    prefix = "train" if "train" in transformsfile else "test"
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3]).astype(np.float32)  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3].astype(np.float32)

            image_path = os.path.join(path, cam_name)
            image_name = os.path.basename(os.path.dirname(image_path)).split("_")[1] # use the folder name for TensoIR
            image = Image.open(image_path)
            
            im_data = np.array(image.convert("RGBA")).astype(np.float32)
            norm_data = im_data / 255
            mask = norm_data[:, :, 3]
            image = norm_data[:,:,:3] 

            width = image.shape[1]
            height = image.shape[0]

            fovy = focal2fov(fov2focal(fovx, width), height)
            FovY = fovy 
            FovX = fovx

            # load prior
            depth = None
            if priors_dict.get("depth") is not None:
                depth = priors_dict["depth"][image_name]
            
            normal = None
            if priors_dict.get("normal") is not None:
                normal = priors_dict["normal"][image_name]
                height_normal, width_normal = normal.shape[0], normal.shape[1]
                # to world space
                normal = np.matmul(R, normal.reshape(-1, 3).T)
                normal = normal.T.reshape(height_normal, width_normal, 3)
            
            albedo = None
            if priors_dict.get("albedo") is not None:
                albedo = priors_dict["albedo"][image_name]
            
            metallic = None
            if priors_dict.get("metallic") is not None:
                metallic = priors_dict["metallic"][image_name]
            
            roughness = None
            if priors_dict.get("roughness") is not None:
                roughness = priors_dict["roughness"][image_name]
            
            # load from reorg_tensoIR dataste
            gt_normal = None
            gt_albedo = None
            
            if load_gt:
                gt_normal_file = image_path.replace("rgba", "normal")
                gt_normal = Image.open(gt_normal_file)
                gt_normal = np.array(gt_normal.convert("RGB")).astype(np.float32)
                gt_normal = gt_normal / 255
                gt_normal = 2 * gt_normal - 1

                gt_albedo_file = image_path.replace("rgba", "albedo")
                gt_albedo = Image.open(gt_albedo_file)
                gt_albedo = np.array(gt_albedo.convert("RGB")).astype(np.float32)
                gt_albedo = gt_albedo / 255
                
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=width, 
                            height=height, mask=mask, depth=depth, normal=normal,
                            albedo=albedo,metallic=metallic,roughness=roughness,
                            gt_albedo=gt_albedo, gt_normal=gt_normal))
            
    return cam_infos


def readTensoIRSyntheticInfo(path, eval, extension=".png", load_prior="", load_gt=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTensoIRTransforms(path, "transforms_train.json", extension, load_prior, load_gt)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTensoIRTransforms(path, "transforms_test.json", extension, load_gt=load_gt)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromSynComTransforms(path, images, load_prior="", load_gt=False):

    cam_infos = []

    priors_dict = dict()
    if "depth" in load_prior:
        depth_prior_file = os.path.join(path, "prior/depth_anything_v2.npy")
        print("[Prior] Loading depth prior from {}".format(depth_prior_file))
        priors_dict["depth"] = np.load(depth_prior_file, allow_pickle=True).item()
    if "normal" in load_prior:
        normal_prior_file = os.path.join(path, "prior/normal_stable.npy")
        print("[Prior] Loading normal prior from {}".format(normal_prior_file))
        priors_dict["normal"] = np.load(normal_prior_file, allow_pickle=True).item()
    if "albedo" in load_prior:
        albedo_prior_file = os.path.join(path, "prior/albedo_IDArb.npy")
        print("[Prior] Loading albedo prior from {}".format(albedo_prior_file))
        priors_dict["albedo"] = np.load(albedo_prior_file, allow_pickle=True).item()
    if "metallic" in load_prior:
        metallic_prior_file = os.path.join(path, "prior/metallic_IDArb.npy")
        print("[Prior] Loading metallic prior from {}".format(metallic_prior_file))
        priors_dict["metallic"] = np.load(metallic_prior_file, allow_pickle=True).item()
    if "roughness" in load_prior:
        roughness_prior_file = os.path.join(path, "prior/roughness_IDArb.npy")
        print("[Prior] Loading roughness prior from {}".format(roughness_prior_file))
        priors_dict["roughness"] = np.load(roughness_prior_file, allow_pickle=True).item()

    reading_dir = "images" if images == None else images
    with open(os.path.join(path, "cameras.json")) as json_file:
        contents = json.load(json_file)

        frames = contents["cameras"]
        for idx, frame in enumerate(frames.values()):
            image_name = frame["name"]
            cam_width = frame["width"]
            cam_height = frame["height"]

            intr = np.array(frame["intr"], dtype=np.float32)
            FovX = focal2fov(intr[0], cam_width)
            FovY = focal2fov(intr[1], cam_height)

            c2w = np.array(frame["extr"], dtype=np.float32).reshape(4, 4)
            c2w[:3, 1:3] *= -1  # blender to opencv
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3]).astype(np.float32)  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3].astype(np.float32)

            # NOTE: Lazy solution for demo (no images are provided)
            try:
                if "object" in path:
                    # NOTE: load HDR [Object]
                    image_path = os.path.join(path, reading_dir, f"{image_name}.exr")
                    image_mask = read_exr(image_path)
                    image_mask = image_mask.astype(np.float32)
                    is_linear_rgb = True
                else:
                    # NOTE: load LDR [Scene]
                    image_path = os.path.join(path, reading_dir, f"{image_name}.png")
                    image_mask = Image.open(image_path)
                    image_mask = np.array(image_mask.convert("RGBA")).astype(np.float32) / 255.0
                    is_linear_rgb = False
            except:
                # print(f"Image {image_name} not found in {path}/{reading_dir}. Loading All black image.")
                image_mask = np.empty((cam_height, cam_width, 4), dtype=np.float32)
                is_linear_rgb = False

            image = image_mask[..., :3]
            mask = image_mask[..., 3]

            height, width = image.shape[0], image.shape[1]

            depth = None
            if priors_dict.get("depth") is not None:
                depth = priors_dict["depth"][image_name].astype(np.float32)
            
            normal = None
            if priors_dict.get("normal") is not None:
                normal = priors_dict["normal"][image_name].astype(np.float32)
                height_normal, width_normal = normal.shape[0], normal.shape[1]
                # to world space
                normal = np.matmul(R, normal.reshape(-1, 3).T)
                normal = normal.T.reshape(height_normal, width_normal, 3)
            
            albedo = None
            if priors_dict.get("albedo") is not None:
                albedo = priors_dict["albedo"][image_name].astype(np.float32)
            
            metallic = None
            if priors_dict.get("metallic") is not None:
                metallic = priors_dict["metallic"][image_name].astype(np.float32)
            
            roughness = None
            if priors_dict.get("roughness") is not None:
                roughness = priors_dict["roughness"][image_name].astype(np.float32)
            
            gt_albedo = None
            gt_normal = None
            if load_gt:
                gt_albedo_file = os.path.join(path, "albedo", f"{image_name}.exr")
                gt_albedo = read_exr(gt_albedo_file).astype(np.float32)[..., :3]

                gt_normal_file = os.path.join(path, "normal", f"{image_name}.exr")
                gt_normal = read_exr(gt_normal_file).astype(np.float32)[..., :3]

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, 
                                        image_name=image_name, width=width, height=height, mask=mask, depth=depth, 
                                        normal=normal, albedo=albedo, metallic=metallic, roughness=roughness, 
                                        is_linear_rgb=is_linear_rgb, gt_albedo=gt_albedo, gt_normal=gt_normal))
            
    return cam_infos


def readSynComSceneInfo(path, images, eval, load_prior="", load_gt=False, composition=False):
    if composition:
        print("Reading Composition Cameras")
        train_cam_infos = readCamerasFromSynComTransforms(path, images)
        test_cam_infos = []
    else:
        print("Reading Training Cameras")
        train_cam_infos = readCamerasFromSynComTransforms(os.path.join(path, "train"), images, load_prior=load_prior, load_gt=load_gt)
        test_cam_infos = []
        if eval:
            print("Reading Test Cameras")
            test_cam_infos = readCamerasFromSynComTransforms(os.path.join(path, "test"), images, load_gt=load_gt)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # If we don't have a point cloud, we start with random points (Objects)
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "SynCom": readSynComSceneInfo,
    "TensoIR": readTensoIRSyntheticInfo,
}