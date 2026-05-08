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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], composition=False, load_gt=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration and not composition:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if "TensoIR" in args.source_path:
            print("Found keyword TensoIR, assuming TensoIR data set!")
            scene_info = sceneLoadTypeCallbacks["TensoIR"](
                args.source_path, args.eval,
                load_prior=args.load_prior,
                load_gt=load_gt
            )
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            print("Found sparse folder, assuming COLMAP data set!")
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval, 
                load_prior=args.load_prior, composition=composition
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.eval, 
                load_prior=args.load_prior,
                load_gt=load_gt
            )
        elif os.path.exists(os.path.join(args.source_path, "cameras.json")) or os.path.exists(os.path.join(args.source_path, "train/cameras.json")):
            print("Found cameras.json, assuming SynCom data set!")
            scene_info = sceneLoadTypeCallbacks["SynCom"](
                args.source_path, args.images, args.eval, 
                load_prior=args.load_prior,
                load_gt=load_gt, 
                composition=composition
            )
        else:
            assert False, "Could not recognize scene type!"
            
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        if not composition:
            if self.loaded_iter:
                self.gaussians.load_ply(os.path.join(
                    self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"),
                    args.train_test_exp)
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getTrainCameras_warmup(self, iteration, warmup_scales, warmup_iterations):

        if len(warmup_scales) == 1:
            return self.train_cameras[warmup_scales[0]]
        
        if len(warmup_scales) != len(warmup_iterations) + 1:
            raise ValueError

        if iteration < warmup_iterations[0]:
            scale_index = 0
        elif iteration >= warmup_iterations[-1]:
            scale_index = -1
        else:
            for i in range(len(warmup_iterations)):
                if warmup_iterations[i] <= iteration < warmup_iterations[i + 1]:
                    scale_index = i + 1
                    break

        scale = warmup_scales[scale_index]
        return self.train_cameras[scale]
    
    def generateCameras(self, center, radius, normal, num_cams):
        """
        Generate camera positions on a hemisphere centered at a given point with cameras pointing towards the center.
        Args:
            center (numpy.ndarray): Center point of the hemisphere as a 3D coordinate [x, y, z].
            radius (float): Radius of the hemisphere.
            num_cams (int): Number of camera positions to generate.
        Returns:
            list: List of camera positions as 3D coordinates.
        """
        pass