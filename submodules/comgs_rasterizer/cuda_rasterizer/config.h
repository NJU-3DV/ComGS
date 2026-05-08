/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHAFFELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16

#define GEO_CHAFFELS 5
#define ALPHA_OFFSET 0
#define DEPTH_OFFSET 1
#define NORMAL_OFFSET 2

#define MAT_CHAFFELS 5

#endif