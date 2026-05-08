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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <functional>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(
		int P, int D, int M,
		const float* orig_points,
		const glm::vec2* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* dc,
		const float* shs,
		bool* clamped,
		const float* transMat_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, 
		const int H,
		const float focal_x,
		const float focal_y,
		const float tan_fovx,
		const float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* transMats,
		float* colors,
		float4* normal_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		const bool requires_render
	);

	// Main rasterization method.
	void render(
		const dim3 grid, 
		const dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		const uint32_t* per_tile_bucket_offset, 
		uint32_t* bucket_to_tile,
		float* sampled_T, 
		float* sampled_ar,
		float* sampled_ar_geo,
		float* sampled_ar_mat,
		const int W, const int H,
		const float* transMats,
		const float2* points_xy_image,
		const float* depths, 
		const float* features,
		const float* materials,
		const float4* normal_opacity,
		float* final_T,
		uint32_t* n_contrib,
		uint32_t* max_contrib,
		const float* bg_color,
		const bool scalar_bg,
		float* out_render,
		float* out_geometry,
		float* out_material,
		const bool flag_max_count,
		int* accum_max_count,
		const bool requires_render,
		const bool requires_geometry,
		const bool requires_material
	);
}


#endif