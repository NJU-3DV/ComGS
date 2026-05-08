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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static std::tuple<int,int> forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			std::function<char* (size_t)> sampleBuffer,
			const int P, int D, int M,
			const float* background,
			const bool scalar_bg,
			const int width, 
			const int height,
			const float* means3D,
			const float* dc,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* transMat_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float* materials,
			const float tan_fovx, 
			const float tan_fovy,
			const bool prefiltered,
			float* out_render,
			float* out_geometry,
			float* out_material,
			int* radii = nullptr,
			const bool flag_max_count = false,
			int* accum_max_count = nullptr,
			const bool require_render = true,
			const bool require_geometry = false,
			const bool require_material = false,
			bool debug = false
		);

		static void backward(
			const int P, int D, int M, int R, int B,
			const float* background,
			const bool scalar_bg,
			const int width, 
			const int height,
			const float* means3D,
			const float* dc,
			const float* shs,
			const float* colors_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* transMat_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float* materials,
			const float tan_fovx, 
			const float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* img_buffer,
			char* sample_buffer,
			float* dL_dout_render,
			float* dL_dout_geometry,
			float* dL_dout_material,
			float* dL_dbg,
			float* dL_dmean2D,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dtransMat,
			float* dL_ddc,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dmaterials,
			bool debug,
			const bool requires_render,
			const bool requires_geometry,
			const bool requires_material);
	};
};

#endif