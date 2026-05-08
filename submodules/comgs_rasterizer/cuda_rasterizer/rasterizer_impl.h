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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* transMat;
		float4* normal_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;
		float* grad_normal;

		static GeometryState fromChunk(
			char*& chunk, size_t P, 
			const bool requires_render=true,
			const bool requires_geometry=false, 
			const bool requires_material=false
		);
	};

	struct ImageState
	{
		uint32_t *bucket_count;//
		uint32_t *bucket_offsets;//
		size_t bucket_count_scan_size;//
		char * bucket_count_scanning_space;//
		float* render_buffers;//
		float* geometry_buffers;//
		float* material_buffers;//
		uint32_t* max_contrib;//

		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(
			char*& chunk, size_t N, 
			const bool requires_render=true, 
			const bool requires_geometry=false, 
			const bool requires_material=false
		);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(
			char*& chunk, size_t P, 
			const bool requires_render=true,
			const bool requires_geometry=false, 
			const bool requires_material=false
		);
	};

	struct SampleState
	{
		uint32_t *bucket_to_tile;
		float *T;
		float *ar;
		float *ar_geo;
		float *ar_mat;
		static SampleState fromChunk(
			char*& chunk, size_t C, 
			const bool requires_render=true,
			const bool requires_geometry=false, 
			const bool requires_material=false
		);
	};

	template<typename T> 
	size_t required(
		size_t P, 
		const bool requires_render=true,
		const bool requires_geometry=false, 
		const bool requires_material=false
	) {
		char* size = nullptr;
		T::fromChunk(size, P, requires_render, requires_geometry, requires_material);
		return ((size_t)size) + 128;
	}
};