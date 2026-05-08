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

#include "forward.h"
#include "auxiliary.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* dc, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* direct_color = ((glm::vec3*)dc) + idx;
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * direct_color[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[0] + SH_C1 * z * sh[1] - SH_C1 * x * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[3] +
				SH_C2[1] * yz * sh[4] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[5] +
				SH_C2[3] * xz * sh[6] +
				SH_C2[4] * (xx - yy) * sh[7];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[8] +
					SH_C3[1] * xy * z * sh[9] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[10] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[11] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[12] +
					SH_C3[5] * z * (xx - yy) * sh[13] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[14];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ void compute_transmat(
	const float3& p_orig,
	const glm::vec2 scale,
	float mod,
	const glm::vec4 rot,
	const float* projmatrix,
	const float* viewmatrix,
	const int W,
	const int H, 
	glm::mat3 &T,
	float3 &normal
) {

	glm::mat3 R = quat_to_rotmat(rot);
	glm::mat3 S = scale_to_mat(scale, mod);
	glm::mat3 L = R * S;

	// center of Gaussians in the camera coordinate
	glm::mat3x4 splat2world = glm::mat3x4(
		glm::vec4(L[0], 0.0),
		glm::vec4(L[1], 0.0),
		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
	);

	glm::mat4 world2ndc = glm::mat4(
		projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
		projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
		projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
		projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
	);

	glm::mat3x4 ndc2pix = glm::mat3x4(
		glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
		glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
		glm::vec4(0.0, 0.0, 0.0, 1.0)
	);

	T = glm::transpose(splat2world) * world2ndc * ndc2pix;
	normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);

}

// Computing the bounding box of the 2D Gaussian and its center
// The center of the bounding box is used to create a low pass filter
__device__ bool compute_aabb(
	glm::mat3 T, 
	float cutoff,
	float2& point_image,
	float2& extent
) {
	glm::vec3 t = glm::vec3(cutoff * cutoff, cutoff * cutoff, -1.0f);
	float d = glm::dot(t, T[2] * T[2]);
	if (d == 0.0) return false;
	glm::vec3 f = (1 / d) * t;

	glm::vec2 p = glm::vec2(
		glm::dot(f, T[0] * T[2]),
		glm::dot(f, T[1] * T[2])
	);

	glm::vec2 h0 = p * p - 
		glm::vec2(
			glm::dot(f, T[0] * T[0]),
			glm::dot(f, T[1] * T[1])
		);

	glm::vec2 h = sqrt(max(glm::vec2(1e-4, 1e-4), h0));
	point_image = {p.x, p.y};
	extent = {h.x, h.y};

	return true;
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(
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
	const float tan_fovx, 
	const float tan_fovy,
	const float focal_x, 
	const float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	const bool requires_render
) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Compute transformation matrix
	glm::mat3 T;
	float3 normal;
	if (transMat_precomp == nullptr)
	{
		compute_transmat(((float3*)orig_points)[idx], scales[idx], scale_modifier, rotations[idx], projmatrix, viewmatrix, W, H, T, normal);
		float3 *T_ptr = (float3*)transMats;
		T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
		T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
		T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};
	} else {
		glm::vec3 *T_ptr = (glm::vec3*)transMat_precomp;
		T = glm::mat3(
			T_ptr[idx * 3 + 0], 
			T_ptr[idx * 3 + 1],
			T_ptr[idx * 3 + 2]
		);
		normal = make_float3(0.0, 0.0, 1.0);
	}

	// We set dual visibility. 
	float cos = -sumf3(p_view * normal);
	if (cos == 0) return;
	float multiplier = cos > 0 ? 1: -1;
	normal = multiplier * normal;

	// tight bounding box
	float cutoff = sqrtf(max(9.f + 2.f * logf(opacities[idx]), 0.000001));

	// Compute center and radius
	float2 point_image;
	float radius;
	{
		float2 extent;
		bool ok = compute_aabb(T, cutoff, point_image, extent);
		if (!ok) return;
		radius = ceil(max(max(extent.x, extent.y), cutoff * FilterSize));
	}

	uint2 rect_min, rect_max;
	getRect(point_image, radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (requires_render && colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, dc, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = (int)radius;
	points_xy_image[idx] = point_image;
	normal_opacity[idx] = { normal.x, normal.y, normal.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

template <bool requires_render, bool requires_geometry, bool requires_material>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const uint32_t* __restrict__ per_tile_bucket_offset, //
	uint32_t* __restrict__ bucket_to_tile,//
	float* __restrict__ sampled_T, //
	float* __restrict__ sampled_ar,//
	float* __restrict__ sampled_ar_geo,//
	float* __restrict__ sampled_ar_mat,//
	const int W, const int H,
	const float* __restrict__ transMats,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ depths,
	const float* __restrict__ materials, //
	const float4* __restrict__ normal_opacity, 
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	uint32_t* __restrict__ max_contrib,
	const float* __restrict__ bg_color,
	const bool scalar_bg, //
	float* __restrict__ out_render,
	float* __restrict__ out_geometry,
	float* __restrict__ out_material,
	const bool flag_max_count,
	int* __restrict__ accum_max_count
) {

	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint32_t tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
	uint2 range = ranges[tile_id];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// what is the number of buckets before me? what is my offset?
	uint32_t bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
	// let's first quickly also write the bucket-to-tile mapping
	int num_buckets = (toDo + 31) / 32;
	for (int i = 0; i < (num_buckets + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
		int bucket_idx = i * BLOCK_SIZE + block.thread_rank();
		if (bucket_idx < num_buckets) {
			bucket_to_tile[bbm + bucket_idx] = tile_id;
		}
	}
	
	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	// blended elements
	float C_render[NUM_CHAFFELS] = { 0.0f }; 
	float C_geometry[GEO_CHAFFELS] = { 0.0f };
	float C_material[MAT_CHAFFELS] = { 0.0f };

	float weight_max = 0;
	int idx_max=0;
	int flag_update=0;

	int contribs = 0;
	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// add incoming T value for every 32nd gaussian
			if (j % 32 == 0) {
				sampled_T[(bbm * BLOCK_SIZE) + block.thread_rank()] = T;

				if constexpr (requires_render)
					for (int ch = 0; ch < NUM_CHAFFELS; ++ch)	
						sampled_ar[(bbm * BLOCK_SIZE * NUM_CHAFFELS) + ch * BLOCK_SIZE + block.thread_rank()] = C_render[ch];

				if constexpr (requires_geometry)
					for (int ch = 0; ch < GEO_CHAFFELS; ++ch)	
						sampled_ar_geo[(bbm * BLOCK_SIZE * GEO_CHAFFELS) + ch * BLOCK_SIZE + block.thread_rank()] = C_geometry[ch];

				if constexpr (requires_material)
					for (int ch = 0; ch < MAT_CHAFFELS; ++ch)	
						sampled_ar_mat[(bbm * BLOCK_SIZE * MAT_CHAFFELS) + ch * BLOCK_SIZE + block.thread_rank()] = C_material[ch];

				++bbm;
			}

			// Keep track of current position in range
			contributor++;

			float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];

			// Transform the two planes into local u-v system. 
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			// Cross product of two planes is a line, Eq. (9)
			float3 p = cross(k, l);
			if (p.z == 0.0) continue;
			// Perspective division to get the intersection (u,v), Eq. (10)
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y); 
			// Add low pass filter
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 
			float rho = min(rho3d, rho2d);

			// compute depth
			float depth = (s.x * Tw.x + s.y * Tw.y) + Tw.z;
			// if a point is too small, its depth is not reliable?
			// depth = (rho3d <= rho2d) ? depth : Tw.z 
			if (depth < near_n) continue;

			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;
			
			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, opa * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// render
			// Eq. (3) from 3D Gaussian splatting paper.
			float weight = alpha * T;
			if constexpr (requires_render){
				#pragma unroll
				for (int ch = 0; ch < NUM_CHAFFELS; ch++)
					C_render[ch] += features[collected_id[j] * NUM_CHAFFELS + ch] * weight;
			}

			if constexpr (requires_geometry){
				C_geometry[ALPHA_OFFSET] += weight;
				C_geometry[DEPTH_OFFSET] += weight * depth;

				#pragma unroll
				for (int ch = 0; ch < 3; ch++){
					C_geometry[ch + NORMAL_OFFSET] += normal[ch] * weight;
				}
			}

			if constexpr (requires_material){
				for (int ch = 0; ch < MAT_CHAFFELS; ch++){
					C_material[ch] += materials[collected_id[j] * MAT_CHAFFELS + ch] * weight;
				}
			}
			
			if(weight_max < weight)
			{
				weight_max = weight;
				idx_max = collected_id[j];
				flag_update = 1;
			}

			T = test_T;

			// Keep track of last range entry to update this pixel.
			last_contributor = contributor;
			contribs++;
		}
	}

	if(flag_update == 1 && flag_max_count){
		atomicAdd(&(accum_max_count[idx_max]), 1);
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;

		if constexpr (requires_render){
			#pragma unroll
			for (int ch = 0; ch < NUM_CHAFFELS; ch++){
				float tmp_out = C_render[ch];
				if (scalar_bg) tmp_out += T * bg_color[ch];
				else tmp_out += T * bg_color[ch * H * W + pix_id];
				
				out_render[ch * H * W + pix_id] = tmp_out;
			}
		}

		if constexpr(requires_geometry){
			#pragma unroll
			for (int ch = 0; ch < GEO_CHAFFELS; ch++){
				out_geometry[ch * H * W + pix_id] = C_geometry[ch];
			}
		}

		if constexpr(requires_material){
			#pragma unroll
			for (int ch = 0; ch < MAT_CHAFFELS; ch++){
				out_material[ch * H * W + pix_id] = C_material[ch];
			}
		}
	}

	// max reduce the last contributor
    typedef cub::BlockReduce<uint32_t, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    last_contributor = BlockReduce(temp_storage).Reduce(last_contributor, cub::Max());
	if (block.thread_rank() == 0) {
		max_contrib[tile_id] = last_contributor;
	}
}

void FORWARD::render(
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
	const int W, 
	const int H,
	const float* transMats,
	const float2* means2D,
	const float* depths,
	const float* colors,
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
) {
	if (requires_render && !requires_geometry && !requires_material){
		renderCUDA<true, false, false> 
		<< <grid, block >> > (
			ranges,
			point_list,
			per_tile_bucket_offset, 
			bucket_to_tile,
			sampled_T, 
			sampled_ar,
			sampled_ar_geo,
			sampled_ar_mat,
			W, H,
			transMats,
			means2D,
			colors,
			depths,
			materials,
			normal_opacity,
			final_T,
			n_contrib,
			max_contrib,
			bg_color,
			scalar_bg,
			out_render,
			out_geometry,
			out_material,
			flag_max_count,
			accum_max_count
		);
	}
	else if (requires_render && requires_geometry && !requires_material){
		renderCUDA<true, true, false> 
		<< <grid, block >> > (
			ranges,
			point_list,
			per_tile_bucket_offset, 
			bucket_to_tile,
			sampled_T, 
			sampled_ar,
			sampled_ar_geo,
			sampled_ar_mat,
			W, H,
			transMats,
			means2D,
			colors,
			depths,
			materials,
			normal_opacity,
			final_T,
			n_contrib,
			max_contrib,
			bg_color,
			scalar_bg,
			out_render,
			out_geometry,
			out_material,
			flag_max_count,
			accum_max_count
		);
	}
	else if (requires_render && requires_geometry && requires_material){
		renderCUDA<true, true, true> 
		<< <grid, block >> > (
			ranges,
			point_list,
			per_tile_bucket_offset, 
			bucket_to_tile,
			sampled_T, 
			sampled_ar,
			sampled_ar_geo,
			sampled_ar_mat,
			W, H,
			transMats,
			means2D,
			colors,
			depths,
			materials,
			normal_opacity,
			final_T,
			n_contrib,
			max_contrib,
			bg_color,
			scalar_bg,
			out_render,
			out_geometry,
			out_material,
			flag_max_count,
			accum_max_count
		);
	}
	else if(!requires_render && requires_geometry && requires_material){
		renderCUDA<false, true, true> 
		<< <grid, block >> > (
			ranges,
			point_list,
			per_tile_bucket_offset, 
			bucket_to_tile,
			sampled_T, 
			sampled_ar,
			sampled_ar_geo,
			sampled_ar_mat,
			W, H,
			transMats,
			means2D,
			colors,
			depths,
			materials,
			normal_opacity,
			final_T,
			n_contrib,
			max_contrib,
			bg_color,
			scalar_bg,
			out_render,
			out_geometry,
			out_material,
			flag_max_count,
			accum_max_count
		);
	}
	else{
		std::cerr << "This render configuration is not supported. " << std::endl;
		throw std::runtime_error("Invalid render configuration.");
	}
}

void FORWARD::preprocess(
	int P, int D, int M,
	const float* means3D,
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
	float2* means2D,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	const bool requires_render
){
	preprocessCUDA<NUM_CHAFFELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		dc,
		shs,
		clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, 
		H,
		tan_fovx, 
		tan_fovy,
		focal_x, 
		focal_y,
		radii,
		means2D,
		depths,
		transMats,
		rgb,
		normal_opacity,
		grid,
		tiles_touched,	
		prefiltered,
		requires_render
	);
}
