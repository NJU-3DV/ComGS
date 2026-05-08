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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <iostream>
#include <stdexcept>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* dc, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_ddc, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* direct_color = ((glm::vec3*)dc) + idx;
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_ddirect_color = dL_ddc + idx;
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_ddirect_color[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[0] = dRGBdsh1 * dL_dRGB;
		dL_dsh[1] = dRGBdsh2 * dL_dRGB;
		dL_dsh[2] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[2];
		dRGBdy = -SH_C1 * sh[0];
		dRGBdz = SH_C1 * sh[1];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[3] = dRGBdsh4 * dL_dRGB;
			dL_dsh[4] = dRGBdsh5 * dL_dRGB;
			dL_dsh[5] = dRGBdsh6 * dL_dRGB;
			dL_dsh[6] = dRGBdsh7 * dL_dRGB;
			dL_dsh[7] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[3] + SH_C2[2] * 2.f * -x * sh[5] + SH_C2[3] * z * sh[6] + SH_C2[4] * 2.f * x * sh[7];
			dRGBdy += SH_C2[0] * x * sh[3] + SH_C2[1] * z * sh[4] + SH_C2[2] * 2.f * -y * sh[5] + SH_C2[4] * 2.f * -y * sh[7];
			dRGBdz += SH_C2[1] * y * sh[4] + SH_C2[2] * 2.f * 2.f * z * sh[5] + SH_C2[3] * x * sh[6];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[8] = dRGBdsh9 * dL_dRGB;
				dL_dsh[9] = dRGBdsh10 * dL_dRGB;
				dL_dsh[10] = dRGBdsh11 * dL_dRGB;
				dL_dsh[11] = dRGBdsh12 * dL_dRGB;
				dL_dsh[12] = dRGBdsh13 * dL_dRGB;
				dL_dsh[13] = dRGBdsh14 * dL_dRGB;
				dL_dsh[14] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[8] * 3.f * 2.f * xy +
					SH_C3[1] * sh[9] * yz +
					SH_C3[2] * sh[10] * -2.f * xy +
					SH_C3[3] * sh[11] * -3.f * 2.f * xz +
					SH_C3[4] * sh[12] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[13] * 2.f * xz +
					SH_C3[6] * sh[14] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[8] * 3.f * (xx - yy) +
					SH_C3[1] * sh[9] * xz +
					SH_C3[2] * sh[10] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[11] * -3.f * 2.f * yz +
					SH_C3[4] * sh[12] * -2.f * xy +
					SH_C3[5] * sh[13] * -2.f * yz +
					SH_C3[6] * sh[14] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[9] * xy +
					SH_C3[2] * sh[10] * 4.f * 2.f * yz +
					SH_C3[3] * sh[11] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[12] * 4.f * 2.f * xz +
					SH_C3[5] * sh[13] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}


__device__ void compute_transmat_aabb(
	int idx, 
	const float* Ts_precomp,
	const float3* p_origs, 
	const glm::vec2* scales, 
	const glm::vec4* rots, 
	const float* projmatrix, 
	const float* viewmatrix, 
	const int W, const int H, 
	const float3* dL_dnormals,
	const float3* dL_dmean2Ds, 
	float* dL_dTs, 
	glm::vec3* dL_dmeans, 
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots
){
	glm::mat3 T;
	float3 normal;
	glm::mat3x4 P;
	glm::mat3 R;
	glm::mat3 S;
	float3 p_orig;
	glm::vec4 rot;
	glm::vec2 scale;
	
	// Get transformation matrix of the Gaussian
	if (Ts_precomp != nullptr) {
		T = glm::mat3(
			Ts_precomp[idx * 9 + 0], Ts_precomp[idx * 9 + 1], Ts_precomp[idx * 9 + 2],
			Ts_precomp[idx * 9 + 3], Ts_precomp[idx * 9 + 4], Ts_precomp[idx * 9 + 5],
			Ts_precomp[idx * 9 + 6], Ts_precomp[idx * 9 + 7], Ts_precomp[idx * 9 + 8]
		);
		normal = {0.0, 0.0, 0.0};
	} else {
		p_orig = p_origs[idx];
		rot = rots[idx];
		scale = scales[idx];
		R = quat_to_rotmat(rot);
		S = scale_to_mat(scale, 1.0f);
		
		glm::mat3 L = R * S;
		glm::mat3x4 M = glm::mat3x4(
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

		P = world2ndc * ndc2pix;
		T = glm::transpose(M) * P;
		normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);
	}

	// Update gradients w.r.t. transformation matrix of the Gaussian
	glm::mat3 dL_dT = glm::mat3(
		dL_dTs[idx*9+0], dL_dTs[idx*9+1], dL_dTs[idx*9+2],
		dL_dTs[idx*9+3], dL_dTs[idx*9+4], dL_dTs[idx*9+5],
		dL_dTs[idx*9+6], dL_dTs[idx*9+7], dL_dTs[idx*9+8]
	);
	float3 dL_dmean2D = dL_dmean2Ds[idx];
	if(dL_dmean2D.x != 0 || dL_dmean2D.y != 0)
	{
		glm::vec3 t_vec = glm::vec3(9.0f, 9.0f, -1.0f);
		float d = glm::dot(t_vec, T[2] * T[2]);
		glm::vec3 f_vec = t_vec * (1.0f / d);
		glm::vec3 dL_dT0 = dL_dmean2D.x * f_vec * T[2];
		glm::vec3 dL_dT1 = dL_dmean2D.y * f_vec * T[2];
		glm::vec3 dL_dT3 = dL_dmean2D.x * f_vec * T[0] + dL_dmean2D.y * f_vec * T[1];
		glm::vec3 dL_df = dL_dmean2D.x * T[0] * T[2] + dL_dmean2D.y * T[1] * T[2];
		float dL_dd = glm::dot(dL_df, f_vec) * (-1.0 / d);
		glm::vec3 dd_dT3 = t_vec * T[2] * 2.0f;
		dL_dT3 += dL_dd * dd_dT3;
		dL_dT[0] += dL_dT0;
		dL_dT[1] += dL_dT1;
		dL_dT[2] += dL_dT3;

		if (Ts_precomp != nullptr) {
			dL_dTs[idx * 9 + 0] = dL_dT[0].x;
			dL_dTs[idx * 9 + 1] = dL_dT[0].y;
			dL_dTs[idx * 9 + 2] = dL_dT[0].z;
			dL_dTs[idx * 9 + 3] = dL_dT[1].x;
			dL_dTs[idx * 9 + 4] = dL_dT[1].y;
			dL_dTs[idx * 9 + 5] = dL_dT[1].z;
			dL_dTs[idx * 9 + 6] = dL_dT[2].x;
			dL_dTs[idx * 9 + 7] = dL_dT[2].y;
			dL_dTs[idx * 9 + 8] = dL_dT[2].z;
			return;
		}
	}
	
	if (Ts_precomp != nullptr) return;

	// Update gradients w.r.t. scaling, rotation, position of the Gaussian
	glm::mat3x4 dL_dM = P * glm::transpose(dL_dT);
	float3 dL_dtn = transformVec4x3Transpose(dL_dnormals[idx], viewmatrix);

	float3 p_view = transformPoint4x3(p_orig, viewmatrix);
	float cos = -sumf3(p_view * normal);
	float multiplier = cos > 0 ? 1: -1;
	dL_dtn = multiplier * dL_dtn;

	glm::mat3 dL_dRS = glm::mat3(
		glm::vec3(dL_dM[0]),
		glm::vec3(dL_dM[1]),
		glm::vec3(dL_dtn.x, dL_dtn.y, dL_dtn.z)
	);

	glm::mat3 dL_dR = glm::mat3(
		dL_dRS[0] * glm::vec3(scale.x),
		dL_dRS[1] * glm::vec3(scale.y),
		dL_dRS[2]);
	
	dL_drots[idx] = quat_to_rotmat_vjp(rot, dL_dR);
	dL_dscales[idx] = glm::vec2(
		(float)glm::dot(dL_dRS[0], R[0]),
		(float)glm::dot(dL_dRS[1], R[1])
	);
	dL_dmeans[idx] = glm::vec3(dL_dM[2]);
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means3D,
	const float* transMats,
	const int* radii,
	const float* dc,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, 
	const float focal_y,
	const float tan_fovx,
	const float tan_fovy,
	const glm::vec3* campos, 
	// grad input
	float* dL_dtransMats,
	const float* dL_dnormals,
    const bool requires_render,
	float* dL_dcolors,
	float* dL_dc,
	float* dL_dshs,
	float3* dL_dmean2Ds,
	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const int W = int(focal_x * tan_fovx * 2);
	const int H = int(focal_y * tan_fovy * 2);
	const float * Ts_precomp = (scales) ? nullptr : transMats;
	compute_transmat_aabb(
		idx, 
		Ts_precomp,
		means3D, scales, rotations, 
		projmatrix, viewmatrix, W, H, 
		(float3*)dL_dnormals, 
		dL_dmean2Ds,
		(dL_dtransMats), 
		dL_dmean3Ds, 
		dL_dscales, 
		dL_drots
	);

	if (requires_render && shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means3D, *campos, dc, shs, clamped, (glm::vec3*)dL_dcolors, (glm::vec3*)dL_dmean3Ds, (glm::vec3*)dL_dc, (glm::vec3*)dL_dshs);
	
	// hack the gradient here for densitification
	float depth = transMats[idx * 9 + 8];
	dL_dmean2Ds[idx].x = dL_dtransMats[idx * 9 + 2] * depth * 0.5 * float(W); // to ndc 
	dL_dmean2Ds[idx].y = dL_dtransMats[idx * 9 + 5] * depth * 0.5 * float(H); // to ndc
}

// Backward version of the rendering procedure.
template <bool requires_render, bool requires_geometry, bool requires_material>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const bool scalar_bg,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ transMats,
	const float* __restrict__ colors,
	const float* __restrict__ materials,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dout_render,
	const float* __restrict__ dL_dout_geometry,
	const float* __restrict__ dL_dout_material,
	float3* __restrict__ dL_dmean2D,
	float * __restrict__ dL_dtransMat,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dnormals,
	float* __restrict__ dL_dmaterials
) {
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float last_alpha = 0;
	float elm_color[NUM_CHAFFELS] = { 0 };
	float last_elm_color[NUM_CHAFFELS] = { 0 };
	float accum_rec[NUM_CHAFFELS] = { 0 };
	float dL_dpixel[NUM_CHAFFELS] = { 0 };

	float elm_geo[GEO_CHAFFELS] = { 0 };
	float last_elm_geo[GEO_CHAFFELS] = { 0 };
	float accum_rec_geo[GEO_CHAFFELS] = { 0 };
	float dL_dpixel_geo[GEO_CHAFFELS] = { 0 };

	float elm_mat[MAT_CHAFFELS] = { 0 };
	float last_elm_mat[MAT_CHAFFELS] = { 0 };
	float accum_rec_mat[MAT_CHAFFELS] = { 0 };
	float dL_dpixel_mat[MAT_CHAFFELS] = { 0 };
	
	if (inside) {
		if constexpr(requires_render){
			for (int i = 0; i < NUM_CHAFFELS; i++)
				dL_dpixel[i] = dL_dout_render[i * H * W + pix_id];
		}
		if constexpr(requires_geometry){
			for (int i = 0; i < GEO_CHAFFELS; i++)
				dL_dpixel_geo[i] = dL_dout_geometry[i * H * W + pix_id];
		}
		if constexpr(requires_material){
			for (int i = 0; i < MAT_CHAFFELS; i++)
				dL_dpixel_mat[i] = dL_dout_material[i * H * W + pix_id];
		}
	}

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// compute ray-splat intersection as before
			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			if (p.z == 0.0) continue;
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y); 
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 
			float rho = min(rho3d, rho2d);

			// compute depth
			float c_d = (s.x * Tw.x + s.y * Tw.y) + Tw.z; // Tw * [u,v,1]
			// if a point is too small, its depth is not reliable?
			// c_d = (rho3d <= rho2d) ? c_d : Tw.z; 
			if (c_d < near_n) continue;
			
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			// accumulations
			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, opa * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);

			// get features
			const int global_id = collected_id[j];
			if constexpr(requires_render){
				for (int ch = 0; ch < NUM_CHAFFELS; ch++)
					elm_color[ch] = colors[global_id * NUM_CHAFFELS + ch];
			}
			
			if constexpr (requires_geometry){
				elm_geo[ALPHA_OFFSET] = 1.0f;
				elm_geo[DEPTH_OFFSET] = c_d;
				for (int ch = 0; ch < 3; ch++){
					elm_geo[ch + NORMAL_OFFSET] = normal[ch];
				}
			}

			if constexpr (requires_material){
				for (int ch = 0; ch < MAT_CHAFFELS; ch++){
					elm_mat[ch] = materials[global_id * MAT_CHAFFELS + ch];
				}
			}

			const float dchannel_dcolor = alpha * T;
			const float w = alpha * T;
			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			float dL_dz = 0.0f;

			if constexpr(requires_render){
				for (int ch = 0; ch < NUM_CHAFFELS; ch++){
					// Update last color (to be used in the next iteration)
					accum_rec[ch] = last_alpha * last_elm_color[ch] + (1.f - last_alpha) * accum_rec[ch];
					last_elm_color[ch] = elm_color[ch];
					dL_dalpha += (elm_color[ch] - accum_rec[ch]) * dL_dpixel[ch];
					// Update the gradients w.r.t. color of the Gaussian. 
					// Atomic, since this pixel is just one of potentially
					// many that were affected by this Gaussian.
					atomicAdd(&(dL_dcolors[global_id * NUM_CHAFFELS + ch]), dchannel_dcolor * dL_dpixel[ch]);
				}
			}

			if constexpr(requires_geometry){

				accum_rec_geo[ALPHA_OFFSET] = last_alpha * last_elm_geo[ALPHA_OFFSET] + (1.f - last_alpha) * accum_rec_geo[ALPHA_OFFSET];
				last_elm_geo[ALPHA_OFFSET] = elm_geo[ALPHA_OFFSET];
				dL_dalpha += (elm_geo[ALPHA_OFFSET] - accum_rec_geo[ALPHA_OFFSET]) * dL_dpixel_geo[ALPHA_OFFSET];

				accum_rec_geo[DEPTH_OFFSET] = last_alpha * last_elm_geo[DEPTH_OFFSET] + (1.f - last_alpha) * accum_rec_geo[DEPTH_OFFSET];
				last_elm_geo[DEPTH_OFFSET] = elm_geo[DEPTH_OFFSET];
				dL_dalpha += (elm_geo[DEPTH_OFFSET] - accum_rec_geo[DEPTH_OFFSET]) * dL_dpixel_geo[DEPTH_OFFSET];
				dL_dz = dchannel_dcolor * dL_dpixel_geo[DEPTH_OFFSET];

				for(int ch = 0; ch < 3; ch++){
					// Update last color (to be used in the next iteration)
					accum_rec_geo[ch + NORMAL_OFFSET] = last_alpha * last_elm_geo[ch + NORMAL_OFFSET] + (1.f - last_alpha) * accum_rec_geo[ch + NORMAL_OFFSET];
					last_elm_geo[ch + NORMAL_OFFSET] = elm_geo[ch + NORMAL_OFFSET];
					dL_dalpha += (elm_geo[ch + NORMAL_OFFSET] - accum_rec_geo[ch + NORMAL_OFFSET]) * dL_dpixel_geo[ch + NORMAL_OFFSET];

					atomicAdd(&(dL_dnormals[global_id * 3 + ch]), dchannel_dcolor * dL_dpixel_geo[ch + NORMAL_OFFSET]);
				}
			}

			if constexpr (requires_material){
				for(int ch = 0; ch < MAT_CHAFFELS; ch++){
					// Update last color (to be used in the next iteration)
					accum_rec_mat[ch] = last_alpha * last_elm_mat[ch] + (1.f - last_alpha) * accum_rec_mat[ch];
					last_elm_mat[ch] = elm_mat[ch];
					dL_dalpha += (elm_mat[ch] - accum_rec_mat[ch]) * dL_dpixel_mat[ch];

					atomicAdd(&(dL_dmaterials[global_id * MAT_CHAFFELS + ch]), dchannel_dcolor * dL_dpixel_mat[ch]);
				}
			}

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int ch = 0; ch < NUM_CHAFFELS; ch++){
				if(scalar_bg)
					bg_dot_dpixel += bg_color[ch] * dL_dpixel[ch];
				else
					bg_dot_dpixel += bg_color[ch * H * W + pix_id] * dL_dpixel[ch];
			}
				
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			// Helpful reusable temporary variables
			const float dL_dG = nor_o.w * dL_dalpha;
			if (rho3d <= rho2d) {
				// Update gradients w.r.t. covariance of Gaussian 3x3 (T)
				const float2 dL_ds = {
					dL_dG * -G * s.x + dL_dz * Tw.x,
					dL_dG * -G * s.y + dL_dz * Tw.y
				};
				const float3 dz_dTw = {s.x, s.y, 1.0};
				const float dsx_pz = dL_ds.x / p.z;
				const float dsy_pz = dL_ds.y / p.z;
				const float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
				const float3 dL_dk = cross(l, dL_dp);
				const float3 dL_dl = cross(dL_dp, k);

				const float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
				const float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
				const float3 dL_dTw = {
					pixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * dz_dTw.x, 
					pixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * dz_dTw.y, 
					pixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz * dz_dTw.z};


				// Update gradients w.r.t. 3D covariance (3x3 matrix)
				atomicAdd(&dL_dtransMat[global_id * 9 + 0],  dL_dTu.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 1],  dL_dTu.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 2],  dL_dTu.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 3],  dL_dTv.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 4],  dL_dTv.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 5],  dL_dTv.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 6],  dL_dTw.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 7],  dL_dTw.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dTw.z);
			} else {
				// // Update gradients w.r.t. center of Gaussian 2D mean position
				const float dG_ddelx = -G * FilterInvSquare * d.x;
				const float dG_ddely = -G * FilterInvSquare * d.y;
				atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx); // not scaled
				atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely); // not scaled
				// // Propagate the gradients of depth
				atomicAdd(&dL_dtransMat[global_id * 9 + 6],  s.x * dL_dz);
				atomicAdd(&dL_dtransMat[global_id * 9 + 7],  s.y * dL_dz);
				atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dz);
			}

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}


template<bool requires_render, bool requires_geometry, bool requires_material>
__global__ void PerGaussianRenderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H, int B, //
	const uint32_t* __restrict__ per_tile_bucket_offset, //
	const uint32_t* __restrict__ bucket_to_tile, //
	const float* __restrict__ sampled_T,  //
	const float* __restrict__ sampled_ar, //
	const float* __restrict__ sampled_ar_geo, //
	const float* __restrict__ sampled_ar_mat, //
	const float* __restrict__ bg_color,
	const bool scalar_bg,  //
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ transMats,
	const float* __restrict__ colors,
	const float* __restrict__ materials, //
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const uint32_t* __restrict__ max_contrib, //
	const float* __restrict__ render_buffers, //
	const float* __restrict__ geometry_buffers, //
	const float* __restrict__ material_buffers, //
	const float* __restrict__ dL_dout_render,
    const float* __restrict__ dL_dout_geometry,
    const float* __restrict__ dL_dout_material,
	float3* __restrict__ dL_dmean2D,
	float* __restrict__ dL_dtransMat,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dnormals,
	float* __restrict__ dL_dmaterials
) {
	// global_bucket_idx = warp_idx
	auto block = cg::this_thread_block();
	auto my_warp = cg::tiled_partition<32>(block);
	uint32_t global_bucket_idx = block.group_index().x * my_warp.meta_group_size() + my_warp.meta_group_rank();
	bool valid_bucket = global_bucket_idx < (uint32_t) B;
	if (!valid_bucket) return;

	bool valid_splat = false;

	uint32_t tile_id, bbm;
	uint2 range;
	int num_splats_in_tile, bucket_idx_in_tile;
	int splat_idx_in_tile, splat_idx_global;

	tile_id = bucket_to_tile[global_bucket_idx];
	range = ranges[tile_id];
	num_splats_in_tile = range.y - range.x;
	// What is the number of buckets before me? what is my offset?
	bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
	bucket_idx_in_tile = global_bucket_idx - bbm;
	splat_idx_in_tile = bucket_idx_in_tile * 32 + my_warp.thread_rank();
	splat_idx_global = range.x + splat_idx_in_tile;
	valid_splat = (splat_idx_in_tile < num_splats_in_tile);

	// if first gaussian in bucket is useless, then others are also useless
	if (bucket_idx_in_tile * 32 >= max_contrib[tile_id]) {
		return;
	}

	// Load Gaussian properties into registers
	int gaussian_idx = 0;
	float2 xy = {0.0f, 0.0f};
	float3 Tu = {0.0f, 0.0f, 0.0f};
	float3 Tv = {0.0f, 0.0f, 0.0f};
	float3 Tw = {0.0f, 0.0f, 0.0f};
	float4 nor_o = {0.0f, 0.0f, 0.0f, 0.0f};
	
	if (valid_splat) {
		gaussian_idx = point_list[splat_idx_global];
		xy = points_xy_image[gaussian_idx];
		Tu = make_float3(
			transMats[9 * gaussian_idx + 0], 
			transMats[9 * gaussian_idx + 1], 
			transMats[9 * gaussian_idx + 2]);
		Tv = make_float3(
			transMats[9 * gaussian_idx + 3], 
			transMats[9 * gaussian_idx + 4], 
			transMats[9 * gaussian_idx + 5]);
		Tw = make_float3(
			transMats[9 * gaussian_idx + 6], 
			transMats[9 * gaussian_idx + 7], 
			transMats[9 * gaussian_idx + 8]);
		nor_o = normal_opacity[gaussian_idx];
	}

	// Gradient accumulation variables
	float Register_dL_dmean2D_x = 0.0f;
	float Register_dL_dmean2D_y = 0.0f;
	float Register_dL_dtransMat[9] = {0.0f};
	float Register_dL_dopacity = 0.0f;
	float Register_dL_dcolors[NUM_CHAFFELS] = {0.0f};
	float Register_dL_dgeometry[GEO_CHAFFELS] = {0.0f};
    float Register_dL_dmaterials[MAT_CHAFFELS] = {0.0f};
	
	// tile metadata
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 tile = {tile_id % horizontal_blocks, tile_id / horizontal_blocks};
	const uint2 pix_min = {tile.x * BLOCK_X, tile.y * BLOCK_Y};

	// values useful for gradient calculation
	float T;
	float T_final;
	float last_contributor;
	float ar[NUM_CHAFFELS];
    float ar_geo[GEO_CHAFFELS];
    float ar_mat[MAT_CHAFFELS];
	float dL_dpixel[NUM_CHAFFELS];
    float dL_dpixel_geo[GEO_CHAFFELS];
    float dL_dpixel_mat[MAT_CHAFFELS];

	// shared memory
	__shared__ float Shared_sampled_ar[32 * NUM_CHAFFELS + 1];
    __shared__ float Shared_sampled_ar_geo[32 * GEO_CHAFFELS + 1];
    __shared__ float Shared_sampled_ar_mat[32 * MAT_CHAFFELS + 1];
	if constexpr(requires_render)
		sampled_ar += global_bucket_idx * BLOCK_SIZE * NUM_CHAFFELS;
	if constexpr(requires_geometry)
		sampled_ar_geo += global_bucket_idx * BLOCK_SIZE * GEO_CHAFFELS;
	if constexpr(requires_material)
		sampled_ar_mat += global_bucket_idx * BLOCK_SIZE * MAT_CHAFFELS;

	__shared__ float Shared_pixels[32 * NUM_CHAFFELS];
    __shared__ float Shared_pixels_geo[32 * GEO_CHAFFELS];
    __shared__ float Shared_pixels_mat[32 * MAT_CHAFFELS];
   
	// iterate over all pixels in the tile
  	#pragma unroll
	for (int i = 0; i < BLOCK_SIZE + 31; ++i) {
		if (i % 32 == 0) {
            
            if constexpr(requires_render){
                #pragma unroll
                for (int ch = 0; ch < NUM_CHAFFELS; ++ch) {
                    int shift = BLOCK_SIZE * ch + i + block.thread_rank();
                    Shared_sampled_ar[ch * 32 + block.thread_rank()] = sampled_ar[shift];
                }
            }
            
            if constexpr(requires_geometry){
                #pragma unroll
                for (int ch = 0; ch < GEO_CHAFFELS; ++ch) {
                    int shift = BLOCK_SIZE * ch + i + block.thread_rank();
                    Shared_sampled_ar_geo[ch * 32 + block.thread_rank()] = sampled_ar_geo[shift];
                }
            }

            if constexpr(requires_material){
                #pragma unroll
                for (int ch = 0; ch < MAT_CHAFFELS; ++ch) {
                    int shift = BLOCK_SIZE * ch + i + block.thread_rank();
                    Shared_sampled_ar_mat[ch * 32 + block.thread_rank()] = sampled_ar_mat[shift];
                }
            }

			const uint32_t local_id = i + block.thread_rank();
			const uint2 pix = {pix_min.x + local_id % BLOCK_X, pix_min.y + local_id / BLOCK_X};
			const uint32_t id = W * pix.y + pix.x;

            if constexpr(requires_render){
                #pragma unroll
                for (int ch = 0; ch < NUM_CHAFFELS; ++ch) {
                    Shared_pixels[ch * 32 + block.thread_rank()] = render_buffers[ch * H * W + id];
                }
            }
            if constexpr(requires_geometry){
                #pragma unroll
                for (int ch = 0; ch < GEO_CHAFFELS; ++ch) {
                    Shared_pixels_geo[ch * 32 + block.thread_rank()] = geometry_buffers[ch * H * W + id];
                }
            }
            if constexpr(requires_material){
                #pragma unroll
                for (int ch = 0; ch < MAT_CHAFFELS; ++ch) {
                    Shared_pixels_mat[ch * 32 + block.thread_rank()] = material_buffers[ch * H * W + id];
                }
            }

			block.sync();
		}

		// SHUFFLING
		// At this point, T already has my (1 - alpha) multiplied.
		// So pass this ready-made T value to next thread.
		T = my_warp.shfl_up(T, 1);
		last_contributor = my_warp.shfl_up(last_contributor, 1);
		T_final = my_warp.shfl_up(T_final, 1);

        if constexpr(requires_render){
            #pragma unroll
            for (int ch = 0; ch < NUM_CHAFFELS; ++ch) {
                ar[ch] = my_warp.shfl_up(ar[ch], 1);
                dL_dpixel[ch] = my_warp.shfl_up(dL_dpixel[ch], 1);
            }
        }

        if constexpr(requires_geometry){
            #pragma unroll
            for (int ch = 0; ch < GEO_CHAFFELS; ++ch) {
                ar_geo[ch] = my_warp.shfl_up(ar_geo[ch], 1);
                dL_dpixel_geo[ch] = my_warp.shfl_up(dL_dpixel_geo[ch], 1);
            }
        }

        if constexpr(requires_material){
            #pragma unroll
            for (int ch = 0; ch < MAT_CHAFFELS; ++ch) {
                ar_mat[ch] = my_warp.shfl_up(ar_mat[ch], 1);
                dL_dpixel_mat[ch] = my_warp.shfl_up(dL_dpixel_mat[ch], 1);
            }
        }

		// which pixel index should this thread deal with?
		int idx = i - my_warp.thread_rank();
		const uint2 pix = {pix_min.x + idx % BLOCK_X, pix_min.y + idx / BLOCK_X};
		const uint32_t pix_id = W * pix.y + pix.x;
		const float2 pixf = {(float) pix.x, (float) pix.y};
		bool valid_pixel = pix.x < W && pix.y < H;
		
		// every 32nd thread should read the stored state from memory
		if (valid_splat && valid_pixel && my_warp.thread_rank() == 0 && idx < BLOCK_SIZE) {
			
			T = sampled_T[global_bucket_idx * BLOCK_SIZE + idx];
			T_final = final_Ts[pix_id];
			last_contributor = n_contrib[pix_id];

      		int ii = i % 32;

            if constexpr (requires_render) {
                #pragma unroll
                for (int ch = 0; ch < NUM_CHAFFELS; ++ch) {
                    ar[ch] = -Shared_pixels[ch * 32 + ii] + Shared_sampled_ar[ch * 32 + ii];
                    // NOTE: this is important...
                    if(scalar_bg)
                        ar[ch] += T_final * bg_color[ch];
                    else
                        ar[ch] += T_final * bg_color[ch * H * W + pix_id];
					
					dL_dpixel[ch] = dL_dout_render[ch * H * W + pix_id];
                }
            }

            if constexpr (requires_geometry) {
                #pragma unroll
                for (int ch = 0; ch < GEO_CHAFFELS; ++ch) {
                    ar_geo[ch] = -Shared_pixels_geo[ch * 32 + ii] + Shared_sampled_ar_geo[ch * 32 + ii];

					dL_dpixel_geo[ch] = dL_dout_geometry[ch * H * W + pix_id];
                }
            }

            if constexpr (requires_material) {
                #pragma unroll
                for (int ch = 0; ch < MAT_CHAFFELS; ++ch) {
                    ar_mat[ch] = -Shared_pixels_mat[ch * 32 + ii] + Shared_sampled_ar_mat[ch * 32 + ii];

					dL_dpixel_mat[ch] = dL_dout_material[ch * H * W + pix_id];
                }
            }
		}

		// do work
		if (valid_splat && valid_pixel && 0 <= idx && idx < BLOCK_SIZE) {
			if (W <= pix.x || H <= pix.y) continue;

			if (splat_idx_in_tile >= last_contributor) continue;

			// compute ray-splat intersection as before
			// Fisrt compute two homogeneous planes, See Eq. (8) in 2DGS paper.
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			if (p.z == 0.0) continue;
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y); 
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 
			float rho = min(rho3d, rho2d);

			// compute depth
			float depth = (s.x * Tw.x + s.y * Tw.y) + Tw.z; // Tw * [u,v,1]
			// if a point is too small, its depth is not reliable?
			// depth = (rho3d <= rho2d) ? depth : Tw.z; 
			if (depth < near_n) continue;

			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;
			
			const float G = exp(power);
			const float alpha = min(0.99f, opa * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			const float dchannel_dcolor = alpha * T;
	        const float one_minus_alpha_reci = 1.0f / (1.0f - alpha);

            // add the gradient contribution of this pixel to the gaussian
            float dL_dalpha = 0.0f;
			float dL_dz = 0.0f;
            if constexpr(requires_render){
                #pragma unroll
                for (int ch = 0; ch < NUM_CHAFFELS; ++ch) {
                    float tmp_c = colors[gaussian_idx * NUM_CHAFFELS + ch];

                    ar[ch] += dchannel_dcolor * tmp_c;
                    const float &dL_dchannel = dL_dpixel[ch];
                    Register_dL_dcolors[ch] += dchannel_dcolor * dL_dchannel;
                    dL_dalpha += (tmp_c * T + one_minus_alpha_reci * ar[ch]) * dL_dchannel;
                }

                float bg_dot_dpixel = 0.0f;
                #pragma unroll
                for (int ch = 0; ch < NUM_CHAFFELS; ++ch) {
                    if(scalar_bg)
                        bg_dot_dpixel += bg_color[ch] * dL_dpixel[ch];
                    else
                        bg_dot_dpixel += bg_color[ch * H * W + pix_id] * dL_dpixel[ch];
                }

                dL_dalpha += (-T_final * one_minus_alpha_reci) * bg_dot_dpixel;
            }

            if constexpr(requires_geometry){
                #pragma unroll
                for (int ch = 0; ch < GEO_CHAFFELS; ++ch) {
                    float tmp_c = 0.0;
					if(ch == ALPHA_OFFSET)
						tmp_c = 1.0;
                    else if(ch == DEPTH_OFFSET)
                        tmp_c = depth;
                    else
                        tmp_c = normal[ch - NORMAL_OFFSET];

                    ar_geo[ch] += dchannel_dcolor * tmp_c;
                    const float &dL_dchannel = dL_dpixel_geo[ch];
                    if(ch == DEPTH_OFFSET)
                        dL_dz = dchannel_dcolor * dL_dchannel;
                    else
                        Register_dL_dgeometry[ch] += dchannel_dcolor * dL_dchannel;
                    
                    dL_dalpha += (tmp_c * T + one_minus_alpha_reci * ar_geo[ch]) * dL_dchannel;
                }
            }

            if constexpr(requires_material){
                #pragma unroll
                for (int ch = 0; ch < MAT_CHAFFELS; ++ch) {
                    float tmp_c = materials[gaussian_idx * MAT_CHAFFELS + ch];

                    ar_mat[ch] += dchannel_dcolor * tmp_c;
                    const float &dL_dchannel = dL_dpixel_mat[ch];
                    Register_dL_dmaterials[ch] += dchannel_dcolor * dL_dchannel;

                    dL_dalpha += (tmp_c * T + one_minus_alpha_reci * ar_mat[ch]) * dL_dchannel;
                }
            }
			
			T *= (1.0f - alpha);

			// Helpful reusable temporary variables
			const float dL_dG = nor_o.w * dL_dalpha;				
			if (rho3d <= rho2d) {
				// Update gradients w.r.t. covariance of Gaussian 3x3 (T)
				const float2 dL_ds = {
					dL_dG * -G * s.x + dL_dz * Tw.x,
					dL_dG * -G * s.y + dL_dz * Tw.y
				};
				const float3 dz_dTw = {s.x, s.y, 1.0};
				const float dsx_pz = dL_ds.x / p.z;
				const float dsy_pz = dL_ds.y / p.z;
				const float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
				const float3 dL_dk = cross(l, dL_dp);
				const float3 dL_dl = cross(dL_dp, k);

				const float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
				const float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
				const float3 dL_dTw = {
					pixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * dz_dTw.x, 
					pixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * dz_dTw.y, 
					pixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz * dz_dTw.z};

				// Update gradients w.r.t. 3D covariance (3x3 matrix)
				Register_dL_dtransMat[0] += dL_dTu.x;
				Register_dL_dtransMat[1] += dL_dTu.y;
				Register_dL_dtransMat[2] += dL_dTu.z;
				Register_dL_dtransMat[3] += dL_dTv.x;
				Register_dL_dtransMat[4] += dL_dTv.y;
				Register_dL_dtransMat[5] += dL_dTv.z;
				Register_dL_dtransMat[6] += dL_dTw.x;
				Register_dL_dtransMat[7] += dL_dTw.y;
				Register_dL_dtransMat[8] += dL_dTw.z;
			} else {
				// // Update gradients w.r.t. center of Gaussian 2D mean position
				const float dG_ddelx = -G * FilterInvSquare * d.x;
				const float dG_ddely = -G * FilterInvSquare * d.y;

				Register_dL_dmean2D_x += dG_ddelx * dL_dG; // not scaled
				Register_dL_dmean2D_y += dG_ddely * dL_dG; // not scaled
				// // Propagate the gradients of depth
				Register_dL_dtransMat[6] += s.x * dL_dz;
				Register_dL_dtransMat[7] += s.y * dL_dz;
				Register_dL_dtransMat[8] += dL_dz;
			}

			// Update gradients w.r.t. opacity of the Gaussian
			Register_dL_dopacity += G * dL_dalpha;
		}
	}

	// finally add the gradients using atomics
	if (valid_splat) {
		# pragma unroll
		for (int i =0; i < 9; ++i) {
			atomicAdd(&dL_dtransMat[gaussian_idx * 9 + i], Register_dL_dtransMat[i]);
		}

		atomicAdd(&dL_dmean2D[gaussian_idx].x, Register_dL_dmean2D_x);
		atomicAdd(&dL_dmean2D[gaussian_idx].y, Register_dL_dmean2D_y);
		atomicAdd(&dL_dopacity[gaussian_idx], Register_dL_dopacity);

        if constexpr (requires_render){
            #pragma unroll	
            for (int ch = 0; ch < NUM_CHAFFELS; ++ch) {
                atomicAdd(&dL_dcolors[gaussian_idx * NUM_CHAFFELS + ch], Register_dL_dcolors[ch]);
            }
        }
		
		if constexpr (requires_geometry) {
			#pragma unroll
			for (int ch = 0; ch < 3; ++ch){
				atomicAdd(&dL_dnormals[gaussian_idx * 3 + ch], Register_dL_dgeometry[ch + NORMAL_OFFSET]);
			}
		}

		if constexpr (requires_material){
			#pragma unroll
			for (int ch = 0; ch < MAT_CHAFFELS; ++ch){
				atomicAdd(&dL_dmaterials[gaussian_idx * MAT_CHAFFELS + ch], Register_dL_dmaterials[ch]);
			}
		}
	}
}

__global__ void RenderBackgroundBackwardCUDAKernel(
	const int W, const int H,
	const float* __restrict__ final_Ts,
	const float* __restrict__ dL_dpixels,
	float* __restrict__ dL_dbg
){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= W * H) {
		return;
	}

	#pragma unroll
	for (int ch = 0; ch < NUM_CHAFFELS; ++ch) {
		dL_dbg[ch * W * H + idx] = final_Ts[idx] * dL_dpixels[ch * W * H + idx];
	}
}


void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* dc,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* transMats,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, 
	const float focal_y,
	const float tan_fovx, 
	const float tan_fovy,
	const glm::vec3* campos,
	float3* dL_dmean2D,
	const float* dL_dnormals,
    const bool requires_render,
	float* dL_dtransMats,
	float* dL_dcolors,
	float* dL_ddc,
	float* dL_dshs,
	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots
){
	preprocessCUDA<NUM_CHAFFELS><< <(P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		transMats,
		radii,
		dc,
		shs,
		clamped,
		(glm::vec2*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		focal_x, 
		focal_y,
		tan_fovx,
		tan_fovy,
		campos,	
		dL_dtransMats,
		dL_dnormals,
        requires_render,
		dL_dcolors,
		dL_ddc,
		dL_dshs,
		dL_dmean2D,
		dL_dmean3Ds,
		dL_dscales,
		dL_drots
	);
}

void BACKWARD::render(
	const dim3 grid, 
	const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H, int R, int B,
	const uint32_t* per_bucket_tile_offset,
	const uint32_t* bucket_to_tile,
	const float* sampled_T, 
	const float* sampled_ar,
	const float* sampled_ar_geo,
	const float* sampled_ar_mat,
	const float* bg_color,
	const bool scalar_bg,
	const float2* means2D,
	const float4* normal_opacity,
	const float* transMats,
	const float* colors,
	const float* materials,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const uint32_t* max_contrib,
	const float* render_buffers,
    const float* geometry_buffers,
    const float* material_buffers,
	float* dL_dout_render,
    float* dL_dout_geometry,
    float* dL_dout_material,
	float* dL_dbg,
	float3* dL_dmean2D,
	float* dL_dtransMat,
	float* dL_dnormals,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_dmaterials,
    const bool requires_render,
	const bool requires_geometry,
	const bool requires_material
){
	if(requires_render && !scalar_bg){
		RenderBackgroundBackwardCUDAKernel << <(H * W + 255) / 256, 256 >> > (
			W, H, final_Ts, dL_dout_render, dL_dbg
		);
	}

	const int THREADS = 32;
	if (requires_render && !requires_geometry && !requires_material){
		renderCUDA<true, false, false>
		<<<grid, block>>>(
			ranges,
			point_list,
			W, H,
			bg_color,
			scalar_bg,
			means2D,
			normal_opacity,
			transMats,
			colors,
			materials,
			final_Ts,
			n_contrib,
			dL_dout_render,
			dL_dout_geometry,
			dL_dout_material,
			dL_dmean2D,
			dL_dtransMat,
			dL_dopacity,
			dL_dcolors,
			dL_dnormals,
			dL_dmaterials
		);
		// PerGaussianRenderCUDA<true, false, false> 
		// <<<((B*32) + THREADS - 1) / THREADS,THREADS>>>(
		// 	ranges,
		// 	point_list,
		// 	W, H, B,
		// 	per_bucket_tile_offset,
		// 	bucket_to_tile,
		// 	sampled_T, 
		// 	sampled_ar,
		// 	sampled_ar_geo,
        //     sampled_ar_mat,
		// 	bg_color,
		// 	scalar_bg,
		// 	means2D,
		// 	normal_opacity,
		// 	transMats,
		// 	colors,
		// 	materials,
		// 	final_Ts,
		// 	n_contrib,
		// 	max_contrib,
		// 	render_buffers,
        //     geometry_buffers,
        //     material_buffers,
		// 	dL_dout_render,
        //     dL_dout_geometry,
        //     dL_dout_material,
		// 	dL_dmean2D,
		// 	dL_dtransMat,
		// 	dL_dopacity,
		// 	dL_dcolors,
		// 	dL_dnormals,
		// 	dL_dmaterials
        // );
	}
	else if (requires_render && requires_geometry && !requires_material){
		renderCUDA<true, true, false>
		<<<grid, block>>>(
			ranges,
			point_list,
			W, H,
			bg_color,
			scalar_bg,
			means2D,
			normal_opacity,
			transMats,
			colors,
			materials,
			final_Ts,
			n_contrib,
			dL_dout_render,
			dL_dout_geometry,
			dL_dout_material,
			dL_dmean2D,
			dL_dtransMat,
			dL_dopacity,
			dL_dcolors,
			dL_dnormals,
			dL_dmaterials
		);
		// PerGaussianRenderCUDA<true, true, false> 
		// <<<((B*32) + THREADS - 1) / THREADS,THREADS>>>(
		// 	ranges,
		// 	point_list,
		// 	W, H, B,
		// 	per_bucket_tile_offset,
		// 	bucket_to_tile,
		// 	sampled_T, 
		// 	sampled_ar,
		// 	sampled_ar_geo,
        //     sampled_ar_mat,
		// 	bg_color,
		// 	scalar_bg,
		// 	means2D,
		// 	normal_opacity,
		// 	transMats,
		// 	colors,
		// 	materials,
		// 	final_Ts,
		// 	n_contrib,
		// 	max_contrib,
		// 	render_buffers,
        //     geometry_buffers,
        //     material_buffers,
		// 	dL_dout_render,
        //     dL_dout_geometry,
        //     dL_dout_material,
		// 	dL_dmean2D,
		// 	dL_dtransMat,
		// 	dL_dopacity,
		// 	dL_dcolors,
		// 	dL_dnormals,
		// 	dL_dmaterials
        // );
	}
	else if (requires_render && requires_geometry && requires_material){
		renderCUDA<true, true, true>
		<<<grid, block>>>(
			ranges,
			point_list,
			W, H,
			bg_color,
			scalar_bg,
			means2D,
			normal_opacity,
			transMats,
			colors,
			materials,
			final_Ts,
			n_contrib,
			dL_dout_render,
			dL_dout_geometry,
			dL_dout_material,
			dL_dmean2D,
			dL_dtransMat,
			dL_dopacity,
			dL_dcolors,
			dL_dnormals,
			dL_dmaterials
		);
		// PerGaussianRenderCUDA<true, true, true> 
		// <<<((B*32) + THREADS - 1) / THREADS,THREADS>>>(
		// 	ranges,
		// 	point_list,
		// 	W, H, B,
		// 	per_bucket_tile_offset,
		// 	bucket_to_tile,
		// 	sampled_T, 
		// 	sampled_ar,
		// 	sampled_ar_geo,
        //     sampled_ar_mat,
		// 	bg_color,
		// 	scalar_bg,
		// 	means2D,
		// 	normal_opacity,
		// 	transMats,
		// 	colors,
		// 	materials,
		// 	final_Ts,
		// 	n_contrib,
		// 	max_contrib,
		// 	render_buffers,
        //     geometry_buffers,
        //     material_buffers,
		// 	dL_dout_render,
        //     dL_dout_geometry,
        //     dL_dout_material,
		// 	dL_dmean2D,
		// 	dL_dtransMat,
		// 	dL_dopacity,
		// 	dL_dcolors,
		// 	dL_dnormals,
		// 	dL_dmaterials
        // );
	}
    else if(!requires_render && requires_geometry && requires_material){
        renderCUDA<false, true, true>
		<<<grid, block>>>(
			ranges,
			point_list,
			W, H,
			bg_color,
			scalar_bg,
			means2D,
			normal_opacity,
			transMats,
			colors,
			materials,
			final_Ts,
			n_contrib,
			dL_dout_render,
			dL_dout_geometry,
			dL_dout_material,
			dL_dmean2D,
			dL_dtransMat,
			dL_dopacity,
			dL_dcolors,
			dL_dnormals,
			dL_dmaterials
		);
		// PerGaussianRenderCUDA<false, true, true>
        // <<<((B*32) + THREADS - 1) / THREADS,THREADS>>>(
        //     ranges,
        //     point_list,
        //     W, H, B,
        //     per_bucket_tile_offset,
        //     bucket_to_tile,
        //     sampled_T, 
        //     sampled_ar,
		// 	sampled_ar_geo,
        //     sampled_ar_mat,
        //     bg_color,
        //     scalar_bg,
        //     means2D,
        //     normal_opacity,
        //     transMats,
        //     colors,
        //     materials,
        //     final_Ts,
        //     n_contrib,
        //     max_contrib,
        //     render_buffers,
        //     geometry_buffers,
        //     material_buffers,
        //     dL_dout_render,
        //     dL_dout_geometry,
        //     dL_dout_material,
        //     dL_dmean2D,
        //     dL_dtransMat,
        //     dL_dopacity,
        //     dL_dcolors,
        //     dL_dnormals,
        //     dL_dmaterials
        // );
    }
	else{
		std::cerr << "Error: Incompatible combination of requires_geometry and requires_material." << std::endl;
		throw std::runtime_error("Invalid combination: requires_geometry and requires_material.");
	}
}