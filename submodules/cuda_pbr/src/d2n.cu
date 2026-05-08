
#include "d2n.h"

namespace d2n{

    __global__ void __launch_bounds__(D2N_BLOCK_X * D2N_BLOCK_Y)
    Depth2SurfaceXYZForwardCUDAKernel(
        const float* __restrict__ intr,
        const float* __restrict__ depth,
        const int32_t H, const int32_t W,
        float* __restrict__ sxyz
    ){
        const uint2 p = {
            blockIdx.x * blockDim.x + threadIdx.x, 
            blockIdx.y * blockDim.y + threadIdx.y
        };
        
        // Check if this thread is associated with a valid pixel or outside.
        bool withinBounds = p.x < W && p.y < H;
        if (!withinBounds)
            return;

        int32_t pidx = W * p.y + p.x;
        float d = depth[pidx];
        if(d < 1e-10f)
            return;
        
        float x_comp = (p.x + 0.5f - intr[2]) / intr[0];
        float y_comp = (p.y + 0.5f - intr[3]) / intr[1];
        
        sxyz[pidx] = x_comp * d;
        sxyz[H * W + pidx] = y_comp * d;
        sxyz[2 * H * W + pidx] = d;
    }

    __global__ void __launch_bounds__(D2N_BLOCK_X * D2N_BLOCK_Y)
    SurfaceXYZ2NormalForwardCUDAKernel(
        const float* __restrict__ sxyz,
        const float* __restrict__ depth,
        const int32_t H, const int32_t W,
        const float* __restrict__ Rc2w,
        float* __restrict__ normal)
    {
        const uint2 p = {
            blockIdx.x * blockDim.x + threadIdx.x, 
            blockIdx.y * blockDim.y + threadIdx.y
        };
        
        // Check if this thread is associated with a valid pixel or outside.
        bool withinBounds = p.x < W && p.y < H;
        if (!withinBounds) 
            return;
        
        int32_t HW = H * W;
        int32_t pidx00 = W * (p.y==0? 0:p.y-1) + (p.x==0? 0:p.x-1);
        int32_t pidx01 = W * (p.y==0? 0:p.y-1) + p.x;
        int32_t pidx02 = W * (p.y==0? 0:p.y-1) + (p.x==W-1? W-1:p.x+1);
        int32_t pidx10 = W * p.y + (p.x==0? 0:p.x-1);
        int32_t pidx11 = W * p.y + p.x;
        int32_t pidx12 = W * p.y + (p.x==W-1? W-1:p.x + 1);
        int32_t pidx20 = W * (p.y==H-1?H-1:p.y+1) + (p.x==0?0:p.x-1);
        int32_t pidx21 = W * (p.y==H-1?H-1:p.y+1) + p.x;
        int32_t pidx22 = W * (p.y==H-1?H-1:p.y+1) + (p.x==W-1?W-1:p.x+1);

        // depth == 0, skip
        if(depth[pidx11] < 1e-10f)
            return;
        
        float xyz00[3] = {sxyz[pidx00],sxyz[HW + pidx00],sxyz[2 * HW + pidx00]};
        float xyz01[3] = {sxyz[pidx01],sxyz[HW + pidx01],sxyz[2 * HW + pidx01]};
        float xyz02[3] = {sxyz[pidx02],sxyz[HW + pidx02],sxyz[2 * HW + pidx02]};
        float xyz10[3] = {sxyz[pidx10],sxyz[HW + pidx10],sxyz[2 * HW + pidx10]};
        float xyz11[3] = {sxyz[pidx11],sxyz[HW + pidx11],sxyz[2 * HW + pidx11]};
        float xyz12[3] = {sxyz[pidx12],sxyz[HW + pidx12],sxyz[2 * HW + pidx12]};
        float xyz20[3] = {sxyz[pidx20],sxyz[HW + pidx20],sxyz[2 * HW + pidx20]};
        float xyz21[3] = {sxyz[pidx21],sxyz[HW + pidx21],sxyz[2 * HW + pidx21]};
        float xyz22[3] = {sxyz[pidx22],sxyz[HW + pidx22],sxyz[2 * HW + pidx22]};

        float grad_a[3], grad_b[3];
        #pragma unrolls
        for (int32_t i = 0; i < 3; i++){
            grad_a[i] = -0.125f * xyz00[i] + 0.125f * xyz02[i] - 0.25f * xyz10[i] 
                        + 0.25f * xyz12[i] - 0.125f * xyz20[i] + 0.125f * xyz22[i];
            grad_b[i] = -0.125f * xyz00[i] - 0.25f * xyz01[i] - 0.125f * xyz02[i] 
                        + 0.125f * xyz20[i] + 0.25f * xyz21[i] + 0.125f * xyz22[i];
        }

        float3 n = {
            grad_a[1] * grad_b[2] - grad_a[2] * grad_b[1],
            -grad_a[0] * grad_b[2] + grad_a[2] * grad_b[0],
            grad_a[0] * grad_b[1] - grad_a[1] * grad_b[0]
        };

        float norm = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
        if (norm < 1e-10f)
            return ;
        
        n.x = -n.x / norm;
        n.y = -n.y / norm;
        n.z = -n.z / norm;

        // go to world coordinate system
        float3 n_wld = {
            Rc2w[0] * n.x + Rc2w[1] * n.y + Rc2w[2] * n.z,
            Rc2w[3] * n.x + Rc2w[4] * n.y + Rc2w[5] * n.z,
            Rc2w[6] * n.x + Rc2w[7] * n.y + Rc2w[8] * n.z
        };

        normal[pidx11] = n_wld.x;
        normal[HW + pidx11] = n_wld.y;
        normal[2 * HW + pidx11] = n_wld.z;
    }
    
    std::tuple<torch::Tensor, torch::Tensor> Depth2Normal(
        const torch::Tensor &intr,
        const torch::Tensor &Rc2w,
        const torch::Tensor &depth
    ){
        const int H = depth.size(0);
        const int W = depth.size(1);

        torch::Tensor sxyz = torch::zeros({3, H, W}, depth.options());
        torch::Tensor normal = torch::zeros({3, H, W}, depth.options());

        dim3 blocks((W + D2N_BLOCK_X - 1) / D2N_BLOCK_X, (H + D2N_BLOCK_Y - 1) / D2N_BLOCK_Y);
        dim3 threads(D2N_BLOCK_X, D2N_BLOCK_Y, 1);

        Depth2SurfaceXYZForwardCUDAKernel <<<blocks, threads>>>(
            intr.contiguous().data_ptr<float>(),
            depth.contiguous().data_ptr<float>(),
            H, W,
            sxyz.contiguous().data_ptr<float>()
        );

        SurfaceXYZ2NormalForwardCUDAKernel <<<blocks, threads>>>(
            sxyz.contiguous().data_ptr<float>(),
            depth.contiguous().data_ptr<float>(),
            H, W,
            Rc2w.contiguous().data_ptr<float>(),
            normal.contiguous().data_ptr<float>()
        );

        return std::make_tuple(normal, sxyz); 
    }

    __global__ void __launch_bounds__(D2N_BLOCK_X * D2N_BLOCK_Y)
    SurfaceXYZ2NormalBackwardCUDAKernel(
        const float* __restrict__ sxyz,
        const float* __restrict__ depth,
        const int32_t H, const int32_t W,
        const float* __restrict__ Rc2w,
        const float* __restrict__ grad_normal,
        float* __restrict__ grad_sxyz
    ) {
        uint2 p = {
            blockIdx.x * blockDim.x + threadIdx.x, 
            blockIdx.y * blockDim.y + threadIdx.y
        };
        
        // Check if this thread is associated with a valid pixel or outside.
        bool withinBounds = p.x < W && p.y < H;
        if (!withinBounds) 
            return;
        
        int32_t HW = H * W;
        int32_t pidx00 = W * (p.y==0? 0:p.y-1) + (p.x==0? 0:p.x-1);
        int32_t pidx01 = W * (p.y==0? 0:p.y-1) + p.x;
        int32_t pidx02 = W * (p.y==0? 0:p.y-1) + (p.x==W-1? W-1:p.x+1);
        int32_t pidx10 = W * p.y + (p.x==0? 0:p.x-1);
        int32_t pidx11 = W * p.y + p.x;
        int32_t pidx12 = W * p.y + (p.x==W-1? W-1:p.x + 1);
        int32_t pidx20 = W * (p.y==H-1?H-1:p.y+1) + (p.x==0?0:p.x-1);
        int32_t pidx21 = W * (p.y==H-1?H-1:p.y+1) + p.x;
        int32_t pidx22 = W * (p.y==H-1?H-1:p.y+1) + (p.x==W-1?W-1:p.x+1);

        // depth == 0, skip
        if(depth[pidx11] < 1e-10f)
            return;

        float xyz00[3] = {sxyz[pidx00],sxyz[HW + pidx00],sxyz[2 * HW + pidx00]};
        float xyz01[3] = {sxyz[pidx01],sxyz[HW + pidx01],sxyz[2 * HW + pidx01]};
        float xyz02[3] = {sxyz[pidx02],sxyz[HW + pidx02],sxyz[2 * HW + pidx02]};
        float xyz10[3] = {sxyz[pidx10],sxyz[HW + pidx10],sxyz[2 * HW + pidx10]};
        // float xyz11[3] = {sxyz[pidx11],sxyz[HW + pidx11],sxyz[2 * HW + pidx11]};
        float xyz12[3] = {sxyz[pidx12],sxyz[HW + pidx12],sxyz[2 * HW + pidx12]};
        float xyz20[3] = {sxyz[pidx20],sxyz[HW + pidx20],sxyz[2 * HW + pidx20]};
        float xyz21[3] = {sxyz[pidx21],sxyz[HW + pidx21],sxyz[2 * HW + pidx21]};
        float xyz22[3] = {sxyz[pidx22],sxyz[HW + pidx22],sxyz[2 * HW + pidx22]};

        float grad_a[3], grad_b[3];
        for (int32_t i = 0; i < 3; i++){
            grad_a[i] = -0.125f * xyz00[i] + 0.125f * xyz02[i] - 0.25f * xyz10[i] + 0.25f * xyz12[i] - 0.125f * xyz20[i] + 0.125f * xyz22[i];
            grad_b[i] = -0.125f * xyz00[i] - 0.25f * xyz01[i] - 0.125f * xyz02[i] + 0.125f * xyz20[i] + 0.25f * xyz21[i] + 0.125f * xyz22[i];
        }

        float3 n = {
            grad_a[1] * grad_b[2] - grad_a[2] * grad_b[1],
            -grad_a[0] * grad_b[2] + grad_a[2] * grad_b[0],
            grad_a[0] * grad_b[1] - grad_a[1] * grad_b[0]
        };

        float norm = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
        if (norm < 1e-10f)
            return;

        n.x = -n.x / norm;
        n.y = -n.y / norm;
        n.z = -n.z / norm;

        // go to world coordinate system
        // float3 n_wld = {
        // 	Rc2w[0] * n.x + Rc2w[1] * n.y + Rc2w[2] * n.z,
        // 	Rc2w[3] * n.x + Rc2w[4] * n.y + Rc2w[5] * n.z,
        // 	Rc2w[6] * n.x + Rc2w[7] * n.y + Rc2w[8] * n.z
        // };

        float3 grad_n_wld = {
            grad_normal[pidx11], 
            grad_normal[HW + pidx11], 
            grad_normal[2 * HW + pidx11]
        }; 

        float3 grad_n_cam = {
            Rc2w[0] * grad_n_wld.x + Rc2w[3] * grad_n_wld.y + Rc2w[6] * grad_n_wld.z,
            Rc2w[1] * grad_n_wld.x + Rc2w[4] * grad_n_wld.y + Rc2w[7] * grad_n_wld.z,
            Rc2w[2] * grad_n_wld.x + Rc2w[5] * grad_n_wld.y + Rc2w[8] * grad_n_wld.z
        };

        float3 grad_n = {
            grad_n_cam.x * (n.x * n.x - 1) 
            + grad_n_cam.y * n.x * n.y 
            + grad_n_cam.z * n.x * n.z,
            grad_n_cam.x * n.x * n.y 
            + grad_n_cam.y * (n.y * n.y - 1) 
            + grad_n_cam.z * n.y * n.z,
            grad_n_cam.x * n.x * n.z 
            + grad_n_cam.y * n.y * n.z 
            + grad_n_cam.z * (n.z * n.z - 1),
        };

        grad_n.x /= norm;
        grad_n.y /= norm;
        grad_n.z /= norm;

        float3 grad_ga = {
            - grad_n.y * grad_b[2] + grad_n.z * grad_b[1],
            grad_n.x  * grad_b[2] - grad_n.z * grad_b[0],
            - grad_n.x * grad_b[1] + grad_n.y * grad_b[0]
        };

        float3 grad_gb = {
            grad_n.y * grad_a[2] - grad_n.z * grad_a[1],
            - grad_n.x * grad_a[2] + grad_n.z * grad_a[0],
            grad_n.x * grad_a[1] - grad_n.y * grad_a[0]
        };

        atomicAdd(&grad_sxyz[pidx00], -0.125f * grad_ga.x - 0.125f * grad_gb.x);
        atomicAdd(&grad_sxyz[HW + pidx00], -0.125f * grad_ga.y - 0.125f * grad_gb.y);
        atomicAdd(&grad_sxyz[2 * HW + pidx00], -0.125f * grad_ga.z - 0.125f * grad_gb.z);

        atomicAdd(&grad_sxyz[pidx01], -0.25f * grad_gb.x);
        atomicAdd(&grad_sxyz[HW + pidx01], -0.25 * grad_gb.y);
        atomicAdd(&grad_sxyz[2 * HW + pidx01], -0.25 * grad_gb.z);

        atomicAdd(&grad_sxyz[pidx02], 0.125f * grad_ga.x - 0.125f * grad_gb.x);
        atomicAdd(&grad_sxyz[HW + pidx02], 0.125f * grad_ga.y - 0.125f * grad_gb.y);
        atomicAdd(&grad_sxyz[2 * HW + pidx02], 0.125f * grad_ga.z - 0.125f * grad_gb.z);
        
        atomicAdd(&grad_sxyz[pidx10], - 0.25f * grad_ga.x);
        atomicAdd(&grad_sxyz[HW + pidx10], - 0.25f * grad_ga.y);
        atomicAdd(&grad_sxyz[2 * HW + pidx10], - 0.25f * grad_ga.z);

        atomicAdd(&grad_sxyz[pidx12], 0.25f * grad_ga.x);
        atomicAdd(&grad_sxyz[HW + pidx12], 0.25f * grad_ga.y);
        atomicAdd(&grad_sxyz[2 * HW + pidx12], 0.25f * grad_ga.z);

        atomicAdd(&grad_sxyz[pidx20], -0.125f * grad_ga.x + 0.125f * grad_gb.x);
        atomicAdd(&grad_sxyz[HW + pidx20], -0.125f * grad_ga.y + 0.125f * grad_gb.y);
        atomicAdd(&grad_sxyz[2 * HW + pidx20], -0.125f * grad_ga.z + 0.125f * grad_gb.z);
        
        atomicAdd(&grad_sxyz[pidx21], 0.25f * grad_gb.x);
        atomicAdd(&grad_sxyz[HW + pidx21], 0.25f * grad_gb.y);
        atomicAdd(&grad_sxyz[2 * HW + pidx21], 0.25f * grad_gb.z);

        atomicAdd(&grad_sxyz[pidx22], 0.125f * grad_ga.x + 0.125f * grad_gb.x);
        atomicAdd(&grad_sxyz[HW + pidx22], 0.125f * grad_ga.y + 0.125f * grad_gb.y);
        atomicAdd(&grad_sxyz[2 * HW + pidx22], 0.125f * grad_ga.z + 0.125f * grad_gb.z);
    }

    __global__ void __launch_bounds__(D2N_BLOCK_X * D2N_BLOCK_Y)
    Depth2SurfaceXYZBackwardCUDAKernel(
        const float* __restrict__ intr,
        const float* __restrict__ depth,
        const int32_t H, const int32_t W,
        const float* __restrict__ grad_sxyz,
        float* __restrict__ grad_rdepth
    ){
        uint2 p = {
            blockIdx.x * blockDim.x + threadIdx.x,
            blockIdx.y * blockDim.y + threadIdx.y
        };
        
        // Check if this thread is associated with a valid pixel or outside.
        bool withinBounds = p.x < W && p.y < H;
        if (!withinBounds) 
            return;
        
        int32_t pidx = W * p.y + p.x;
        float d = depth[pidx];
        if(d < 1e-10f)
            return;

        float4 intr_reg = {intr[0], intr[1], intr[2], intr[3]};
        float3 grad_sxyz_reg = {grad_sxyz[pidx], grad_sxyz[H * W + pidx], grad_sxyz[2 * H * W + pidx]};
        float grad_depths_reg = 0.0f;

        float x_comp = (p.x + 0.5f - intr_reg.z) / intr_reg.x;
        float y_comp = (p.y + 0.5f - intr_reg.w) / intr_reg.y;

        grad_depths_reg += x_comp * grad_sxyz_reg.x;
        grad_depths_reg += y_comp * grad_sxyz_reg.y;
        grad_depths_reg += grad_sxyz_reg.z;

        grad_rdepth[pidx] += grad_depths_reg;
    }

    torch::Tensor Depth2NormalBackward(
        const torch::Tensor &intr,
        const torch::Tensor &Rc2w,
        const torch::Tensor &depth,
        const torch::Tensor &sxyz,
        const torch::Tensor &normal,
        const torch::Tensor &dL_dnormal
    ){
        const int H = depth.size(0);
        const int W = depth.size(1);

        torch::Tensor grad_sxyz = torch::zeros_like(sxyz);
        torch::Tensor grad_depth = torch::zeros_like(depth);

        dim3 blocks((W + D2N_BLOCK_X - 1) / D2N_BLOCK_X, (H + D2N_BLOCK_Y - 1) / D2N_BLOCK_Y);
        dim3 threads(D2N_BLOCK_X, D2N_BLOCK_Y, 1);

        SurfaceXYZ2NormalBackwardCUDAKernel<<<blocks, threads>>>(
            sxyz.contiguous().data_ptr<float>(),
            depth.contiguous().data_ptr<float>(),
            H, W,
            Rc2w.contiguous().data_ptr<float>(),
            dL_dnormal.contiguous().data_ptr<float>(),
            grad_sxyz.contiguous().data_ptr<float>()
        );

        Depth2SurfaceXYZBackwardCUDAKernel <<<blocks, threads>>>(
            intr.contiguous().data_ptr<float>(),
            depth.contiguous().data_ptr<float>(),
            H, W,
            grad_sxyz.contiguous().data_ptr<float>(),
            grad_depth.contiguous().data_ptr<float>()
        );

        return grad_depth;
    }

}

