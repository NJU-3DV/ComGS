
#include "sample.h"
#include "utils.h"
#include <random>


namespace sample{


    __forceinline__ __device__ float RadicalInverse_VdC(unsigned int bits) {
        bits = (bits << 16u) | (bits >> 16u);
        bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
        bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
        bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
        bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
        return float(bits) * 2.3283064365386963e-10; // / 0x100000000
    }
    
    __forceinline__ __device__ float2 Hammersley(unsigned int i, unsigned int N){
        return make_float2(float(i)/float(N), RadicalInverse_VdC(i));
    }  

    __global__ void uniform_sample_cuda_kernel(
        const int P, const int N, 
        const float cos_theta_min, 
        const float cos_theta_max,
        const float3* __restrict__ normals_ptr,
        float3* __restrict__ ray_ptr,
        float* __restrict__ invpdf_ptr,
        const float random_angle
    ){
        const int pid = blockIdx.x * blockDim.x + threadIdx.x;
        const int lid = blockIdx.y * blockDim.y + threadIdx.y;

        if(pid >= P || lid >= N)
            return;
        
        // low discrepancy random 
        float2 ur = Hammersley(lid, N);
        
        // get Phi and Theta based on cdf^{-1}
        float phi = 2 * PBR_PI * ur.x;
        float cos_theta = cos_theta_min + ur.y * (cos_theta_max - cos_theta_min);
        float sin_theta = sqrtf(1 - cos_theta * cos_theta);

        // get a ray
        float3 ray = {
            sin_theta * cosf(phi),
            sin_theta * sinf(phi),
            cos_theta
        };

        // rotate a random angle around z axis
        float cos_angle = cosf(random_angle);
        float sin_angle = sinf(random_angle);
        float3 rotated_ray = {
            ray.x * cos_angle - ray.y * sin_angle,
            ray.x * sin_angle + ray.y * cos_angle,
            ray.z
        };

        // rotate z axis to normal
        float3 normal = normals_ptr[pid];
        ray = normalize(tangent2world(normal, rotated_ray));

        // inverse pdf
        invpdf_ptr[pid * N + lid] = 2 * PBR_PI;
        ray_ptr[pid * N + lid] = ray;
    }

    std::tuple<torch::Tensor, torch::Tensor>
    uniform_sample(
        const torch::Tensor& normals,
        const int num_ray,
        const float cos_theta_min,
        const float cos_theta_max,
        const bool random_rotate
    ){
        const int P = normals.size(0);
        auto float_opts = normals.options().dtype(torch::kFloat32);

        torch::Tensor ray = torch::empty({P, num_ray, 3}, float_opts);
        torch::Tensor invpdf = torch::empty({P, num_ray, 1}, float_opts);

        const int thread_n = min((num_ray + 7) / 8 * 8, PBR_MAX_RAY);
        const int thread_p = PBR_MAX_RAY / thread_n;

        dim3 blocks((P + thread_p - 1) / thread_p, (num_ray + thread_n - 1) / thread_n, 1);
        dim3 threads(thread_p, thread_n, 1);

        float random_angle = 0.0f;
        // a random angle for training...
        if(random_rotate){
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(0.0f, 2.0f * M_PI);
            random_angle = dis(gen);     
        }
        
        uniform_sample_cuda_kernel<<<blocks, threads>>>(
            P, num_ray, cos_theta_min, cos_theta_max,
            (float3*)normals.contiguous().data_ptr<float>(),
            (float3*)ray.contiguous().data_ptr<float>(),
            invpdf.contiguous().data_ptr<float>(),
            random_angle
        );

        return std::make_tuple(ray, invpdf);
    }
}

