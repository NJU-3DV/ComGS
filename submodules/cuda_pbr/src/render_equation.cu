
# include "render_equation.h"


namespace pbr{

    __global__ void RenderingEquationKernel(
        const int P, const int L,
        const float3* __restrict__ normal_ptr,
        const float3* __restrict__ albedo_ptr,
        const float* __restrict__ metallic_ptr,
        const float* __restrict__ roughness_ptr,
        const float3* __restrict__ inray_ptr,
        const float3* __restrict__ radiance_ptr,
        const float* __restrict__ inarea_ptr,
        const float3* __restrict__ outray_ptr,
        float3* __restrict__ out_rgb_ptr
    ){
        const int pid = blockIdx.x * blockDim.x + threadIdx.x;
        const int lid = blockIdx.y * blockDim.y + threadIdx.y;

        if(pid >= P || lid >= L)
            return;
        
        float3 normal = normal_ptr[pid];
        float3 albedo = albedo_ptr[pid];
        float metallic = metallic_ptr[pid];
        float roughness = roughness_ptr[pid];

        float3 ldir = inray_ptr[pid * L + lid];
        float3 radiance = radiance_ptr[pid * L + lid];
        float inarea = inarea_ptr[pid * L + lid];
        float3 vdir = outray_ptr[pid];

        // dots
        float3 half = normalize(ldir + vdir);
        float h_dot_n = dotf(half, normal);
        float n_dot_l = dotf(normal, ldir);
        float n_dot_v = dotf(normal, vdir);
        float h_dot_v = dotf(half, vdir);

        // clamped
        bool is_hdn_clamped = h_dot_n < 0.0f;
        bool is_ndl_clamped = n_dot_l < 0.0f;
        bool is_ndv_clamped = n_dot_v < 0.0f;
        bool is_hdv_clamped = h_dot_v < 0.0f;

        h_dot_n = is_hdn_clamped ? 0.0f : h_dot_n;
        n_dot_l = is_ndl_clamped ? 0.0f : n_dot_l;
        n_dot_v = is_ndv_clamped ? 0.0f : n_dot_v;
        h_dot_v = is_hdv_clamped ? 0.0f : h_dot_v;

        // Fresnel term: active microgeometry normal h must be substituted for the surface normal n:
        float3 F0 = 0.04f * (1.0f - metallic) + albedo * metallic;
        float3 F = F0 + (1.0f - F0) * powf(1.0f - h_dot_v, 5.0f);

        // normal distribution term (GGX) 
        float a = roughness * roughness;
        float a2 = powf(a, 2.0f);
        float denom = powf(h_dot_n, 2.0f) * (a2 - 1.0f) + 1.0f;
        float D = a2 / (PBR_PI * powf(denom, 2.0f) + PBR_EPS);

        // Geometry Function (Smith)
        float ggx1 = geometry_schlick_ggx(n_dot_v, roughness);
        float ggx2 = geometry_schlick_ggx(n_dot_l, roughness);
        float G = ggx1 * ggx2;

        // Specular term
        float3 specular = F * D * G / (4.0f * n_dot_v * n_dot_l + PBR_EPS);

        // diffuse component kd
        float3 kd = (1.0f - F) * (1.0f - metallic);

        // Lambertian diffuse
        float3 diffuse = kd * albedo / PBR_PI;

        // transport
        float3 transport = radiance * inarea * n_dot_l;
        
        int idx = pid * L + lid;
        out_rgb_ptr[idx] = (diffuse + specular) * transport;
    }

    torch::Tensor RenderingEquation(
        const torch::Tensor& normals,
        const torch::Tensor& albedo,
        const torch::Tensor& metallic,
        const torch::Tensor& roughness,
        const torch::Tensor& inray,
        const torch::Tensor& radiance,
        const torch::Tensor& inarea,
        const torch::Tensor& outray
    ){
        const int P = normals.size(0);
        const int L = inray.size(1);

        // config kernel 
        const int thread_l = min((L + 7) / 8 * 8, PBR_MAX_RAY);
        const int thread_p = PBR_MAX_RAY / thread_l;

        dim3 blocks((P + thread_p - 1) / thread_p, (L + thread_l - 1) / thread_l, 1);
        dim3 threads(thread_p, thread_l, 1);

        auto float_opts = normals.options().dtype(torch::kFloat32);
        auto bool_opts = normals.options().dtype(torch::kBool);
        torch::Tensor rgb = torch::empty({P, L, 3}, float_opts);

        RenderingEquationKernel<<<blocks, threads>>>(
            P, L,
            (float3*)normals.contiguous().data_ptr<float>(),
            (float3*)albedo.contiguous().data_ptr<float>(),
            metallic.contiguous().data_ptr<float>(),
            roughness.contiguous().data_ptr<float>(),
            (float3*)inray.contiguous().data_ptr<float>(),
            (float3*)radiance.contiguous().data_ptr<float>(),
            inarea.contiguous().data_ptr<float>(),
            (float3*)outray.contiguous().data_ptr<float>(),
            (float3*)rgb.contiguous().data_ptr<float>()
        );
    
        return rgb;
    }

    __forceinline__ __device__ float grad_geometry_schlick_ggx(
        const float& n_dot_x, 
        const float& roughness,
        const float& dL_dggx,
        const bool is_ndx_clamped,
        float& dL_dndx,
        float& dL_droughness
    ) {
        float r = roughness + 1.0f;
        float k = powf(r, 2) / 8.0f;
        float denom = n_dot_x * (1.0f - k) + k + PBR_EPS;
        
        float dL_dk = dL_dggx * n_dot_x * (n_dot_x - 1.0f) / (denom * denom);
        dL_dndx += is_ndx_clamped ? 0.0f: dL_dggx * k / (denom * denom);
        
        dL_droughness += dL_dk * (roughness + 1.0f) / 4.0f;
    }

    __global__ void RenderingEquationBackwardKernel(
        const int P, const int L,
        const float3* __restrict__ normal_ptr,
        const float3* __restrict__ albedo_ptr,
        const float* __restrict__ metallic_ptr,
        const float* __restrict__ roughness_ptr,
        const float3* __restrict__ inray_ptr,
        const float3* __restrict__ radiance_ptr,
        const float* __restrict__ inarea_ptr,
        const float3* __restrict__ outray_ptr,
        const float3* __restrict__ dL_drgb_ptr,
        float3* __restrict__ dL_dnormal_ptr,
        float3* __restrict__ dL_dalbedo_ptr,
        float* __restrict__ dL_dmetallic_ptr,
        float* __restrict__ dL_droughness_ptr,
        float3* __restrict__ dL_dradiance_ptr
    ){
        const int pid = blockIdx.x * blockDim.x + threadIdx.x;
        const int lid = blockIdx.y * blockDim.y + threadIdx.y;

        if(pid >= P || lid >= L)
            return;

        int idx = pid * L + lid;
        
        float3 normal = normal_ptr[pid];
        float3 albedo = albedo_ptr[pid];
        float metallic = metallic_ptr[pid];
        float roughness = roughness_ptr[pid];

        float3 ldir = inray_ptr[pid * L + lid];
        float3 radiance = radiance_ptr[pid * L + lid];
        float inarea = inarea_ptr[pid * L + lid];
        float3 vdir = outray_ptr[pid];

        /// dots
        float3 half = normalize(ldir + vdir);
        float h_dot_n = dotf(half, normal);
        float n_dot_l = dotf(normal, ldir);
        float n_dot_v = dotf(normal, vdir);
        float h_dot_v = dotf(half, vdir);

        // clamped
        bool is_hdn_clamped = h_dot_n < 0.0f;
        bool is_ndl_clamped = n_dot_l < 0.0f;
        bool is_ndv_clamped = n_dot_v < 0.0f;
        bool is_hdv_clamped = h_dot_v < 0.0f;

        h_dot_n = is_hdn_clamped ? 0.0f : h_dot_n;
        n_dot_l = is_ndl_clamped ? 0.0f : n_dot_l;
        n_dot_v = is_ndv_clamped ? 0.0f : n_dot_v;
        h_dot_v = is_hdv_clamped ? 0.0f : h_dot_v;

        // Fresnel term: active microgeometry normal h must be substituted for the surface normal n:
        float3 F0 = 0.04f * (1.0f - metallic) + albedo * metallic;
        float3 F = F0 + (1.0f - F0) * powf(1.0f - h_dot_v, 5.0f);

        // normal distribution term (GGX) 
        float a = roughness * roughness;
        float a2 = powf(a, 2.0f);
        float denom = powf(h_dot_n, 2.0f) * (a2 - 1.0f) + 1.0f;
        float tmp = PBR_PI * powf(denom, 2.0f) + PBR_EPS;
        float D = a2 / tmp;

        // Geometry Function (Smith)
        float ggx1 = geometry_schlick_ggx(n_dot_v, roughness);
        float ggx2 = geometry_schlick_ggx(n_dot_l, roughness);
        float G = ggx1 * ggx2;

        // Specular term
        float norm = 4.0f * n_dot_v * n_dot_l + PBR_EPS;
        float3 specular = F * D * G / norm;

        // diffuse component kd
        float3 kd = (1.0f - F) * (1.0f - metallic);

        // Lambertian diffuse
        float3 diffuse = kd * albedo / PBR_PI;

        // transport
        float3 transport = radiance * inarea * n_dot_l;

        // =============================================Back propagation====================================
        // load gradient from global memory
        float3 dL_dout = dL_drgb_ptr[idx];

        // (diffuse + specular) * transport
        float3 dL_ddiffuse = dL_dout * transport;
        float3 dL_dspecular = dL_dout * transport;
        float3 dL_dtransport = dL_dout * (diffuse + specular);

        // transport = radiance * inarea * n_dot_l
        float3 dL_dradiance = dL_dtransport * inarea * n_dot_l;
        float dL_dndl = is_ndl_clamped ? 0.0f: sum(dL_dtransport * radiance * inarea * vdir);

        // diffuse = kd * albedo / PBR_PI;
        float3 dL_dkd = dL_ddiffuse * albedo / PBR_PI;
        float3 dL_dalbedo = dL_ddiffuse * kd / PBR_PI;

        // kd = (1.0f - F) * (1.0f - metallic);
        float3 dL_dF = dL_dkd * (metallic - 1.0f);
        float dL_dmetallic = sum(dL_dkd * (F - 1.0f));

        // specular = F * D * G / norm
        // norm = 4.0f * n_dot_v * n_dot_l + PBR_EPS;
        dL_dF += dL_dspecular * D * G / norm;
        float dL_dD = sum(dL_dspecular * F * G / norm);
        float dL_dG = sum(dL_dspecular * F * D / norm);
        float dL_dndv = is_ndv_clamped ? 0.0f: sum(- dL_dspecular * 4.0f * F * D * G * n_dot_l / (norm * norm)); // √
        dL_dndl += is_ndl_clamped ? 0.0f: sum(- dL_dspecular * 4.0f * F * D * G * n_dot_v / (norm * norm)); // √

        // G = ggx1 * ggx2
        float dL_dggx1 = dL_dG * ggx2;
        float dL_dggx2 = dL_dG * ggx1;

        // ggx1 = geometry_schlick_ggx(n_dot_v, roughness);
        // ggx2 = geometry_schlick_ggx(n_dot_l, roughness);
        float dL_droughness = 0.0f;
        grad_geometry_schlick_ggx(n_dot_v, roughness, dL_dggx1, is_ndv_clamped, dL_dndv, dL_droughness);
        grad_geometry_schlick_ggx(n_dot_l, roughness, dL_dggx2, is_ndl_clamped, dL_dndl, dL_droughness);

        // D = a2 / (PBR_PI * powf(denom, 2.0f) + PBR_EPS);
        // tmp = PBR_PI * powf(denom, 2.0f) + PBR_EPS
        float dL_da2 = dL_dD / tmp;
        float dL_ddenom = - dL_dD * a2 / (tmp * tmp) * 2.0f * PBR_PI * denom;

        //  denom = powf(h_dot_n, 2.0f) * (a2 - 1.0f) + 1.0f
        dL_da2 += dL_ddenom * powf(h_dot_n, 2.0f);
        float dL_dhdn = is_hdn_clamped ? 0.0f: dL_ddenom * 2.0f * h_dot_n * (a2 - 1.0f);
        
        // a2 = roughness ^ 4
        dL_droughness += dL_da2 * 4.0f * powf(roughness, 3.0f); 

        // F = F0 + (1.0f - F0) * powf(1.0f - h_dot_v, 5.0f);
        float3 dL_dF0 = dL_dF * (1.0f - powf(1.0f - h_dot_v, 5.0f));

        // F0 = 0.04f * (1.0f - metallic) + albedo * metallic;
        dL_dmetallic += sum(dL_dF0 * (albedo - 0.04f));
        dL_dalbedo += dL_dF0 * metallic;

        // n_dot_v = dotf(normal, vdir);
        float3 dL_dnormal = dL_dndv * vdir;
        // n_dot_l = dotf(normal, ldir);
        dL_dnormal += dL_dndl * ldir;
        // h_dot_n = dotf(half, normal);
        dL_dnormal += dL_dhdn * half;
        // no gradient for vdir, ldir and thus no for half vector
        // h_dot_v = dotf(half, vdir); 
        // half = normalize(ldir + vdir);

        // return
        dL_dnormal_ptr[idx] = dL_dnormal;
        dL_dalbedo_ptr[idx] = dL_dalbedo;
        dL_dmetallic_ptr[idx] = dL_dmetallic;
        dL_droughness_ptr[idx] = dL_droughness;
        dL_dradiance_ptr[idx] = dL_dradiance;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    RenderingEquationBackward(
        const torch::Tensor& normals,
        const torch::Tensor& albedo,
        const torch::Tensor& metallic,
        const torch::Tensor& roughness,
        const torch::Tensor& inray,
        const torch::Tensor& radiance,
        const torch::Tensor& inarea,
        const torch::Tensor& outray,
        const torch::Tensor& dL_drgb
    ){
        const int P = normals.size(0);
        const int L = inray.size(1);

        // config kernel 
        const int thread_l = min((L + 7) / 8 * 8, PBR_MAX_RAY);
        const int thread_p = PBR_MAX_RAY / thread_l;

        dim3 blocks((P + thread_p - 1) / thread_p, (L + thread_l - 1) / thread_l, 1);
        dim3 threads(thread_p, thread_l, 1);

        auto float_opts = normals.options().dtype(torch::kFloat32);

        torch::Tensor dL_dnormals = torch::empty({P, L, 3}, float_opts);
        torch::Tensor dL_dalbedo = torch::empty({P, L, 3}, float_opts);
        torch::Tensor dL_dmetallic = torch::empty({P, L, 1}, float_opts);
        torch::Tensor dL_droughness = torch::empty({P, L, 1}, float_opts);
        torch::Tensor dL_dradiance = torch::empty({P, L, 3}, float_opts);

        RenderingEquationBackwardKernel<<<blocks, threads>>>(
            P, L,
            (float3*)normals.contiguous().data_ptr<float>(),
            (float3*)albedo.contiguous().data_ptr<float>(),
            metallic.contiguous().data_ptr<float>(),
            roughness.contiguous().data_ptr<float>(),
            (float3*)inray.contiguous().data_ptr<float>(),
            (float3*)radiance.contiguous().data_ptr<float>(),
            inarea.contiguous().data_ptr<float>(),
            (float3*)outray.contiguous().data_ptr<float>(),
            (float3*)dL_drgb.contiguous().data_ptr<float>(),
            (float3*)dL_dnormals.contiguous().data_ptr<float>(),
            (float3*)dL_dalbedo.contiguous().data_ptr<float>(),
            dL_dmetallic.contiguous().data_ptr<float>(),
            dL_droughness.contiguous().data_ptr<float>(),
            (float3*)dL_dradiance.contiguous().data_ptr<float>()
        );
        
        return std::make_tuple(dL_dnormals, dL_dalbedo, dL_dmetallic, dL_droughness, dL_dradiance);
    }
}