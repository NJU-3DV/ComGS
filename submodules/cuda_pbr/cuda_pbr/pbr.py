


import torch
import torch.nn as nn
from . import _C
from torch import Tensor
from jaxtyping import Float, jaxtyped
from typeguard import typechecked as typechecker


@jaxtyped
@typechecker
def physically_based_rendering(
    normal: Float[Tensor, "P 3"],
    albedo: Float[Tensor, "P 3"],
    metallic: Float[Tensor, "P 1"],
    roughness: Float[Tensor, "P 1"],
    ldir: Float[Tensor, "P L 3"],
    radiance: Float[Tensor, "P L 3"],
    inarea: Float[Tensor, "P L 1"],
    vdir: Float[Tensor, "P 3"],
    cuda_backend: bool=False
)->Float[Tensor, "P 3"]:
    normal = torch.nn.functional.normalize(normal, dim=-1)
    ldir = torch.nn.functional.normalize(ldir, dim=-1)
    vdir = torch.nn.functional.normalize(vdir, dim=-1)

    if cuda_backend:
        rgb = _PhysicallyBasedRendering.apply(
            normal, albedo, metallic, roughness, ldir, radiance, inarea, vdir
        ) # [P, L, 3]

        return rgb.mean(dim=-2)

    else:
        return _physically_based_rendering_pytorch_impl(
            normal, albedo, metallic, roughness, ldir, radiance, inarea, vdir
        )
        # return rendering_equation(albedo, roughness, normal, vdir, radiance, ldir, inarea)

class _PhysicallyBasedRendering(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normal, albedo, metallic, roughness, ldir, radiance, inarea, vdir):

        rgb =_C.render_equation(
            normal, albedo, metallic, roughness, ldir, radiance, inarea, vdir
        )

        ctx.save_for_backward(normal, albedo, metallic, roughness, ldir, radiance, inarea, vdir)

        return rgb

    @staticmethod
    def backward(ctx, grad_output):

        normal, albedo, metallic, roughness, ldir, radiance, inarea, vdir = ctx.saved_tensors

        (
            grad_normal, 
            grad_albedo, 
            grad_metallic, 
            grad_roughness, 
            grad_radiance
        ) = _C.render_equation_backward(
            normal, 
            albedo, 
            metallic, 
            roughness, 
            ldir, 
            radiance, 
            inarea, 
            vdir, 
            grad_output
        )

        grad_normal = grad_normal.sum(-2)
        grad_albedo = grad_albedo.sum(-2)
        grad_metallic = grad_metallic.sum(-2)
        grad_roughness = grad_roughness.sum(-2)

        grads = (
            grad_normal,
            grad_albedo,
            grad_metallic,
            grad_roughness,
            None,
            grad_radiance,
            None,
            None
        )
        return grads


def _physically_based_rendering_pytorch_impl(normal, albedo, metallic, roughness, ldir, radiance, inarea, vdir):

    # the Cook-Torrance BRDF model.
    normal = normal.unsqueeze(1) # [P, 1 3]
    albedo = albedo.unsqueeze(1) # [P, 1, 3]
    metallic = metallic.unsqueeze(1) # [P, 1, 1]
    roughness = roughness.unsqueeze(1) # [P, 1, 1]
    # ldir [P, L, 3]
    # radiance [P, L, 3]
    # invpdf [P, L, 1]
    vdir = vdir.unsqueeze(1) # [P, 1, 3]
    
    # dots
    half = ldir + vdir
    norm = torch.norm(half, dim=-1, keepdim=True) + 1e-7
    half = half / norm # [P, L, 3]
    h_dot_n = (half * normal).sum(dim=-1, keepdim=True).clamp(0)  # [P, L, 1]
    h_dot_v = (half * vdir).sum(dim=-1, keepdim=True).clamp(0)  # [P, L, 1]
    n_dot_l = (normal * ldir).sum(dim=-1, keepdim=True).clamp(0)  # [P, L, 1]
    n_dot_v = (normal * vdir).sum(dim=-1, keepdim=True).clamp(0)  # [P, 1, 1]

    # Fresnel term: active microgeometry normal h must be substituted for the surface normal n:
    F0 = 0.04 * (1. - metallic) + albedo * metallic # [P, 1, 3]
    F = F0 + (1. - F0) * torch.pow(1.0 - h_dot_v, 5.0) # [P, L, 3]

    # # normal distribution term (GGX) 
    a = roughness * roughness
    a2 = a ** 2.
    denom = h_dot_n ** 2 * (a2 - 1.0) + 1.0
    D = a2 / (torch.pi * denom**2 + 1e-7) # [P, L, 1]

    # # Geometry Function (Smith)
    def geometry_schlick_ggx(n_dot_x, roughness):
        r = roughness + 1.0
        k = (r**2) / 8.0
        denom = (n_dot_x * (1.0 - k) + k + 1e-7)
        return n_dot_x / denom
    
    ggx1 = geometry_schlick_ggx(n_dot_v, roughness)
    ggx2 = geometry_schlick_ggx(n_dot_l, roughness)
    G = ggx1 * ggx2 # [P, L, 1]

    # Specular term [P, L, 3]
    specular = (F * D * G) / (4.0 * n_dot_v * n_dot_l + 1e-7)

    # diffuse component Kd [P, L, 3]
    kd = (1.0 - F) * (1.0 - metallic)

    # Lambertian diffuse [P, 1, 3]
    diffuse = kd * albedo / torch.pi

    # transport [P, L, 3]
    transport = radiance * inarea * n_dot_l

    rgb = (diffuse + specular) * transport
    # rgb = diffuse * transport
    rgb = rgb.mean(dim=-2)
    
    return rgb


def rotation_between_z(vec):
    """
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    Args:
        vec: [..., 3]

    Returns:
        R: [..., 3, 3]

    """
    v1 = -vec[..., 1]
    v2 = vec[..., 0]
    v3 = torch.zeros_like(v1)
    v11 = v1 * v1
    v22 = v2 * v2
    v33 = v3 * v3
    v12 = v1 * v2
    v13 = v1 * v3
    v23 = v2 * v3
    cos_p_1 = (vec[..., 2] + 1).clamp_min(1e-7)
    R = torch.zeros(vec.shape[:-1] + (3, 3,), dtype=torch.float32, device="cuda")
    R[..., 0, 0] = 1 + (-v33 - v22) / cos_p_1
    R[..., 0, 1] = -v3 + v12 / cos_p_1
    R[..., 0, 2] = v2 + v13 / cos_p_1
    R[..., 1, 0] = v3 + v12 / cos_p_1
    R[..., 1, 1] = 1 + (-v33 - v11) / cos_p_1
    R[..., 1, 2] = -v1 + v23 / cos_p_1
    R[..., 2, 0] = -v2 + v13 / cos_p_1
    R[..., 2, 1] = v1 + v23 / cos_p_1
    R[..., 2, 2] = 1 + (-v22 - v11) / cos_p_1
    R = torch.where((vec[..., 2] + 1 > 0)[..., None, None], R,
                    -torch.eye(3, dtype=torch.float32, device="cuda").expand_as(R))
    return R

def fibonacci_sphere_sampling(normals, sample_num, random_rotate=True):
    import torch.nn.functional as F

    pre_shape = normals.shape[:-1]
    if len(pre_shape) > 1:
        normals = normals.reshape(-1, 3)
    delta = np.pi * (3.0 - np.sqrt(5.0))

    # fibonacci sphere sample around z axis
    idx = torch.arange(sample_num, dtype=torch.float, device='cuda')[None]
    z = (1 - 2 * idx / (2 * sample_num - 1)).clamp_min(np.sin(10/180*np.pi))
    rad = torch.sqrt(1 - z ** 2)
    theta = delta * idx
    if random_rotate:
        theta = torch.rand(*pre_shape, 1, device='cuda') * 2 * np.pi + theta
    y = torch.cos(theta) * rad
    x = torch.sin(theta) * rad
    z_samples = torch.stack([x, y, z.expand_as(y)], dim=-2)

    # rotate to normal
    # z_vector = torch.zeros_like(normals)
    # z_vector[..., 2] = 1  # [H, W, 3]
    # rotation_matrix = rotation_between_vectors(z_vector, normals)
    rotation_matrix = rotation_between_z(normals)
    incident_dirs = rotation_matrix @ z_samples
    incident_dirs = F.normalize(incident_dirs, dim=-2).transpose(-1, -2)
    incident_areas = torch.ones_like(incident_dirs)[..., 0:1] * 2 * np.pi
    if len(pre_shape) > 1:
        incident_dirs = incident_dirs.reshape(*pre_shape, sample_num, 3)
        incident_areas = incident_areas.reshape(*pre_shape, sample_num, 1)
    return incident_dirs, incident_areas

def sample_incident_rays(normals, is_training=False, sample_num=24):
    if is_training:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=True)
    else:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=False)

    return incident_dirs, incident_areas  # [N, S, 3], [N, S, 1]


def rendering_equation(
    base_color, # [N, 3]
    roughness, # [N, 1]
    normals,  # [N, 3]
    viewdirs, # [N, 3]
    incident_lights,  # [N, L, 3]
    incident_dirs, # [N, L, 3]
    incident_areas # [N, L, 1]
):
    def GGX_specular(normal, pts2c, pts2l, roughness, fresnel):
        
        import torch.nn.functional as F

        L = F.normalize(pts2l, dim=-1)  # [nrays, nlights, 3]
        V = F.normalize(pts2c, dim=-1)  # [nrays, 3]
        H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)  # [nrays, nlights, 3]
        N = F.normalize(normal, dim=-1)  # [nrays, 3]

        NoV = torch.sum(V * N, dim=-1, keepdim=True)  # [nrays, 1]
        N = N * NoV.sign()  # [nrays, 3]

        NoL = torch.sum(N[:, None, :] * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
        NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, 1]
        NoH = torch.sum(N[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
        VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]

        alpha = roughness * roughness  # [nrays, 3]
        alpha2 = alpha * alpha  # [nrays, 3]
        k = (alpha + 2 * roughness + 1.0) / 8.0
        FMi = ((-5.55473) * VoH - 6.98316) * VoH
        frac0 = fresnel + (1 - fresnel) * torch.pow(2.0, FMi)  # [nrays, nlights, 3]
        
        frac = frac0 * alpha2[:, None, :]  # [nrays, 1]
        nom0 = NoH * NoH * (alpha2[:, None, :] - 1) + 1

        nom1 = NoV * (1 - k) + k
        nom2 = NoL * (1 - k[:, None, :]) + k[:, None, :]
        nom = (4 * torch.pi * nom0 * nom0 * nom1[:, None, :] * nom2).clamp_(1e-6, 4 * torch.pi)
        spec = frac / nom
        return spec

    n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp(min=0)
    f_d = base_color[:, None] / torch.pi
    f_s = GGX_specular(normals, viewdirs, incident_dirs, roughness, fresnel=0.04)

    transport = incident_lights * incident_areas * n_d_i  # （num_pts, num_sample, 3)
    pbr = ((f_d + f_s) * transport).mean(dim=-2)

    return pbr
