
import torch
import trimesh
from scene.gaussian_model import GaussianModel
from gtracer import GaussianTracer
from utils.general_utils import build_scaling_rotation


class Tracer:
    def __init__(self, gaussians: GaussianModel, transmittance_min: float = 0.001):

        self.gaussians = gaussians

        # icosahedron, outer sphere radius is 1.0
        icosahedron = trimesh.creation.icosahedron()
        icosahedron.vertices[..., -1] *= 1e-3

        # change to inner sphere radius equal to 1.0
        # the central point of each face must be on the unit sphere
        self.unit_icosahedron_vertices = torch.from_numpy(icosahedron.vertices).float().cuda() * 1.2584 
        self.unit_icosahedron_faces = torch.from_numpy(icosahedron.faces).long().cuda()

        self.tracer = GaussianTracer(transmittance_min=transmittance_min)
        self.alpha_min = 1 / 255

        self.build_bvh()

    def get_boundings(self, alpha_min=0.01):
        mu = self.gaussians.get_xyz
        opacity = self.gaussians.get_opacity
        
        L = build_scaling_rotation(self.gaussians.get_scaling, self.gaussians._rotation)
        vertices_b = (2 * (opacity/alpha_min).log()).sqrt()[:, None] * (self.unit_icosahedron_vertices[None] @ L.transpose(-1, -2)) + mu[:, None]
        faces_b = self.unit_icosahedron_faces[None] + torch.arange(mu.shape[0], device="cuda")[:, None, None] * 12
        gs_id = torch.arange(mu.shape[0], device="cuda")[:, None].expand(-1, faces_b.shape[1])

        return vertices_b.reshape(-1, 3), faces_b.reshape(-1, 3), gs_id.reshape(-1)
    
    def has_bvh(self):
        return hasattr(self.tracer, 'gs_idxs')
    
    def build_bvh(self):
        vertices_b, faces_b, gs_id = self.get_boundings(alpha_min=self.alpha_min)
        self.tracer.build_bvh(vertices_b, faces_b, gs_id)
        
    def update_bvh(self):
        vertices_b, faces_b, gs_id = self.get_boundings(alpha_min=self.alpha_min)
        self.tracer.update_bvh(vertices_b, faces_b, gs_id)
    
    def trace(self, rays_o, rays_d):
        means3D = self.gaussians.get_xyz.contiguous()
        scales = self.gaussians.get_scaling
        rotation = self.gaussians.get_rotation
        shs = self.gaussians.get_features
        opacity = self.gaussians.get_opacity
        deg = self.gaussians.active_sh_degree

        SinvR = build_scaling_rotation(1 / scales.clamp(min=1e-8), rotation)
        
        colors, depth, alpha = self.tracer.trace(
            rays_o, rays_d, means3D, opacity, SinvR, shs, 
            alpha_min=self.alpha_min, deg=deg
        )

        depth = depth / alpha
        depth = torch.nan_to_num(depth, 0, 0, 0)

        return {
            "render": colors,
            "depth": depth,
            "alpha" : alpha
        }


def trace_scene_360(gaussians: GaussianModel, ray_o, resolution=512):

    tracer = Tracer(gaussians)
    
    def get_rays_360(resolution=32, device="cuda"):
        h, w = resolution, resolution * 2
        v, u = torch.meshgrid([torch.linspace(0.5, h-0.5, h), torch.linspace(0.5, w-0.5, w)], indexing="ij")
        v = v.to(device)
        u = u.to(device)

        nu = u / w
        nv = v / h

        phi = nv * torch.pi
        theta = torch.pi - 2 * nu * torch.pi

        viewdirs = torch.stack([
            torch.sin(phi) * torch.cos(theta),
            torch.sin(phi) * torch.sin(theta),
            torch.cos(phi)], dim=-1)    # [H, W, 3]
        
        viewdirs = viewdirs.reshape(h, w, 3)

        return viewdirs
    
    ray_d = get_rays_360(resolution).reshape(-1, 3)
    ray_o = ray_o.unsqueeze(0).repeat(ray_d.shape[0], 1)

    trace_pkg = tracer.trace(ray_o, ray_d)
    rgb = trace_pkg["render"]
    depth = trace_pkg["depth"]
    alpha = trace_pkg["alpha"]

    image = rgb.reshape(resolution, resolution * 2, 3)
    depth = depth.reshape(resolution, resolution * 2)
    alpha = alpha.reshape(resolution, resolution * 2)

    del tracer

    return image, depth, alpha