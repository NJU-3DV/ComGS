
import math
import torch
from torch import nn
from torch import Tensor
from jaxtyping import Float, jaxtyped
from typeguard import typechecked as typechecker
import frnn
import fpsample

from scene.gaussian_model import GaussianModel
from gaussian_renderer.trace import Tracer

@torch.no_grad()
@jaxtyped
@typechecker
def create_octahedral_direction_map(
    resolution: int, device: str = "cuda"
) -> Float[Tensor, "H W 3"]:
    """
    Create a mapping from octahedral UV coordinates to 3D directions.

    Parameters
    ----------
    resolution : int
        Resolution of the octahedral map, resulting in HxW texture.
    device : str
        Device to create tensor on, by default "cuda".

    Returns
    -------
    Float[Tensor, "H W 3"]
        Direction vectors for each UV coordinate, normalized to unit length.
        Shape is [H, W, 3] where H=W=resolution+2.
    """
    w = resolution
    v, u = torch.meshgrid(
        [torch.linspace(0.5, w - 0.5, w), torch.linspace(0.5, w - 0.5, w)],
        indexing="ij",
    )
    v, u = v.to(device), u.to(device)

    x = 2.0 * u / w - 1.0
    y = 2.0 * v / w - 1.0
    z = 1.0 - x.abs() - y.abs()

    up_part = z > 0
    x_down = (1.0 - y[~up_part].abs()) * x[~up_part].sgn()
    y_down = (1.0 - x[~up_part].abs()) * y[~up_part].sgn()
    x[~up_part] = x_down
    y[~up_part] = y_down

    vdir = torch.stack([x, y, z], dim=-1)
    vdir = torch.nn.functional.normalize(vdir, dim=-1)

    # add margin for correct bilinear interpolation
    l = w // 2 - 1
    vdir_margin = torch.empty([w + 2, w + 2, 3], dtype=torch.float32, device=vdir.device)

    vdir_margin[0, 0] = vdir[-1, -1]
    vdir_margin[-1, 0] = vdir[0, -1]
    vdir_margin[0, -1] = vdir[-1, 0]
    vdir_margin[-1, -1] = vdir[0, 0]

    vdir_margin[0, 1:l+2] = vdir[0, l+1:].flip(dims=[0])
    vdir_margin[0, l+2:-1] = vdir[0, 0:l+1].flip(dims=[0])
    vdir_margin[-1, 1:l+2] = vdir[-1, l+1:].flip(dims=[0])
    vdir_margin[-1, l+2:-1] = vdir[-1, 0:l+1].flip(dims=[0])
    
    vdir_margin[1:l+2, 0] = vdir[l+1:, 0].flip(dims=[0])
    vdir_margin[l+2:-1, 0] = vdir[0:l+1, 0].flip(dims=[0])
    vdir_margin[1:l+2, -1] = vdir[l+1:, -1].flip(dims=[0])
    vdir_margin[l+2:-1, -1] = vdir[0:l+1, -1].flip(dims=[0])

    vdir_margin[1:-1, 1:-1] = vdir

    return vdir_margin


def get_voxel_size(points, lower_quantile = 0.05, upper_quantile = 0.95, num_cell=100):

    # Shape [3], lower quantile for x, y, z
    min_coords = torch.quantile(points, lower_quantile, dim=0) 
    # Shape [3], upper quantile for x, y, z
    max_coords = torch.quantile(points, upper_quantile, dim=0)
    
    # Compute side lengths of the bounding box
    side_lengths = max_coords - min_coords
    voxel_size = torch.max(side_lengths).item() / num_cell

    return voxel_size


def farthest_point_ampling(points, rgb, normals, num_points):

    points_np = points.cpu().numpy()
    rgb_np = rgb.cpu().numpy()
    normals_np = normals.cpu().numpy()

    fps_samples_idx = fpsample.bucket_fps_kdline_sampling(points_np, num_points, h=9)

    points_np = points_np[fps_samples_idx]
    rgb_np = rgb_np[fps_samples_idx]
    normals_np = normals_np[fps_samples_idx]

    points = torch.tensor(points_np, dtype=torch.float32)
    rgb = torch.tensor(rgb_np, dtype=torch.float32)
    normals = torch.tensor(normals_np, dtype=torch.float32)

    return points, rgb, normals


class SurfaceOctahedralProbes:

    def __init__(self, num_probes=35000, resolution=32, device="cuda"):

        self.num_probes = num_probes
        self.resolution = resolution
        self.device = device

        # probe infos
        self.position = torch.empty(0)
        self.voxel_size = 0.0
        self.grid = None
        self.pos_min = torch.empty(0)
        self.pos_max = torch.empty(0)   

        # probe textures
        self.tex_radiance = torch.empty(0)
        self.tex_alpha = torch.empty(0)

        # trace textures
        self.trace_radiance = torch.empty(0)
        self.trace_alpha = torch.empty(0)

        # octahedral template
        self.uv_xyz = create_octahedral_direction_map(resolution=self.resolution, device=self.device)

        # optimizer
        self.optimizer = None

        self.reset_query_pose()

    def reset_query_pose(self, device=None):
        pose_device = device
        if pose_device is None:
            pose_device = self.position.device if self.position.numel() > 0 else self.device
        self.query_rotation = torch.eye(3, device=pose_device, dtype=torch.float32)
        self.query_translation = torch.zeros(3, device=pose_device, dtype=torch.float32)
        self.query_scale = torch.tensor(1.0, device=pose_device, dtype=torch.float32)

    def set_query_pose(self, transform=None, rotation=None, translation=None, scale=None):
        if transform is not None:
            linear = transform[:3, :3]
            scales = torch.linalg.norm(linear, dim=1)
            scale = scales.mean()
            rotation = linear / scale.clamp_min(1e-8)
            translation = transform[:3, 3]

        pose_device = self.position.device if self.position.numel() > 0 else self.device
        self.query_rotation = (
            rotation if rotation is not None else torch.eye(3, device=pose_device, dtype=torch.float32)
        ).to(device=pose_device, dtype=torch.float32)
        self.query_translation = (
            translation if translation is not None else torch.zeros(3, device=pose_device, dtype=torch.float32)
        ).to(device=pose_device, dtype=torch.float32)
        self.query_scale = torch.as_tensor(
            1.0 if scale is None else scale, device=pose_device, dtype=torch.float32
        )

    def transform(self, transform):
        # Backward-compatible wrapper. Query pose is applied lazily during sampling.
        self.set_query_pose(transform=transform)

    @property
    def tsize(self):
        # texture size = texture width = texture height = self.resolution + 2
        # we add 2 pixels to the texture to ensure correct bilinear interpolation
        # at the edges of the texture
        return self.resolution + 2
    
    def training_setup(self, training_args):

        self.tex_radiance = nn.Parameter(self.tex_radiance).requires_grad_(True)
        self.tex_alpha = nn.Parameter(self.tex_alpha).requires_grad_(True)

        l = [{'params': [self.tex_radiance], 'lr': training_args.sops_radiance_lr, "name": "sops_radiance"},
             {'params': [self.tex_alpha], 'lr': training_args.sops_alpha_lr, "name": "sops_alpha_lr"}]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def is_empty(self):
        return self.position.numel() == 0
    
    def capture(self):

        return (
            self.position, 
            self.voxel_size,
            self.tex_radiance, 
            self.tex_alpha,
            self.trace_radiance,
            self.trace_alpha,
            self.optimizer.state_dict()
        )

    def restore(self, model_args):
        
        (
            self.position, 
            self.voxel_size,
            self.tex_radiance,
            self.tex_alpha,
            self.trace_radiance,
            self.trace_alpha,
            opt_dict
        ) = model_args

        if self.optimizer is not None:
            self.optimizer.load_state_dict(opt_dict)
        
        self.pos_min = torch.min(self.position, dim=0)[0]
        self.pos_max = torch.max(self.position, dim=0)[0]
        self.grid = None
        self.reset_query_pose(device=self.position.device)
    
    def build(self, xyz, rgb, normal, rebuttal_scale=2):

        # get voxle size
        # cell_num = math.sqrt(self.num_probes)
        self.voxel_size = get_voxel_size(xyz)
        print("Voxel size: ", self.voxel_size)

        # get SOPs' locations
        sops_xyz, _, sops_normal = farthest_point_ampling(xyz, rgb, normal, self.num_probes)
        # sops_xyz = sops_xyz + 2 * self.voxel_size * sops_normal
        sops_xyz = sops_xyz + rebuttal_scale * self.voxel_size * sops_normal
        print("Rebuttal scale: ", rebuttal_scale)

        self.position = sops_xyz.to(self.device)

        self.pos_min = torch.min(self.position, dim=0)[0]
        self.pos_max = torch.max(self.position, dim=0)[0]
        self.grid = None
        self.reset_query_pose(device=self.position.device)

        # set zeros for tex_radiance and tex_alpha
        self.tex_radiance = torch.zeros(self.num_probes, self.tsize, self.tsize, 3, device=self.device)   # [N_probes, res, res, 3]
        self.tex_alpha = torch.zeros(self.num_probes, self.tsize, self.tsize, 1, device=self.device)    

        torch.cuda.empty_cache()
    
    def initialize_textures_by_tracing(self, gaussian: GaussianModel):
        with torch.no_grad():
            tracer = Tracer(gaussian)

            ray_o = self.position[:, None, None, :].repeat(1, self.tsize, self.tsize, 1)  # [N_probes, tsize, tsize, 3]
            ray_d = self.uv_xyz[None].repeat(ray_o.shape[0], 1, 1, 1) 
            trace_pkg = tracer.trace(ray_o + self.voxel_size * ray_d, ray_d)

            self.trace_radiance = trace_pkg["render"]  # [N_probes, tsize, tsize, 3]
            self.trace_alpha = trace_pkg["alpha"]  # [N_probes, tsize, tsize, 1]

            del tracer
        
        self.tex_radiance = self.trace_radiance.clone()
        self.tex_alpha = self.trace_alpha.clone()
    
    @torch.no_grad()
    def vis(self, xyz=None, rgb=None, probes_idxs=None):

        import pyvista as pv

        plotter = pv.Plotter(title=f"Locations of Surface Octahedral Probes (x{self.num_probes})")

        sphere = pv.Sphere(radius = self.voxel_size * 0.5)
        positions = self.position.cpu().numpy()

        print(positions.shape)

        # activated probes
        probes_pv = pv.PolyData(positions)
        glyphs = probes_pv.glyph(geom=sphere)
        # plotter.add_mesh(glyphs, color="#9bb9de", opacity=1)
        plotter.add_mesh(glyphs, color="#dcdcdc", opacity=1)

        # show points
        if xyz is not None:
            point_cloud_pv = pv.PolyData(xyz.cpu().detach().numpy())
            if rgb is not None:
                point_cloud_pv['RGB'] = rgb.cpu().detach().numpy()
                plotter.add_mesh(point_cloud_pv, scalars='RGB', rgb=True, point_size=10, opacity=1)
            else:
                plotter.add_mesh(point_cloud_pv, color='#dcdcdc', point_size=10, opacity=1) 
        
        if probes_idxs is not None:
            show_idx = probes_idxs.detach().cpu().numpy()
            show_position = positions[show_idx]
            probes_pv = pv.PolyData(show_position)

            sphere = pv.Sphere(radius = self.voxel_size * 0.5)
            glyphs = probes_pv.glyph(geom=sphere)
            plotter.add_mesh(glyphs, color='green', opacity=1)

        plotter.show_axes()
        plotter.camera.fov = 90 # set a large fov to see more probes
        plotter.show()

    def world_to_local_positions(self, xyz: Float[Tensor, "... 3"]) -> Float[Tensor, "... 3"]:
        if xyz.numel() == 0:
            return xyz
        xyz = xyz.to(device=self.query_rotation.device, dtype=torch.float32)
        return ((xyz - self.query_translation) / self.query_scale.clamp_min(1e-8)) @ self.query_rotation

    def world_to_local_normals(self, normals: Float[Tensor, "... 3"]) -> Float[Tensor, "... 3"]:
        if normals.numel() == 0:
            return normals
        normals = normals.to(device=self.query_rotation.device, dtype=torch.float32)
        normals = normals @ self.query_rotation
        return torch.nn.functional.normalize(normals, dim=-1)

    def world_to_local_dirs(self, rays: Float[Tensor, "... 3"]) -> Float[Tensor, "... 3"]:
        if rays.numel() == 0:
            return rays
        rays = rays.to(device=self.query_rotation.device, dtype=torch.float32)
        rays = rays @ self.query_rotation
        return torch.nn.functional.normalize(rays, dim=-1)

    def local_to_world_positions(self, xyz: Float[Tensor, "... 3"]) -> Float[Tensor, "... 3"]:
        if xyz.numel() == 0:
            return xyz
        xyz = xyz.to(device=self.query_rotation.device, dtype=torch.float32)
        return xyz @ self.query_rotation.T * self.query_scale + self.query_translation

    def get_bbox_world(self):
        if self.position.numel() == 0:
            return self.pos_min, self.pos_max

        corners = torch.tensor(
            [
                [self.pos_min[0], self.pos_min[1], self.pos_min[2]],
                [self.pos_min[0], self.pos_min[1], self.pos_max[2]],
                [self.pos_min[0], self.pos_max[1], self.pos_min[2]],
                [self.pos_min[0], self.pos_max[1], self.pos_max[2]],
                [self.pos_max[0], self.pos_min[1], self.pos_min[2]],
                [self.pos_max[0], self.pos_min[1], self.pos_max[2]],
                [self.pos_max[0], self.pos_max[1], self.pos_min[2]],
                [self.pos_max[0], self.pos_max[1], self.pos_max[2]],
            ],
            device=self.position.device,
            dtype=torch.float32,
        )
        world_corners = self.local_to_world_positions(corners)
        return torch.min(world_corners, dim=0)[0], torch.max(world_corners, dim=0)[0]
    
    def compute_wrap_weight(
        self,
        query_position: Float[Tensor, "P 3"],
        surface_normal: Float[Tensor, "P 3"],
        knn_probe_idx: Float[Tensor, "P K"]
    ) -> Float[Tensor, "P K"]:
        
        knn_probe_position = self.position[knn_probe_idx]  # [P. K, 3]

        direction = knn_probe_position - query_position.unsqueeze(1) # [P, K, 3]
        direction = torch.nn.functional.normalize(direction, dim=-1) # [P, K, 3]

        weight = torch.sum(surface_normal.unsqueeze(1) * direction, dim=-1)  # [P, K]
        weight = (weight + 1) * 0.5
        weight = torch.square(weight) + 0.01

        return weight
    
    def query_light_knn(self, xyz, normals, k=4):
        xyz_local = self.world_to_local_positions(xyz)
        normals_local = self.world_to_local_normals(normals)

        with torch.no_grad():
            dists, idxs, _, self.grid = frnn.frnn_grid_points(
                xyz_local[None], self.position[None], K=k, 
                r=self.voxel_size * 20, grid=self.grid, return_nn=False
            )

            dists = dists.squeeze(0) # [N, K]
            idxs = idxs.squeeze(0)   # [N, K]

            # Spatial Weight
            spatial_weight = 1.0 / (dists + 1e-8)
            spatial_weight = spatial_weight / torch.sum(spatial_weight, dim=1, keepdim=True) + 1e-8
            
            # Warp Weight
            # NOTE: Ablation on warp weight!!!!!!!!!!!!!!!!!!!!!!!!
            warp_weight = self.compute_wrap_weight(xyz_local, normals_local, idxs)

            # full weight
            weight = spatial_weight * warp_weight
            weight = weight / torch.sum(weight, dim=1, keepdim=True)
            # NOTE: Ablation on warp weight!!!!!!!!!!!!!!!!!!!!!!!!
            # weight = spatial_weight / torch.sum(spatial_weight, dim=1, keepdim=True) + 1e-8

        # self.tex_radiance[probe_idx] [N. K, res, res, 3] -> [N, res, res, 3]
        tex_radiance = torch.sum(self.tex_radiance[idxs] * weight[..., None, None, None], dim=1)
        # self.tex_alpha[probe_idx] [N. K, res, res, 1] -> [N, res, res, 1]
        tex_alpha = torch.sum(self.tex_alpha[idxs] * weight[..., None, None, None], dim=1)

        return torch.concat([tex_radiance, 1 - tex_alpha], dim=-1) # [N, res, res, 4]
    
    def query_visibility_nearst(self, xyz, normals):
        xyz_local = self.world_to_local_positions(xyz)

        with torch.no_grad():
            dists, idxs, _, self.grid = frnn.frnn_grid_points(
                xyz_local[None], self.position[None], K=1, 
                r=self.voxel_size * 20, grid=self.grid, return_nn=False
            )

            idxs = idxs.squeeze(0).squeeze(-1)   # [N, K]

        tex_alpha = self.tex_alpha[idxs]

        return 1 - tex_alpha # [N, res, res, 1]
    
    def query_visibility_knn(self, xyz, normals, k=4):
        xyz_local = self.world_to_local_positions(xyz)
        normals_local = self.world_to_local_normals(normals)

        with torch.no_grad():
            dists, idxs, _, self.grid = frnn.frnn_grid_points(
                xyz_local[None], self.position[None], K=k, 
                r=self.voxel_size * 20, grid=self.grid, return_nn=False
            )

            dists = dists.squeeze(0) # [N, K]
            idxs = idxs.squeeze(0)   # [N, K]

            # Spatial Weight
            spatial_weight = 1.0 / (dists + 1e-8)
            spatial_weight = spatial_weight / torch.sum(spatial_weight, dim=1, keepdim=True) + 1e-8

            # Warp Weight
            warp_weight = self.compute_wrap_weight(xyz_local, normals_local, idxs)

            # full weight
            weight = spatial_weight * warp_weight
            weight = weight / torch.sum(weight, dim=1, keepdim=True)

        tex_alpha = torch.sum(self.tex_alpha[idxs] * weight[..., None, None, None], dim=1)

        return 1 - tex_alpha # [N, res, res, 1]
    
def sample_from_octahedral_envmaps(envmap: torch.Tensor, rays: torch.Tensor):
    """
    Samples values from octahedral environment maps based on input rays.

    Parameters:
    - envmap: torch.Tensor, shape [B, H=res+2, W=res+2, C]
    - rays: torch.Tensor, shape [B, N, 3]

    Returns:
    - torch.Tensor, shape [B, N, C]
    """
    # Validate input shapes
    assert envmap.ndim == 4 and envmap.shape[1] == envmap.shape[2], \
        "envmap must have shape [B, res+2, res+2, C]"
    assert rays.ndim == 3 and rays.shape[-1] == 3, \
        "rays must have shape [B, N, 3]"

    # Rearrange envmap to [B, C, H, W]
    data = envmap.permute(0, 3, 1, 2)  # [B, C, H, W]
    tsize = data.shape[-1]
    res = tsize - 2

    with torch.no_grad():
        # Normalize rays
        rays = torch.nn.functional.normalize(rays, dim=-1)  # [B, N, 3]

        # Compute UV coordinates
        r = torch.nn.functional.normalize(rays, p=1, dim=-1)  # [B, N, 3]
        up_part = r[..., 2] > 0

        # Compute nu and nv efficiently
        nu = torch.where(up_part, r[..., 0], (1.0 - r[..., 1].abs()) * torch.sign(r[..., 0]))
        nv = torch.where(up_part, r[..., 1], (1.0 - r[..., 0].abs()) * torch.sign(r[..., 1]))

        # Convert [-1, 1] to texture UV
        u = (nu + 1) * 0.5 * res + 1
        v = (nv + 1) * 0.5 * res + 1

        # Convert texture UV to [-1, 1] for grid sampling
        sampled_x = u / tsize * 2.0 - 1.0
        sampled_y = v / tsize * 2.0 - 1.0

        # Create sampling grid
        grid = torch.stack((sampled_x, sampled_y), dim=-1).unsqueeze(-2)  # [B, N, 1, 2]

    # Perform grid sampling
    light = torch.nn.functional.grid_sample(data, grid, align_corners=False, mode="nearest")  # [B, C, N, 1]
    light = light.squeeze(-1)  # [B, C, N]

    return light.permute(0, 2, 1)  # [B, N, C]
