
import torch
from . import _C
from torch import Tensor
from typing import Tuple
from jaxtyping import Float, jaxtyped
from typeguard import typechecked as typechecker


@jaxtyped
@typechecker
def uniform_sample_hemisphere(
    normal: Float[Tensor, "P 3"],  
    num_ray: int,
    cuda_backend: bool=False,
    random_rotate: bool=False
)->Tuple[Float[Tensor, "P N 3"], 
         Float[Tensor, "P N 1"]]:
    
    if cuda_backend:
        return _C.uniform_sample(normal, num_ray, 0.0, 1.0, random_rotate)
    else:
        return uniform_sample_hemisphere_pytorch_impl(normal, num_ray)

@jaxtyped
@typechecker
def uniform_sample_sphere(
    normal: Float[Tensor, "P 3"],
    num_ray: int,
    cos_theta_min: float=0.0,
    cos_theta_max: float=1.0,
    cuda_backend: bool=False
)->Tuple[Float[Tensor, "P N 3"], 
         Float[Tensor, "P N 1"]]:
    
    if cuda_backend:
        return _C.uniform_sample(normal, num_ray, cos_theta_min, cos_theta_max)
    else:
        return uniform_sample_pytorch_impl(normal, num_ray, cos_theta_min, cos_theta_max)

@jaxtyped
@typechecker
def uniform_random(
    num_samples: int, 
    device: torch.device = "cuda"
):
    # uniform random in [0, 1]
    return torch.rand(num_samples * 2, device=device)

@jaxtyped
@typechecker
def sobel_sequence(
    num_samples: int, 
    device: torch.device = "cuda"
):
    soboleng = torch.quasirandom.SobolEngine(dimension=2)
    return soboleng.draw(num_samples).to(device)

@torch.no_grad()
@jaxtyped
@typechecker
def rotation_between_vectors(
    vec1: Float[Tensor, "P 3"], 
    vec2: Float[Tensor, "P 3"]
)-> Float[Tensor, "P 3 3"]:
    ''' Retruns rotation matrix between two vectors (for Tensor object) '''
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    batch_size = vec1.shape[0]
    
    v = torch.cross(vec1, vec2)                                                     # [P, 3, 3]

    cos = torch.bmm(vec1.view(batch_size, 1, 3), vec2.view(batch_size, 3, 1))
    cos = cos.reshape(batch_size, 1, 1).repeat(1, 3, 3)                             # [P, 3, 3]
    
    skew_sym_mat = torch.zeros(batch_size, 3, 3).to(vec1.device)
    skew_sym_mat[:, 0, 1] = -v[:, 2]
    skew_sym_mat[:, 0, 2] = v[:, 1]
    skew_sym_mat[:, 1, 0] = v[:, 2]
    skew_sym_mat[:, 1, 2] = -v[:, 0]
    skew_sym_mat[:, 2, 0] = -v[:, 1]
    skew_sym_mat[:, 2, 1] = v[:, 0]

    identity_mat = torch.eye(3, device=vec1.device).expand(batch_size, -1, -1)  

    R = identity_mat + skew_sym_mat
    R = R + torch.bmm(skew_sym_mat, skew_sym_mat) / (1 + cos).clamp(min=1e-7)
    zero_cos_loc = (cos == -1).float()
    R_inverse = -identity_mat
    R_out = R * (1 - zero_cos_loc) + R_inverse * zero_cos_loc                       # [P, 3, 3]

    return R_out        

@torch.no_grad()
@jaxtyped
@typechecker
def uniform_sample_pytorch_impl(
    normal: Float[Tensor, "P 3"],
    num_ray: int,
    cos_theta_min: float=0.0,
    cos_theta_max: float=1.0
)->Tuple[Float[Tensor, "P N 3"], 
         Float[Tensor, "P N 1"]]:
    
    P = normal.shape[0]
    ur = sobel_sequence(P * num_ray, device=normal.device)
    ur = ur.reshape(P, num_ray, 2)

    # get Phi and Theta based on cdf^{-1}
    phi = 2 * torch.pi * ur[..., 0]
    cos_theta = cos_theta_min + (cos_theta_max - cos_theta_min) * ur[..., 1]
    sin_theta = torch.sqrt(1 - cos_theta ** 2)

    # get sampled direction [P, N, 3]
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = cos_theta
    ray = torch.stack([x, y, z], dim=-1)

    # rotate z axis to normal
    z_aixs = torch.zeros_like(normal)
    z_aixs[:, 2] = 1
    normal = torch.nn.functional.normalize(normal, dim=-1)
    
    rot_mat = rotation_between_vectors(z_aixs, normal) # [P, 3, 3]
    # [P, 1, 3, 3] * [P, N, 1, 3], sum in dim=-1
    ray = torch.sum(rot_mat.unsqueeze(1) * ray.unsqueeze(-2), dim=-1)
    ray = torch.nn.functional.normalize(ray, dim=-1)

    # get invpdf
    invpdf = torch.ones_like(ray)[..., 0:1] * 2 * torch.pi
    
    return ray, invpdf

@torch.no_grad()
@jaxtyped
@typechecker
def uniform_sample_hemisphere_pytorch_impl(
    normal: Float[Tensor, "P 3"],
    num_ray: int
)->Tuple[Float[Tensor, "P N 3"], 
         Float[Tensor, "P N 1"]]:
    
    return uniform_sample_pytorch_impl(normal, num_ray, cos_theta_min=0.0, cos_theta_max=1.0)

@torch.no_grad()
@jaxtyped
@typechecker
def fibonacci_sampling_hemisphere(
    normal: Float[Tensor, "P 3"], 
    num_ray: int, 
    random_rotate: bool=True
):
    
    P = normal.shape[0]
    delta = 0.7639320225 * torch.pi

    # fibonacci sphere sample around z axis 
    # idx: [P, N]
    idx = torch.arange(num_ray, device=normal.device).float().unsqueeze(0).repeat([P, 1])
    z = 1 - 2 * idx / (2 * num_ray - 1)

    # x and y axis
    rad = torch.sqrt(1 - z ** 2)
    theta = delta * idx

    if random_rotate:
        theta = torch.rand(P, 1, device=normal.device) * 2 * torch.pi + theta
    
    y = torch.cos(theta) * rad
    x = torch.sin(theta) * rad 

    ray = torch.stack([x, y, z], dim=-1)

    # rotate z axis to normal
    z_aixs = torch.zeros_like(normal)
    z_aixs[:, 2] = 1
    normal = torch.nn.functional.normalize(normal, dim=-1)

    rot_mat = rotation_between_vectors(z_aixs, normal) # [P, 3, 3]
    # [P, 1, 3, 3] * [P, N, 1, 3], sum in dim=-1
    ray = torch.sum(rot_mat.unsqueeze(1) * ray.unsqueeze(-2), dim=-1)
    ray = torch.nn.functional.normalize(ray, dim=-1)

    # get pdf
    invpdf = torch.ones_like(ray)[..., 0:1] * 2 * torch.pi

    return ray, invpdf


@torch.no_grad()
@jaxtyped
@typechecker
def cosine_sample_hemisphere(
    normal: Float[Tensor, "P 3"],
    num_ray: int
)->Tuple[Float[Tensor, "P N 3"], 
         Float[Tensor, "P N 1"]]:
    
    P = normal.shape[0]
    # ur = sobel_sequence(P * num_ray, device=normal.device)
    ur = uniform_random(P * num_ray, device=normal.device)
    ur = ur.reshape(P, num_ray, 2)

    # get Phi and Theta based on cdf^{-1}
    phi = 2 * torch.pi * ur[..., 0]
    cos_theta = torch.sqrt(ur[..., 1])
    sin_theta = torch.sqrt(1 - cos_theta ** 2)

    # get sampled direction [P, N, 3]
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = cos_theta
    ray = torch.stack([x, y, z], dim=-1)

    # rotate z axis to normal
    z_aixs = torch.zeros_like(normal)
    z_aixs[:, 2] = 1
    normal = torch.nn.functional.normalize(normal, dim=-1)

    rot_mat = rotation_between_vectors(z_aixs, normal) # [P, 3, 3]
    # [P, 1, 3, 3] * [P, N, 1, 3], sum in dim=-1
    ray = torch.sum(rot_mat.unsqueeze(1) * ray.unsqueeze(-2), dim=-1)
    ray = torch.nn.functional.normalize(ray, dim=-1)

    # get pdf
    invpdf = torch.ones_like(ray)[..., 0:1] * torch.pi / (cos_theta[..., None] + 1e-7)
    
    return ray, invpdf

@torch.no_grad()
@jaxtyped
@typechecker
def ggx_importance_sample_hemisphere(
    normal: Float[Tensor, "P 3"], 
    num_ray: int, 
    roughness: Float[Tensor, "P 1"]
)->Tuple[Float[Tensor, "P N 3"], 
         Float[Tensor, "P N 1"]]:
    
    P = normal.shape[0]
    ur = sobel_sequence(P * num_ray, device=normal.device)
    ur = ur.reshape(P, num_ray, 2)

    # get Phi and Theta based on cdf^{-1}
    a2 = roughness ** 4
    phi = 2 * torch.pi * ur[..., 0]
    cos_theta = torch.sqrt((1 - ur[..., 1]) / (1 + (a2 - 1) * ur[..., 1]))
    sin_theta = torch.sqrt(1 - cos_theta ** 2)

    # get sampled direction [P, N, 3]
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = cos_theta
    ray = torch.stack([x, y, z], dim=-1)

    # rotate z axis to normal
    z_aixs = torch.zeros_like(normal)
    z_aixs[:, 2] = 1
    normal = torch.nn.functional.normalize(normal, dim=-1)
    
    rot_mat = rotation_between_vectors(z_aixs, normal) # [P, 3, 3]
    # [P, 1, 3, 3] * [P, N, 1, 3], sum in dim=-1
    ray = torch.sum(rot_mat.unsqueeze(1) * ray.unsqueeze(-2), dim=-1)
    ray = torch.nn.functional.normalize(ray, dim=-1)

    d = (cos_theta * a2 - cos_theta) * cos_theta + 1
    D = a2 / (torch.pi * d * d)
    invpdf = 1 / (D * cos_theta + 1e-7)
    invpdf = invpdf.unsqueeze(-1)

    return ray, invpdf

@torch.no_grad()
@jaxtyped
@typechecker
def octahedral_importance_sample_sphere(
    octmap: Float[Tensor, "3 H W"], 
    num_ray: int,
    random_offset: bool = True
):
    data = octmap
    _, H, W = data.shape
    # intensity
    intensity = 0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2]
    pdf = intensity / (intensity.sum() + 1e-7)

    # NOTE: Use sampling with replacement to ensure consistent probability distribution 
    # and independent samples for accurate importance sampling based on environment map.
    indices = torch.multinomial(pdf.view(-1), num_ray, replacement=True)
    
    # get pixel coordinate
    px, py = indices % W, indices / W

    if random_offset:
        rand_x = torch.rand(num_ray, device=px.device) - 0.5
        rand_y = torch.rand(num_ray, device=px.device) - 0.5
        px = (px + rand_x).clamp(0, W - 1)
        py = (py + rand_y).clamp(0, H - 1)
    else:
        px, py = px.float(), py.float()

    # to uv [0, 1]
    u, v = px / W, py / H

    # [0, 1] -> [-1, 1] -> ray
    u = 2 * u - 1
    v = 2 * v - 1
    z = 1 - u.abs() - v.abs()
    x, y = u.clone(), v.clone()

    up_part = z > 0
    x[~up_part] = (1 - v[~up_part].abs()) * u[~up_part].sgn()
    y[~up_part] = (1 - u[~up_part].abs()) * v[~up_part].sgn()

    ray = torch.stack([x, y, z], dim=-1)
    ray = torch.nn.functional.normalize(ray, dim=-1)

    # get pdf at location x, y with grid sample. 
    grid = torch.stack((u, v), dim=-1)
    
    sampled_pdf = torch.nn.functional.grid_sample(
        pdf[None, None], grid[None, None], mode='bilinear', align_corners=False)
    
    # We need pdf(wi), and octmap makes pdf easy!
    invpdf = 4 * torch.pi / (H * W * sampled_pdf[0, 0] + 1e-7)
    invpdf = invpdf.permute(1, 0)

    return ray, invpdf

@torch.no_grad()
@jaxtyped
@typechecker
def visualize_rays(
    ray: Float[Tensor, "P N 3"],
    normal: Float[Tensor, "P 3"]=None, 
    vdir: Float[Tensor, "P 3"]=None,
    octmap: Float[Tensor, "3 H W"]=None,
    point_color = [1, 1, 1],
    point_size: float = 5.0,
    normal_color = [0, 0, 1]
):
    try:
        import numpy as np
        import open3d as o3d
    except ImportError:
        raise ImportError("Open3D is not installed. Please install Open3D to use this function.")
    
    def create_arrow(length=2, radius=0.1, color=[1, 0, 0]):
        # Create a cylinder for the main body of the arrow
        direction = np.zeros(3)
        direction[2] = 1.0

        # Create cylinder
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
        # Rotate the cylinder to point from start to end
        cylinder.translate(direction * (length / 2))  # Center the cylinder on the line

        # Create a cone for the tip of the arrow
        cone = o3d.geometry.TriangleMesh.create_cone(radius=radius * 2, height=length/4)
        # Rotate and position the cone at the end of the line
        cone.translate(direction * length)  # Move cone to the end of the line

        # Set colors
        cylinder.paint_uniform_color(color)
        cone.paint_uniform_color(color)

        # Return combined geometries
        return cylinder + cone
    
    def draw_plane(scale=1, color=[0.5, 0.5, 0.5]):
        # Define a point on the plane (the origin in this case)
        # Define two points that lie on the XY plane (i.e., Z=0)
        point1 = np.array([-1, -1, 0])  # Bottom left
        point2 = np.array([-1, 1, 0])   # Top left
        point3 = np.array([1, 1, 0])    # Top right
        point4 = np.array([1, -1, 0])   # Bottom right
        
        # Create mesh for the two triangles
        mesh = o3d.geometry.TriangleMesh()

        # Add vertices for the two triangles
        vertices = np.array([point1, point2, point3, point4])  # Four corner points
        triangles = [[0, 1, 2], [0, 2, 3],  # Front triangles
                     [0, 2, 1], [0, 3, 2]]  # Back triangles (with reversed triangles)
        mesh.vertices = o3d.utility.Vector3dVector(vertices * scale)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        # Compute vertex normals
        mesh.compute_vertex_normals()
        
        # Set a color for the plane
        mesh.paint_uniform_color(color)  # Gray color

        return mesh
    
    # probe
    if octmap is None:
        probe = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        probe.compute_vertex_normals()
        z_values = np.array(probe.vertices)[:, 2] * 0.5 + 0.5
        colors = np.zeros((len(probe.vertices), 3))
        colors[:, 0] = z_values
        colors[:, 1] = 0
        colors[:, 2] = 1 - z_values
        colors = np.asarray(colors, order="C")
        probe.vertex_colors = o3d.utility.Vector3dVector(colors)
    else:
        data = {
            'vertices': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                        [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            'triangles': [[0, 1, 2], [1, 3, 2], [3, 4, 2], [4, 0, 2],
                        [0, 5, 1], [1, 5, 3], [3, 5, 4], [4, 5, 0]],
            'uvs': [[1.0, 0.5], [0.5, 1.0], [0.5, 0.5], [0.5, 1.0], 
                    [0.0, 0.5], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0], 
                    [0.5, 0.5], [0.5, 0.0], [1.0, 0.5], [0.5, 0.5],
                    [1.0, 0.5], [1.0, 1.0], [0.5, 1.0], [0.5, 1.0], 
                    [0.0, 1.0], [0.0, 0.5], [0.0, 0.5], [0.0, 0.0], 
                    [0.5, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 0.5]]
        }

        vertices = np.array(data['vertices'])
        triangles = np.array(data['triangles'], dtype=np.int32)
        uvs = np.array(data['uvs'])

        probe = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vertices), 
                o3d.utility.Vector3iVector(triangles))
        
        probe.triangle_uvs = o3d.utility.Vector2dVector(uvs)
        probe.triangle_material_ids = o3d.utility.IntVector([0] * len(triangles))

        texture = octmap.permute(1, 2, 0).cpu().numpy()
        texture = np.asarray(texture, order="C")
        texture = o3d.geometry.Image(texture)
        probe.textures = [texture]
        probe.compute_vertex_normals()

    # Convert torch Tensors to numpy arrays
    # Assuming ray shape is [P=1, N, 3]
    if octmap is not None:
        ray_norm = torch.nn.functional.normalize(ray[0], p=1, dim=-1)
    else:
        ray_norm = torch.nn.functional.normalize(ray[0], p=2, dim=-1)

    ray_np = ray_norm.cpu().numpy()
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(ray_np)
    colors = np.ones_like(ray_np) * np.array(point_color)[None]
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # create normal direction
    arrow, plane = None, None
    if normal is not None:
        # get rotation matrix
        z_aixs = torch.zeros_like(normal)
        z_aixs[:, 2] = 1
        rot_mat = rotation_between_vectors(z_aixs, normal)[0]
        rotmat_np = rot_mat.cpu().numpy()
        
        arrow = create_arrow(length=1.2, radius=0.05, color=normal_color)
        arrow.rotate(rotmat_np, center=(0, 0, 0))
    
        plane = draw_plane(scale=1.3, color=[0.8, 0.8, 0.8])
        plane.rotate(rotmat_np, center=(0, 0, 0))
    
    arrow2, plane2 = None, None
    if vdir is not None:
        # get rotation matrix
        z_aixs = torch.zeros_like(vdir)
        z_aixs[:, 2] = 1
        rot_mat = rotation_between_vectors(z_aixs, vdir)[0]
        rotmat_np = rot_mat.cpu().numpy()
        
        arrow2 = create_arrow(length=1.2, radius=0.05, color=[1, 0, 0])
        arrow2.rotate(rotmat_np, center=(0, 0, 0))
    
        plane2 = draw_plane(scale=1.3, color=[0.8, 0.8, 0.8])
        plane2.rotate(rotmat_np, center=(0, 0, 0))

    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5)
    # Visualize
    # o3d.visualization.draw_geometries([sphere, point_cloud, arrow, plane])
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for mesh in [probe, point_cloud, arrow, plane, arrow2]:
        if mesh is not None:
            vis.add_geometry(mesh)
    
    render_option = vis.get_render_option()
    render_option.show_coordinate_frame = True
    render_option.light_on = True
    render_option.point_size = point_size

    # Customize the view
    ctr = vis.get_view_control()

    # Set camera position (looking down the positive Y axis)
    # Position the camera at (0, 5, 0) and look at the origin (0, 0, 0)
    ctr.set_lookat([0, 0, 0])  # Look at the origin (0, 0, 0)
    ctr.set_up([0, 0, 1])  # Set the up direction to Z axis
    ctr.set_front([0, -3, 0])  # Set the front direction towards negative Y axis
    ctr.set_zoom(1)  # Set zoom level

    vis.run()
    vis.destroy_window()
