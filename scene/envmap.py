


import os
import torch
import torch.nn as nn
import imageio
import roma
from utils.pbr_utils import srgb_to_rgb

# UV coordinates always start from the top-left corner of the top-left pixel.

class EnvMapBase:
    def __init__(self, device="cuda"):

        self.device = device
        
        self.data = torch.empty(0).to(device=self.device)
        self.activate_fn = None

        self.optimizer = None
    
    @property
    def resolution(self):
        raise NotImplementedError

    def capture(self):
        return (self.data, self.optimizer.state_dict())
    
    def restore(self, model_args, training_args):
        self.data, opt_dict = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)
    
    def training_setup(self, training_args):

        self.data = nn.Parameter(self.data).requires_grad_(True)

        l = [{'params': [self.data], 'lr': training_args.envmap_lr, "name": "envmap"}]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    
    def sample(self, vdir):
        raise NotImplementedError
    
    def export_as_spherical(self, resolution=-1, convention="xyz", angles=[0., 0., 0.], degrees=True):
        if resolution <= 0:
            resolution = self.resolution
        
        h, w = resolution, resolution * 2
        v, u = torch.meshgrid([torch.linspace(0.5, h-0.5, h), torch.linspace(0.5, w-0.5, w)], indexing="ij")
        v = v.to(self.device)
        u = u.to(self.device)

        nu = u / w
        nv = v / h

        phi = nv * torch.pi
        theta = torch.pi - 2 * nu * torch.pi

        viewdirs = torch.stack([
            torch.sin(phi) * torch.cos(theta),
            torch.sin(phi) * torch.sin(theta),
            torch.cos(phi)], dim=0)    # [3, H, W]
        
        angles = torch.tensor(angles, device=viewdirs.device, dtype=torch.float)
        Rmat = roma.euler_to_rotmat(convention, angles, degrees)
        viewdirs = Rmat @ viewdirs.reshape(3, -1)
        viewdirs = viewdirs.reshape(3, h, w)

        envmap = self.sample(viewdirs)

        return envmap
    
    def export_as_octahedral(self, resolution=-1):
        if resolution <= 0:
            resolution = self.resolution
        
        w = resolution
        v, u = torch.meshgrid([torch.linspace(0.5, w-0.5, w), torch.linspace(0.5, w-0.5, w)], indexing="ij")
        v, u = v.to(self.device), u.to(self.device)

        x = 2. * u / w - 1.
        y = 2. * v / w - 1.
        z = 1. - x.abs() - y.abs()
        
        up_part = z > 0
        x_down = (1. - y[~up_part].abs()) * x[~up_part].sgn()
        y_down = (1. - x[~up_part].abs()) * y[~up_part].sgn()
        x[~up_part] = x_down
        y[~up_part] = y_down

        vdir = torch.stack([x, y, z], dim=0)
        envmap = self.sample(vdir)

        return envmap

class SphericalEnvMap(EnvMapBase):
    def __init__(self, resolution=256, init_value=1.5, act_fn="exp", device="cuda"):

        super().__init__(device)

        if act_fn == "exp":
            self.activate_fn = torch.exp
            self.inverse_activate_fn = torch.log
        elif act_fn == "none":
            self.activate_fn = lambda x: x
            self.inverse_activate_fn = lambda x: x
        else:
            raise ValueError(f"Unsupported activation function: {act_fn}")

        self.data = self.inverse_activate_fn(torch.full((3, resolution, resolution * 2), fill_value=init_value).to(self.device))
    
    @property
    def resolution(self):
        return self.data.shape[1]

    @property
    def height(self):
        return self.data.shape[1]
    
    @property
    def width(self):
        return self.data.shape[2]
    
    def set_data(self, data):
        self.data = self.inverse_activate_fn(data)

    def load_from_file(self, file_path, is_linear_rgb=True):
        assert os.path.exists(file_path), f"Cannot find the file: {file_path}"
        img = imageio.imread(file_path)

        assert len(img.shape) == 3 and 2 * img.shape[0] == img.shape[1], f"The aspect ratio of the Spherical map needs to be 2:1."
        data = torch.from_numpy(img).permute(2, 0, 1).to(self.device)

        file_type = os.path.splitext(file_path)[-1]
        if file_type != ".exr" and file_type != ".hdr":
            data = data.float() / 255.
        
        if not is_linear_rgb:
            data = srgb_to_rgb(data)

        # inverse activated
        self.data = self.inverse_activate_fn(data)
        
    def sample(self, vdir):
        # step 1: vdir: check shape, reshape, and normalization
        assert vdir.shape[0] == 3, "vdir must be a tensor with shape [3, ...]" 
        vdir_shape = vdir.shape
        vdir = vdir.reshape(3, -1)
        vdir = torch.nn.functional.normalize(vdir, dim=0)

        # step2: from direction to uv
        theta = torch.atan2(vdir[1], vdir[0])
        phi = torch.acos(vdir[2])

        nu = (torch.pi - theta) / (2 * torch.pi)

        nv = phi / torch.pi

        u = nu * self.width
        v = nv * self.height

        data = self.activate_fn(self.data)

        # step3: interpolation, please note that we have extra margin
        sampled_x = u / self.width * 2. - 1.
        sampled_y = v / self.height * 2. - 1.
        grid = torch.stack((sampled_x, sampled_y), dim=-1).unsqueeze(0).unsqueeze(0)
        light = torch.nn.functional.grid_sample(data[None], grid, align_corners=False, padding_mode="reflection").squeeze()
    
        return light.reshape(vdir_shape)

class OctahedralEnvMap(EnvMapBase):
    def __init__(self, resolution=256, init_value=1.5, act_fn="exp", device="cuda"):

        super().__init__(device)

        if act_fn == "exp":
            self.activate_fn = torch.exp
            self.inverse_activate_fn = torch.log
        elif act_fn == "none":
            self.activate_fn = lambda x: x
            self.inverse_activate_fn = lambda x: x
        else:
            raise ValueError(f"Unsupported activation function: {act_fn}")

        self.data = self.inverse_activate_fn(torch.full((3, resolution, resolution), fill_value=init_value).to(self.device))
    
    @property
    def resolution(self):
        return self.data.shape[1]

    @property
    def height(self):
        return self.data.shape[1]
    
    @property
    def width(self):
        return self.data.shape[2]
    
    def set_data(self, data):
        self.data = self.inverse_activate_fn(data)
    
    def load_from_file(self, file_path, is_linear_rgb=True):
        assert os.path.exists(file_path), f"Cannot find the file: {file_path}"
        img = imageio.imread(file_path)
        
        assert len(img.shape) == 3 and img.shape[0] == img.shape[1], f"Only square octahedral maps are supported."
        data = torch.from_numpy(img).permute(2, 0, 1).to(self.device)

        file_type = os.path.splitext(file_path)[-1]
        if file_type != ".exr" and file_type != ".hdr":
            data = data.float() / 255.
        
        if not is_linear_rgb:
            data = srgb_to_rgb(data)

        # inverse activated
        self.data = self.inverse_activate_fn(data)
        
    def get_at_resolution(self, resolution=32):
        # resize self.data to resolution
        envmap = self.activate_fn(self.data)
        resized = torch.nn.functional.interpolate(
            envmap[None], size=resolution, 
            mode="bilinear", align_corners=False)[0]
        return resized

    def get_data(self):
        return self.activate_fn(self.data)
    
    def sample(self, vdir):

        # step 1: vdir: check shape, reshape, and normalization
        assert vdir.shape[0] == 3, "vdir must be a tensor with shape [3, ...]" 

        with torch.no_grad():
            vdir_shape = vdir.shape
            vdir = vdir.reshape(3, -1)
            vdir = torch.nn.functional.normalize(vdir, dim=0)

            # step2: from direction to uv
            r = torch.nn.functional.normalize(vdir, p=1, dim=0) # [3, N]
            up_part = r[2] > 0

            nu = torch.zeros_like(r[0])
            nv = torch.zeros_like(r[0])
            nu[up_part] = r[0][up_part]
            nv[up_part] = r[1][up_part]

            nu[~up_part] = (1.0 - r[1][~up_part].abs()) * r[0][~up_part].sgn()
            nv[~up_part] = (1.0 - r[0][~up_part].abs()) * r[1][~up_part].sgn()

            # [-1, 1] -> [0, 1] normalized uv
            nu = nu * 0.5 + 0.5
            nv = nv * 0.5 + 0.5

            u = nu * self.width
            v = nv * self.height

            # step3: interpolation, please note that we have extra margin
            sampled_x = (u + 1.) / (self.width + 2.) * 2. - 1.
            sampled_y = (v + 1.) / (self.height + 2.) * 2. - 1.
            # sampled_x = u / (self.width) * 2. - 1.
            # sampled_y = v / (self.height) * 2. - 1.

            grid = torch.stack((sampled_x, sampled_y), dim=-1).unsqueeze(0).unsqueeze(0)

        # step4: add margin and activate the data
        # To ensure accurate bilinear interpolation at the edges, extra margin are added.
        l = self.resolution // 2 - 1
        data = torch.zeros([3, self.height + 2, self.width + 2], dtype=torch.float32, device=self.device)

        data[..., 0, 0] = self.data[..., -1, -1]
        data[..., -1, 0] = self.data[..., 0, -1]
        data[..., 0, -1] = self.data[..., -1, 0]
        data[..., -1, -1] = self.data[..., 0, 0]

        data[..., 0, 1:l+2] = self.data[..., 0, l+1:].flip(dims=[-1])
        data[..., 0, l+2:-1] = self.data[..., 0, 0:l+1].flip(dims=[-1])
        data[..., -1, 1:l+2] = self.data[..., -1, l+1:].flip(dims=[-1])
        data[..., -1, l+2:-1] = self.data[..., -1, 0:l+1].flip(dims=[-1])
        
        data[..., 1:l+2, 0] = self.data[..., l+1:, 0].flip(dims=[-1])
        data[..., l+2:-1, 0] = self.data[..., 0:l+1, 0].flip(dims=[-1])
        data[..., 1:l+2, -1] = self.data[..., l+1:, -1].flip(dims=[-1])
        data[..., l+2:-1, -1] = self.data[..., 0:l+1, -1].flip(dims=[-1])

        data[..., 1:-1, 1:-1] = self.data
        data = self.activate_fn(data)

        light = torch.nn.functional.grid_sample(data[None], grid, align_corners=False).squeeze()
    
        return light.reshape(vdir_shape)