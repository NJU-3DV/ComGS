
import torch
import numpy as np
from typing import List
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from fusion._C import multi_view_fusion

class Reconstruction(object):
    def __init__(
        self, viewpoints: List[Camera], renderFunc, pipe, 
        num_src=20, min_conistency_num=5, reproj_error_threshold=2, 
        relative_depth_threshold=0.01, normal_threshold=10, step=2
    ):
        
        # for rendering
        self.viewpoints = viewpoints
        self.renderFunc = renderFunc
        self.pipe = pipe

        # for multi-view consistency fusion
        self.num_images = viewpoints.__len__()
        self.device = viewpoints[0].data_device
        self.num_src = min(num_src, self.num_images - 1)

        self.Kvecs = [view.intrinsics for view in viewpoints]
        self.Rmats = [view.Rw2c for view in viewpoints]
        self.tvecs = [view.tw2c for view in viewpoints]
        self.pairs = self.generate_pairs()

        self.min_conistency_num = min_conistency_num
        self.reproj_error_threshold = reproj_error_threshold
        self.relative_depth_threshold = relative_depth_threshold
        self.normal_threshold = normal_threshold

        self.step = step
    
    def generate_pairs(self):
        theta0, sigma1, sigma2 = 5.0, 1.0, 10.0
        v = np.array([0, 0, 1], dtype=np.float32)
        score_mat = np.zeros((self.num_images, self.num_images), dtype=np.float32)

        Rmats = [R.detach().cpu().numpy() for R in self.Rmats]
        for i in range(self.num_images):
            # referencecamera view direction
            v_ref = (Rmats[i]).transpose() @ v
            norm1 = np.sqrt(np.dot(v_ref, v_ref))

            for j in range(self.num_images):
                if i == j:
                    continue

                v_src = (Rmats[j]).transpose() @ v
                norm2 = np.sqrt(np.dot(v_src, v_src))

                theta = (180 / np.pi) * np.arccos(np.dot(v_ref, v_src) / (norm1 * norm2)) 
                score_mat[i, j] = np.exp(-(theta - theta0) ** 2 / (2 * (sigma1 if theta <= theta0 else sigma2) ** 2))
        
        return np.argsort(-score_mat, axis=1)[:, :self.num_src].tolist()
    
    @torch.no_grad()
    def run(self, gaussians: GaussianModel):

        bg_func = lambda x, y: torch.zeros(3, device="cuda")
        # Reconstruction
        rgbs, depths, normals, masks = [], [], [], []
        for view in self.viewpoints:
            render_pkg = self.renderFunc(
                view, gaussians, self.pipe, 
                bg_func=bg_func, 
                requires_geometry=True, 
                requires_material=False
            )

            alpha = render_pkg["rend_alpha"]
            depth = render_pkg["rend_depth"]
            normal = render_pkg["rend_normal"] / alpha.clamp_min(1e-6)
            normal = torch.nn.functional.normalize(normal, dim=0, p=2)

            mask = (alpha > 0.9).float()
            depth = depth * mask
            normal = normal * mask            

            rgbs.append(render_pkg["render"].contiguous())
            depths.append(depth.squeeze(0).contiguous())
            normals.append(normal.contiguous())
        
        # Fusion
        xyz, rgb, normal = multi_view_fusion(
            [d.cpu() for d in rgbs], 
            [d.cpu() for d in self.Kvecs],
            [d.cpu() for d in self.Rmats],
            [d.cpu() for d in self.tvecs], 
            [d.cpu() for d in depths], 
            [d.cpu() for d in normals], 
            masks, 
            self.pairs, 
            self.min_conistency_num,
            self.reproj_error_threshold,
            self.relative_depth_threshold,
            self.normal_threshold,
            self.step
        )

        return xyz, rgb, normal