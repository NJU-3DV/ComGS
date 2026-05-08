# Third-Party Notices

This repository contains code and assets originating from multiple upstream
projects. The notes below are intended as a practical license map for this
repository snapshot. They do not replace the original license texts.

## MIT-Licensed ComGS Components

The following components are released under the MIT License by the ComGS
authors. See [LICENSE-MIT-COMGS.md](D:/project/RealComGS/LICENSE-MIT-COMGS.md).

- `submodules/cuda_pbr/**`
- `composition.py`
- `scripts/composition/**`
- `utils/comp_utils.py`
- `scene/sops.py`
- `gaussian_renderer/pbr.py`

## Gaussian Splatting License Components

The following files or directories are derived from, based on, or distributed
under the Gaussian Splatting license family and remain subject to
[LICENSE-GS.md](D:/project/RealComGS/LICENSE-GS.md), including its
non-commercial research-use restrictions.

- `arguments/__init__.py`
- `gaussian_renderer/__init__.py`
- `metrics.py`
- `render.py`
- `scene/__init__.py`
- `scene/cameras.py`
- `scene/colmap_loader.py`
- `scene/dataset_readers.py`
- `scene/gaussian_model.py`
- `scripts/preprocess/convert.py`
- `train.py`
- `utils/camera_utils.py`
- `utils/general_utils.py`
- `utils/graphics_utils.py`
- `utils/gs_utils.py`
- `utils/image_utils.py`
- `utils/loss_utils.py`
- `utils/system_utils.py`
- `submodules/simple-knn/**`
- `submodules/comgs_rasterizer/**`

## Additional Third-Party Components

- `submodules/gtracer/**`
  - Local repository includes multiple notices, including MIT-related and
    NVIDIA-related license files.
  - See files retained under `submodules/gtracer/`, including
    `LICENSE_KIUI` and `LICENSE_NVIDIA`.

- `submodules/fused-ssim/**`
  - See `submodules/fused-ssim/LICENSE`.

- `submodules/glm/**`
  - See `submodules/glm/copying.txt`.

- `utils/render_utils.py`
  - Retains Apache License 2.0 header from Google LLC.

- `utils/read_write_model.py`
  - Retains BSD-style license header from ETH Zurich and UNC Chapel Hill.

- `utils/sh_utils.py`
  - Retains BSD-style attribution header from The PlenOctree Authors.

- `scripts/preprocess/sam2_mask_app.py`
  - Modified from `YunxuanMao/SAM2-GUI`.
  - Upstream project attribution is retained in the source file.
  - Downstream users should verify upstream license terms when redistributing
    or reusing this file independently.

## Model and Asset Notes

- `pano_light/checkpoints/stable-diffusion-2-1/**`
  - Includes upstream Stable Diffusion 2.1 model files and notices.
  - See the files distributed in that directory, including `README.md` and
    `NOTICE.md`.

## Responsibility

Users distributing this repository, or any subset of it, should preserve the
relevant copyright, attribution, notice, and license files that apply to the
materials they redistribute.
