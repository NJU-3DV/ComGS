
# ComGS: Efficient 3D Object-Scene Composition via Surface Octahedral Probes

[![arXiv](https://img.shields.io/badge/arXiv-2510.07729-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.07729)
[![Project Page](https://img.shields.io/badge/Project-Page-1E90FF?logo=githubpages&logoColor=white)](https://nju-3dv.github.io/projects/ComGS/)
[![Dataset](https://img.shields.io/badge/Dataset-SynCom-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/datasets/yGaoJiany/SynCom)


<video autoplay muted loop playsinline src="assets/teaser.mp4" title="ComGS"></video>

## Installation

Clone this repo through:

```shell
git clone --recursive https://github.com/NJU-3DV/ComGS.git
```

Then, create a new virtual environment, activate it and install packages:

``` shell
# Create an environment
conda create -n comgs python==3.11.0
# Activate
conda activate comgs

# install pytorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install packages
pip install -r requirements.txt
```

Make sure you have a C++ compiler and CUDA 12.1 installed. This setup was tested on Windows with VS2019 and Ubuntu with g++ 11.4.0.

```shell
# windows (Powershell)
install.bat

# Ubuntu
sh install.sh 
```

## Datasets

### TensoIR dataset

Download the official TensoIR data from [Zenodo](https://zenodo.org/records/7880113#.ZE68FHZBz18), and download the relighting environment maps from [Google Drive](https://drive.google.com/file/d/10WLc4zk2idf4xGb6nPL43OXTTHvAXSR3/view?pli=1).

The TensoIR data are used for object reconstruction, and `Environment_Maps/high_res_envmaps_2k` is used for relighting evaluation. After downloading and extracting the files, organize them as follows:


```text
<tensoir_root>/
├── armadillo/
├── ficus/
├── hotdog/
├── lego/
└── Environment_Maps/
    └── high_res_envmaps_2k/
        ├── bridge.hdr
        ├── city.hdr
        ├── fireplace.hdr
        ├── forest.hdr
        └── night.hdr
```

### SynCom dataset

Download SynCom from [Hugging Face](https://huggingface.co/datasets/yGaoJiany/SynCom).

After downloading and extracting SynCom, organize it as follows:

```text
<syncom_root>/
├── object/
├── scene/
├── composition/
└── demo/
```

- `object/`: inputs for object reconstruction.
- `scene/`: inputs for scene reconstruction.
- `composition/`: inputs for object-scene composition.
- `demo/`: optional inputs for higher-quality demo rendering.

## Quick Start

To quickly inspect the 3D object-scene composition results, use the reconstructed assets from the experiments: [ComGS Assets](https://huggingface.co/datasets/yGaoJiany/ComGS_asserts).

The quick-start assets should be organized as follows:

```text
<workspace>/
├── syncom/
│   ├── objects/
│   ├── scenes/
│   └── envmap/
├── public/
│   ├── comp_src/
│   ├── objects/
│   ├── scenes/
│   └── envmap/
└── capture/
    ├── comp_src/
    ├── objects/
    ├── scenes/
    └── envmap/
```

- `objects/` and `scenes/` store the reconstructed assets used for composition.
- `envmap/` stores precomputed lighting assets.
- `comp_src/` stores composition cases for the Public and Phone-Captured datasets:

```text
comp_src/
└── {scene_name}_with_{object_name}/
    ├── sparse/ or cameras.json
    ├── transform.json
    └── camera_selection.json  # optional
```

### SynCom dataset

To generate composition results for the SynCom dataset, run:
```shell
# with SOPs
python -m scripts.composition.syncom --root_dir <syncom_root>/composition --exp_dir <workspace>/syncom --fps 5 --with_ao

# with tracing
python -m scripts.composition.syncom --root_dir <syncom_root>/composition --exp_dir <workspace>/syncom --fps 5 --with_trace --with_ao

```
Afterward, the results can be found under `<workspace>/syncom/composition` and `<workspace>/syncom/composition_trace`.

For smoother video results, you can render interpolated camera paths at a higher frame rate. Demo configurations are provided in `<syncom_root>/demo` for 10-second videos rendered at 60 FPS.

To render the demo trajectories, run:
```shell
python -m scripts.composition.syncom --root_dir <syncom_root>/demo --exp_dir <workspace>/syncom --sops_num 40_000 --fps 60 --with_ao
```

### Public dataset

To generate composition results for the Public dataset, run:
```shell
python -m scripts.composition.public --root_dir <workspace>/public/comp_src --exp_dir <workspace>/public
```
Afterward, the results can be found under `<workspace>/public/composition`.

### Phone-Captured dataset

To generate composition results for the Phone-Captured dataset, run:
```shell
python -m scripts.composition.capture --root_dir <workspace>/capture/comp_src --exp_dir <workspace>/capture
```
Afterward, the results can be found under `<workspace>/capture/composition`.

## Reconstruction

![assets/reconstruction.svg](assets/reconstruction.svg)

Run full reconstruction and evaluation with the provided scripts:
```shell
# TensoIR-Object dataset
python -m scripts.reconstruction.tensoir --root_dir <tensoir_root> --exp_dir <workspace>/tensoir

# SynCom-Object dataset
python -m scripts.reconstruction.syncom_object --root_dir <syncom_root>/object --exp_dir <workspace>/syncom/objects

# SynCom-Scene dataset
python -m scripts.reconstruction.syncom_scene --root_dir <syncom_root>/scene --exp_dir <workspace>/syncom/scenes
```

By default, all available GPUs are used for parallel processing. If you want to specify which GPUs to use, you can set it with the `--gpus` argument.

The results for object reconstruction will be organized as follows:

| Type | Description | Path |
|------|--------------|------|
| Quantitative | Geometry & material | `<workspace>/{dataset}/{name}/results.json` |
| Quantitative | Relighting | `<workspace>/{dataset}/{name}/test_rli/results.json` |
| Visualization | Geometry & material | `<workspace>/{dataset}/{name}/test/` |
| Visualization | Relighting | `<workspace>/{dataset}/{name}/test_rli/{env_map}/` |

If these reconstruction results will be used for composition, please organize them under a workspace layout in which `objects/` and `scenes/` are direct subdirectories of the composition workspace. 

## Composition

![\[Figure\]](assets/composition.svg)


For composition, download the [PanoLight weights](https://huggingface.co/yGaoJiany/pano_light) and put them in `pano_light/checkpoints`:

Use the following commands to perform composition on the SynCom dataset. Note that object and scene reconstructions must be completed beforehand.

```shell
# with SOPs
python -m scripts.composition.syncom --root_dir <syncom_root>/composition --exp_dir <workspace>/syncom --fps 5

# with tracing
python -m scripts.composition.syncom --root_dir <syncom_root>/composition --exp_dir <workspace>/syncom --fps 5 --with_trace
```
To exactly reproduce the renderings reported in the paper, use the per-scene lighting assets and configurations from the experiments. Since the illumination completion stage relies on stochastic diffusion sampling, regenerating HDR maps from scratch may lead to different illumination estimates, particularly affecting shadow consistency and high-frequency lighting details. A promising future direction is to incorporate inverse-rendering-guided diffusion priors, such as [Diffusion Posterior Illumination](https://vcai.mpi-inf.mpg.de/projects/2023-DPE/), which could help better constrain illumination reconstruction.

## With your own dataset

### Frame Extraction

Extract frames from videos via:
```shell
python -m scripts.preprocess.extract_frames -i <object_data_root>/video -o <object_data_root>/frames -t 200 
```
This script samples `-t` frames from each video in the input folder (`-i`) proportionally to its duration, and saves the results to the output directory (`-o`).

### Structure from Motion
Then, perform **Structure-from-Motion (SfM)** using **COLMAP**:

```shell
python -m scripts.preprocess.convert -s <object_data_root> --single_camera
```

In the convert step, Model Alignment ensures that the scene’s Z-axis points upward, which has two advantages:  
(1) It facilitates manual object placement;  
(2) It better fits the input requirements for lighting estimation.

In practice, Manhattan alignment may not always guarantee that the Z-axis is perfectly upright. You can verify this by opening the COLMAP GUI and importing the model from `<object_data_root>/sparse/align` for inspection.
Next, manually adjust the orientation using the provided `align.py` script:

```shell
python -m scripts.preprocess.align -s <object_data_root> -a -90 0 0
```

In this example, the entire model is rotated -90 degrees around the X-axis, so that the Z-axis points upward.
Open the COLMAP GUI and import `<object_data_root>/sparse/0` to confirm that the Z-axis is correctly oriented.

As a side note, in COLMAP, the X, Y, and Z axes are visualized using red, green, and blue colors, respectively.

### Object Masking

First, install the extra GUI dependencies:

```shell
# install GUI requirements
pip install -r requirements-sam2.txt
```

Then install `sam2` from the official repository:

```shell
pip install git+https://github.com/facebookresearch/sam2.git
```

By default, the GUI loads `facebook/sam2-hiera-large` from Hugging Face on first run. If you prefer a local checkpoint, you can still pass both `--checkpoint_dir` and `--model_cfg` manually.

To extract object masks from video frames, use this app:

```shell
python -m scripts.preprocess.sam2_mask_app
```
First, enter the path to the image folder in the `Image folder` input box.  
Then click `Get SAM Feature` and select one or more frames to add point prompts for the target object.  
Next, click `Submit mask and track` to segment the object throughout the video.  
Finally, click `Save masks to folder` to save the results.

### Reconstruction

For object reconstruction, run:
```shell
python train.py -s <object_data_root> -m <workspace>/mine/objects/<obj_name> --eval -r 2
```
The option `-r 2` means using a 2× downsampling factor, which can be adjusted as needed.

For scene reconstruction, run:
```shell
python train.py --iterations 30000 --lambda_mask 0.0 --mask_loss_from_iter 30001 --opacity_prune_threshold 0.05 -s <scene_data_root> -m <workspace>/mine/scenes/<scene_name> --eval -r 2
```
The option `--iterations 30000` specifies that only stage 1 is performed for reconstructing the scene’s geometry and radiance.

If you plan to run composition afterward, the directory passed as `--workspace <workspace>/mine` must contain `objects/` and `scenes/` as direct subdirectories. In other words, the outputs above should remain under `<workspace>/mine/objects/...` and `<workspace>/mine/scenes/...`.

### Get transform for object placement.

Use [**CloudCompare**](https://www.cloudcompare.org/) for object–scene composition.

1. Drag both the **object** and **scene** Gaussian point clouds into CloudCompare.  
2. In the **DB Tree** panel on the left, select the **object** point cloud.  
3. Go to **Edit → Translate/Rotate** in the top menu, and move the object to the desired position within the scene point cloud.  
4. In the **Properties** panel on the left, you can find the corresponding transformation matrix.
5. Write the matrix into json file like this:
```json
{
    "obj_name": {
        "transform": [
            0.400000,
            0.000000,
            0.000000,
            0.107700,
            0.000000,
            0.400000,
            0.000000,
            0.755418,
            0.000000,
            0.000000,
            0.400000,
            -1.075989,
            0.000000,
            0.000000,
            0.000000,
            1.000000
        ]
    }
}
``` 
Make sure the `obj_name` key matches the object name under `<workspace>/mine/objects` used for composition.

### Composition

Then, run the following command to perform composition:
```shell
python composition.py -s <composition_data_root> --workspace <workspace>/mine --num_rays 256 --latent_scale 4 --sops_num 40000 --label <label> --fps 30
```

Set `--label` to `indoor` or `outdoor` according to the scene type.

In practice, prepare `<composition_data_root>` by copying `<scene_data_root>`, renaming the copied directory, and adding a `transform.json` file that specifies the object transformation within the scene.

Because albedo and environment lighting are jointly identifiable only up to a scale factor, custom datasets without ground-truth material measurements typically require manual albedo calibration. This can be done by creating an `albedo_ratio.json` file under `<workspace>/mine/objects/<obj_name>`, for example:
```
[0.2, 0.2, 0.2]
```
The three values are multiplicative scale factors for the R, G, and B albedo channels, respectively. If this file is not provided, the system falls back to `[1.0, 1.0, 1.0]`, which may lead to suboptimal material appearance in the final composition.

## Acknowledgements

We gratefully acknowledge the authors and contributors of the following projects and resources:

**Direct technical foundations**
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)
- [GTracer](https://github.com/fudan-zvg/gtracer)
- [FRNN](https://github.com/lxxue/FRNN)
- [ACMH](https://github.com/GhiXu/ACMH)
- [SAM2-GUI](https://github.com/YunxuanMao/SAM2-GUI)

**Datasets and benchmarks**
- [TensoIR](https://github.com/Haian-Jin/TensoIR)

**Lighting and diffusion references**
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [DiffusionLight](https://github.com/DiffusionLight/DiffusionLight)
- [EMLight](https://github.com/fnzhan/EMLight)
- [Tex2Light](https://github.com/FrozenBurning/Text2Light)

**Assets and scene resources**
- [BlenderKit](https://www.blenderkit.com/)
- [Poly Haven](https://polyhaven.com/)

## License

This repository is distributed under a mixed-license structure. No single
license applies to the repository as a whole.

The following original ComGS components are released under the MIT License:

- `submodules/cuda_pbr/**`
- `composition.py`
- `scripts/composition/**`
- `utils/comp_utils.py`
- `scene/sops.py`
- `gaussian_renderer/pbr.py`

These files are covered by [LICENSE-MIT-COMGS.md](LICENSE-MIT-COMGS.md).

Components derived from or based on Gaussian Splatting and related code remain
subject to the Gaussian Splatting License and its non-commercial
research-use restrictions. See [LICENSE-GS.md](LICENSE-GS.md).

This repository also contains additional third-party code, notices, and model
assets under their own terms. See [LICENSE](LICENSE) and
[LICENSE-NOTICES.md](LICENSE-NOTICES.md) for the repository-wide license map.

## Citation

If you find this code or paper useful in your research, please consider citing it:

```
@inproceedings{gao2026comgs,
  title={Com{GS}: Efficient 3D Object-Scene Composition via Surface Octahedral Probes},
  author={Jian Gao and Mengqi Yuan and Yifei Zeng and Chang Zeng and Zhihao Li and Zhenyu Chen and Weichao Qiu and Xiao-Xiao Long and Hao Zhu and Xun Cao and Yao Yao},
  booktitle={ICLR},
  year={2026}
}
```
