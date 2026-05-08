# 3D Gaussian Ray Tracer

An OptiX-based differentiable 3D Gaussian Ray Tracer, inspired by the work [3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes](https://gaussiantracer.github.io/). See [gaussian-raytracing](https://github.com/fudan-zvg/gaussian-raytracing) for 3D reconstruction.

### Install
This is a fork from https://github.com/fudan-zvg/gtracer with few modifications.

#### Install in Linux
```bash
# cd into the folder
cd ./submodules/gtracer

# use cmake to build the project for ptx file (for Optix)
rm -rf ./build && mkdir build && cd build && cmake .. && make && cd ../

# Install the package
pip install .
```

#### Install in Windows
Suppose that we are using VS 2019.

```bash
# cd into the folder
cd ./submodules/gtracer

# use cmake to build the project for ptx file (for Optix)
Remove-Item -Path './build' -Recurse -Force -ErrorAction SilentlyContinue 
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cd ../

# Install the package
pip install .
```

### Example usage
```bash
cd example
# Interactive viewer for 3DGS format point cloud
python renderer.py -p point_cloud.ply
```

### Acknowledgement

* Credits to [Instant-NGP](https://github.com/NVlabs/instant-ngp) and [raytracing](https://github.com/NVlabs/instant-ngp).
* Credits to the original [3D Gaussian Ray Tracing](https://gaussiantracer.github.io/) paper.


## 📜 Citation
If you find this work useful for your research, please cite our github repo:
```bibtex
@misc{gu2024gtracer,
    title = {3D Gaussian Ray Tracer},
    author = {Gu, Chun and Zhang, Li},
    howpublished = {\url{https://github.com/fudan-zvg/gtracer}},
    year = {2024}
}
```
