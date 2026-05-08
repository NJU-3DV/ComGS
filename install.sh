# for gaussian splatting
pip install ./submodules/simple-knn
pip install ./submodules/fused-ssim
# for relightable
pip install ./submodules/comgs_rasterizer
# for inverse pbr
pip install ./submodules/cuda_pbr
# for surface Octahedral Probes
pip install ./submodules/FRNN/external/prefix_sum
pip install ./submodules/FRNN
pip install ./submodules/fusion

# tracer
cd ./submodules/gtracer
# use cmake to build the project for ptx file (for Optix)
rm -rf ./build && mkdir build && cd build && cmake .. && make && cd ../
# Install the package
pip install .
cd ../../
