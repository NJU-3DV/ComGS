@REM  for gaussian splatting
pip install ./submodules/simple-knn
pip install ./submodules/fused-ssim
@REM  for relightable
pip install ./submodules/comgs_rasterizer
@REM  for inverse pbr
pip install ./submodules/cuda_pbr
@REM  for surface Octahedral Probes
pip install ./submodules/FRNN/external/prefix_sum
pip install ./submodules/FRNN
pip install ./submodules/fusion

@REM  tracer
cd ./submodules/gtracer
@REM use cmake to build the project for ptx file (for Optix)
rmdir /s /q build >nul 2>&1 
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cd ../
@REM Install the package
pip install .
cd ../../
