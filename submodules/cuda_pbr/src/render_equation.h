
#pragma once

#include "utils.h"
#include <torch/extension.h>

// points [P, 3]
// normals [P, 3]
// albedo [P, 3]
// metallic [P, 1]
// roughness [P, 1]

// inray [P, W_in, 3]
// radiance [P, W_in, 3]
// inarea [P, W_in, 1]
// outray [P, W_out=1, 3]

namespace pbr{
    
    torch::Tensor RenderingEquation(
        const torch::Tensor& normals,
        const torch::Tensor& albedo,
        const torch::Tensor& metallic,
        const torch::Tensor& roughness,
        const torch::Tensor& inray,
        const torch::Tensor& radiance,
        const torch::Tensor& inarea,
        const torch::Tensor& outray
    );

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    RenderingEquationBackward(
        const torch::Tensor& normals,
        const torch::Tensor& albedo,
        const torch::Tensor& metallic,
        const torch::Tensor& roughness,
        const torch::Tensor& inray,
        const torch::Tensor& radiance,
        const torch::Tensor& inarea,
        const torch::Tensor& outray,
        const torch::Tensor& dL_drgb
    );
}
