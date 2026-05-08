
#pragma once

#include <torch/extension.h>

namespace sample{

    std::tuple<torch::Tensor, torch::Tensor>
    uniform_sample(
        const torch::Tensor& normal,
        const int num_ray,
        const float cos_theta_min,
        const float cos_theta_max,
        const bool random_rotate
    );
    

}
