#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
fusedssim(
    const float C1,
    const float C2,
    const torch::Tensor &img1,
    const torch::Tensor &img2,
    const int window_size,
    const bool train
);

torch::Tensor
fusedssim_backward(
    const float C1,
    const float C2,
    const torch::Tensor &img1,
    const torch::Tensor &img2,
    const int window_size,
    const torch::Tensor &dL_dmap,
    const torch::Tensor &dm_dmu1,
    const torch::Tensor &dm_dsigma1_sq,
    const torch::Tensor &dm_dsigma12
);
