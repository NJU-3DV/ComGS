
#ifndef _FUSION_H_
#define _FUSION_H_

// #include "utils.h"

#include <vector>
#include <torch/extension.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define DEGREE_TO_RADIAN 3.1415926535f / 180.0f

struct Camera {
    float K[4];
    float R[9];
    float t[3];
    float depth_min;
    float depth_max;
    int height;
    int width;
};


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> multi_view_fusion(
    const std::vector<torch::Tensor> &images_list,
    const std::vector<torch::Tensor> &Kvec,
    const std::vector<torch::Tensor> &Rmat,
    const std::vector<torch::Tensor> &tvec,
    const std::vector<torch::Tensor> &depths_list,
    const std::vector<torch::Tensor> &normals_list,
    const std::vector<torch::Tensor> &masks_list,
    const std::vector<std::vector<int>> &pairs_list,
    const int min_num_consistency,
    const float reproj_error_threshold,
    const float relative_depth_threshold,
    const float normal_threshold,
    const int step
);

#endif
