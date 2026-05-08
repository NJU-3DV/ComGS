
#pragma once

#define D2N_BLOCK_X 32
#define D2N_BLOCK_Y 32

#include <torch/extension.h>

namespace d2n{
    std::tuple<torch::Tensor, torch::Tensor> Depth2Normal(
        const torch::Tensor &intr,
        const torch::Tensor &Rc2w,
        const torch::Tensor &depth
    );

    torch::Tensor Depth2NormalBackward(
        const torch::Tensor &intr,
        const torch::Tensor &Rc2w,
        const torch::Tensor &depth,
        const torch::Tensor &sxyz,
        const torch::Tensor &normal,
        const torch::Tensor &dL_dnormal
    );
}
