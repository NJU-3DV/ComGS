// K09
__device__ void load_into_shared_K09(float pixels[SY_K09][SSX_K09], const float *inp, const int CH, const int H, const int W, const int i) {
  auto block = cg::this_thread_block();
  const int batch = block.group_index().z;
  const int start_y = block.group_index().y * BY;
  const int start_x = block.group_index().x * BX;

  const int cnt = SY_K09 * SX_K09;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      int local_y = tid / SX_K09;
      int local_x = tid % SX_K09;
      int y = start_y + local_y;
      int x = start_x + local_x;
      float one = get_pix_value(inp, batch, i, y - 2, x - 2, CH, H, W);
      pixels[local_y][local_x] = one;
    }
  }
}

__device__ void multiply_shared_mem_K09(float pix1[SY_K09][SSX_K09], float pix2[SY_K09][SSX_K09]) {
  auto block = cg::this_thread_block();
  const int cnt = SY_K09 * SX_K09;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      int local_y = tid / SX_K09;
      int local_x = tid % SX_K09;
      float one = pix1[local_y][local_x];
      float two = pix2[local_y][local_x];
      pix1[local_y][local_x] = one * two;
    }
  }
}


__device__ void
flush_conv_scratch_K09(float buf[CY_K09][CCX_K09]) {
  auto block = cg::this_thread_block();
  const int cnt = CY_K09 * CX_K09;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    const int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      const int local_y = tid / CX_K09;
      const int local_x = tid % CX_K09;
      buf[local_y][local_x] = 0.0f;
    }
  }
}

__device__ void do_separable_conv_x_K09(float pixels[SY_K09][SSX_K09], float opt[CY_K09][CCX_K09], int H, int W, bool sq = false) {
  auto block = cg::this_thread_block();

  int local_y = block.thread_index().y;
  int local_x = block.thread_index().x + 1;
  float val = 0.0f;

  if (sq) {
    val += G_K09_00 * do_sq(pixels[local_y][local_x - 1]);
    val += G_K09_01 * do_sq(pixels[local_y][local_x    ]);
    val += G_K09_02 * do_sq(pixels[local_y][local_x + 1]);
  } else {
    val += G_K09_00 * pixels[local_y][local_x - 1];
    val += G_K09_01 * pixels[local_y][local_x    ];
    val += G_K09_02 * pixels[local_y][local_x + 1];
  }
  opt[local_y][local_x] = val;

  val = 0.0f;
  local_y = block.thread_index().y + BY;
  if (local_y < SY_K09) {
    if (sq) {
      val += G_K09_00 * do_sq(pixels[local_y][local_x - 1]);
      val += G_K09_01 * do_sq(pixels[local_y][local_x    ]);
      val += G_K09_02 * do_sq(pixels[local_y][local_x + 1]);
    } else {
      val += G_K09_00 * pixels[local_y][local_x - 1];
      val += G_K09_01 * pixels[local_y][local_x    ];
      val += G_K09_02 * pixels[local_y][local_x + 1];
    }
    opt[local_y][local_x] = val;
  }
}

__device__ float do_separable_conv_y_K09(float pixels[CY_K09][CCX_K09], int H, int W, bool sq = false) {
  auto block = cg::this_thread_block();
  int local_y = block.thread_index().y + 1;
  int local_x = block.thread_index().x + 1;
  float val = 0.0f;

  val += G_K09_00 * pixels[local_y - 1][local_x];
  val += G_K09_01 * pixels[local_y    ][local_x];
  val += G_K09_02 * pixels[local_y + 1][local_x];

  return val;
}

__global__ void fusedssim_K09_CUDA(
  int H, int W, int CH, float C1, float C2,
  float* img1, float* img2, float* ssim_map,
  float* dm_dmu1 = nullptr,
  float* dm_dsigma1_sq = nullptr,
  float* dm_dsigma12 = nullptr
) {
  auto block = cg::this_thread_block();
  const int pix_y = block.group_index().y * BY + block.thread_index().y;
  const int pix_x = block.group_index().x * BX + block.thread_index().x;
  const int pix_id = pix_y * W + pix_x;
  const int num_pix = H * W;
  const int batch = block.group_index().z;

  // shared memory that will be used to load pixels temporarily
  __shared__ float buf1[SY_K09][SSX_K09];
  __shared__ float buf2[SY_K09][SSX_K09];
  __shared__ float buf3[CY_K09][CCX_K09];

  for (int i = 0; i < CH; ++i) {
    // load into shared
    load_into_shared_K09(buf1, img1, CH, H, W, i);
    block.sync();

    // calculate mu1
    flush_conv_scratch_K09(buf3);
    block.sync();
    do_separable_conv_x_K09(buf1, buf3, H, W);
    block.sync();
    float mu1 = do_separable_conv_y_K09(buf3, H, W);
    block.sync();

    // calculate sigma1_sq
    flush_conv_scratch_K09(buf3);
    block.sync();
    do_separable_conv_x_K09(buf1, buf3, H, W, true);
    block.sync();
    float sigma1_sq = do_separable_conv_y_K09(buf3, H, W) - mu1 * mu1;
    block.sync();

    // calculate mu2
    load_into_shared_K09(buf2, img2, CH, H, W, i);
    block.sync();
    flush_conv_scratch_K09(buf3);
    block.sync();
    do_separable_conv_x_K09(buf2, buf3, H, W);
    block.sync();
    float mu2 = do_separable_conv_y_K09(buf3, H, W);
    block.sync();

    // calculate sigma2_sq
    flush_conv_scratch_K09(buf3);
    block.sync();
    do_separable_conv_x_K09(buf2, buf3, H, W, true);
    block.sync();
    float sigma2_sq = do_separable_conv_y_K09(buf3, H, W) - mu2 * mu2;
    block.sync();

    // calculate sigma12
    multiply_shared_mem_K09(buf1, buf2);
    block.sync();
    flush_conv_scratch_K09(buf3);
    block.sync();
    do_separable_conv_x_K09(buf1, buf3, H, W);
    block.sync();
    float sigma12 = do_separable_conv_y_K09(buf3, H, W) - mu1 * mu2;
    block.sync();

    float mu1_sq = mu1 * mu1;
    float mu2_sq = mu2 * mu2;
    float mu1_mu2 = mu1 * mu2;
    float C = (2.0f * mu1_mu2 + C1);
    float D = (2.0f * sigma12 + C2);
    float A = (mu1_sq + mu2_sq + C1);
    float B = (sigma1_sq + sigma2_sq + C2);
    float m = (C * D) / (A * B);
    if (pix_x < W && pix_y < H) {
      const int global_idx = batch * CH * num_pix + i * num_pix + pix_id;
      ssim_map[global_idx] = m;

      if (dm_dmu1) {
        dm_dmu1[global_idx] = (
          (mu2 * 2.0f * D) / (A * B)
          -(mu2 * 2.0f * C) / (A * B)
          -(mu1 * 2.0f * C * D) / ( A * A * B)
          +(mu1 * 2.0f * C * D) / (A * B * B)
        );
        dm_dsigma1_sq[global_idx] = ((-C * D) / (A * B * B));
        dm_dsigma12[global_idx] = ((2 * C) / (A * B));
      }
    }
  }
}

__global__ void fusedssim_K09_backwardCUDA(
  int H, int W, int CH, float C1, float C2,
  float* img1, float* img2,
  float *dL_dmap, float *dL_dimg1,
  float* dm_dmu1 = nullptr,
  float* dm_dsigma1_sq = nullptr,
  float* dm_dsigma12 = nullptr
)
{
  auto block = cg::this_thread_block();
  const int pix_y = block.group_index().y * BY + block.thread_index().y;
  const int pix_x = block.group_index().x * BX + block.thread_index().x;
  const int pix_id = pix_y * W + pix_x;
  const int num_pix = H * W;
  const int batch = block.group_index().z;

  // shared memory that will be used to load pixels temporarily
  __shared__ float buf1[SY_K09][SSX_K09];
  __shared__ float buf2[SY_K09][SSX_K09];
  __shared__ float buf3[CY_K09][CCX_K09];

  for (int i = 0; i < CH; ++i) {
    float dL_dpix = 0.0f;
    float tmp = 0.0f;
    float pix1 = get_pix_value(img1, batch, i, pix_y, pix_x, CH, H, W);
    float pix2 = get_pix_value(img2, batch, i, pix_y, pix_x, CH, H, W);
    load_into_shared_K09(buf1, dL_dmap, CH, H, W, i);

    // gradient from mu1
    load_into_shared_K09(buf2, dm_dmu1, CH, H, W, i);
    block.sync();
    multiply_shared_mem_K09(buf2, buf1);
    block.sync();
    flush_conv_scratch_K09(buf3);
    block.sync();
    do_separable_conv_x_K09(buf2, buf3, H, W);
    block.sync();
    tmp = do_separable_conv_y_K09(buf3, H, W);
    block.sync();
    dL_dpix += tmp;

    // gradient from sigma1_sq
    load_into_shared_K09(buf2, dm_dsigma1_sq, CH, H, W, i);
    block.sync();
    multiply_shared_mem_K09(buf2, buf1);
    block.sync();
    flush_conv_scratch_K09(buf3);
    block.sync();
    do_separable_conv_x_K09(buf2, buf3, H, W);
    block.sync();
    tmp = pix1 * 2.0f * do_separable_conv_y_K09(buf3, H, W);
    block.sync();
    dL_dpix += tmp;

    // gradient from sigma12
    load_into_shared_K09(buf2, dm_dsigma12, CH, H, W, i);
    block.sync();
    multiply_shared_mem_K09(buf2, buf1);
    block.sync();
    flush_conv_scratch_K09(buf3);
    block.sync();
    do_separable_conv_x_K09(buf2, buf3, H, W);
    block.sync();
    tmp = pix2 * do_separable_conv_y_K09(buf3, H, W);
    block.sync();
    dL_dpix += tmp;

    if (pix_x < W && pix_y < H) {
      const int global_idx = batch * CH * num_pix + i * num_pix + pix_id;
      dL_dimg1[global_idx] = dL_dpix;
    }
  }
}