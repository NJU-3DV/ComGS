#include <torch/extension.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <iostream>

namespace cg = cooperative_groups;


#define G_K03_00 0.307801336050033569f
#define G_K03_01 0.384397357702255249f
#define G_K03_02 0.30780133605003356f

#define G_K05_00 0.120078377425670624f
#define G_K05_01 0.233880743384361267f
#define G_K05_02 0.292081713676452637f
#define G_K05_03 0.233880743384361267f
#define G_K05_04 0.120078377425670624f

#define G_K07_00 0.036632847040891647f
#define G_K07_01 0.111280761659145355f
#define G_K07_02 0.216745331883430481f
#define G_K07_03 0.270682156085968018f
#define G_K07_04 0.216745331883430481f
#define G_K07_05 0.111280761659145355f
#define G_K07_06 0.036632847040891647f

#define G_K09_00 0.007614419329911470f
#define G_K09_01 0.036074969917535782f
#define G_K09_02 0.109586082398891449f
#define G_K09_03 0.213444545865058899f
#define G_K09_04 0.266559988260269165f
#define G_K09_05 0.213444545865058899f
#define G_K09_06 0.109586082398891449f
#define G_K09_07 0.036074969917535782f
#define G_K09_08 0.007614419329911470

#define G_K11_00 0.001028380123898387f
#define G_K11_01 0.0075987582094967365f
#define G_K11_02 0.036000773310661316f
#define G_K11_03 0.10936068743467331f
#define G_K11_04 0.21300552785396576f
#define G_K11_05 0.26601171493530273f
#define G_K11_06 0.21300552785396576f
#define G_K11_07 0.10936068743467331f
#define G_K11_08 0.036000773310661316f
#define G_K11_09 0.0075987582094967365f
#define G_K11_10 0.001028380123898387f

// block size
#define BX 32
#define BY 32

// shared memory size
#define SX_K03 (BX + 2)
#define SSX_K03 (BX + 2)
#define SY_K03 (BY + 2)

#define SX_K05 (BX + 4)
#define SSX_K05 (BX + 4)
#define SY_K05 (BY + 4)

#define SX_K07 (BX + 6)
#define SSX_K07 (BX + 6)
#define SY_K07 (BY + 6)

#define SX_K09 (BX + 8)
#define SSX_K09 (BX + 8)  
#define SY_K09 (BY + 8)

#define SX_K11 (BX + 10)
#define SSX_K11 (BX + 10)
#define SY_K11 (BY + 10)

// convolution scratchpad size
#define CX_K03 (BX)
#define CCX_K03 (BX + 0)
#define CY_K03 (BY + 2)

#define CX_K05 (BX)
#define CCX_K05 (BX + 0)
#define CY_K05 (BY + 4)

#define CX_K07 (BX)
#define CCX_K07 (BX + 0)
#define CY_K07 (BY + 6)

#define CX_K09 (BX)
#define CCX_K09 (BX + 0)
#define CY_K09 (BY + 8)

#define CX_K11 (BX)
#define CCX_K11 (BX + 0)
#define CY_K11 (BY + 10)


__device__ float get_pix_value(const float* img, const int b, const int c, const int y, const int x, const int CH, const int H, const int W) {
  if (x >= W || y >= H || x < 0 || y < 0) {
    return 0.0f;
  } else {
    return img[b * CH * H * W + c * H * W + y * W + x];
  }
}

__device__ inline float do_sq(float val) {
  return val * val;
}

// K03
__device__ void load_into_shared_K03(float pixels[SY_K03][SSX_K03], const float *inp, const int CH, const int H, const int W, const int i) {
  auto block = cg::this_thread_block();
  const int batch = block.group_index().z;
  const int start_y = block.group_index().y * BY;
  const int start_x = block.group_index().x * BX;

  const int cnt = SY_K03 * SX_K03;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      int local_y = tid / SX_K03;
      int local_x = tid % SX_K03;
      int y = start_y + local_y;
      int x = start_x + local_x;
      float one = get_pix_value(inp, batch, i, y - 1, x - 1, CH, H, W);
      pixels[local_y][local_x] = one;
    }
  }
}

__device__ void multiply_shared_mem_K03(float pix1[SY_K03][SSX_K03], float pix2[SY_K03][SSX_K03]) {
  auto block = cg::this_thread_block();
  const int cnt = SY_K03 * SX_K03;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      int local_y = tid / SX_K03;
      int local_x = tid % SX_K03;
      float one = pix1[local_y][local_x];
      float two = pix2[local_y][local_x];
      pix1[local_y][local_x] = one * two;
    }
  }
}


__device__ void
flush_conv_scratch_K03(float buf[CY_K03][CCX_K03]) {
  auto block = cg::this_thread_block();
  const int cnt = CY_K03 * CX_K03;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    const int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      const int local_y = tid / CX_K03;
      const int local_x = tid % CX_K03;
      buf[local_y][local_x] = 0.0f;
    }
  }
}

__device__ void do_separable_conv_x_K03(float pixels[SY_K03][SSX_K03], float opt[CY_K03][CCX_K03], int H, int W, bool sq = false) {
  auto block = cg::this_thread_block();

  int local_y = block.thread_index().y;
  int local_x = block.thread_index().x + 1;
  float val = 0.0f;

  if (sq) {
    val += G_K03_00 * do_sq(pixels[local_y][local_x - 1]);
    val += G_K03_01 * do_sq(pixels[local_y][local_x    ]);
    val += G_K03_02 * do_sq(pixels[local_y][local_x + 1]);
  } else {
    val += G_K03_00 * pixels[local_y][local_x - 1];
    val += G_K03_01 * pixels[local_y][local_x    ];
    val += G_K03_02 * pixels[local_y][local_x + 1];
  }
  opt[local_y][local_x - 1] = val;

  val = 0.0f;
  local_y = block.thread_index().y + BY;
  if (local_y < SY_K03) {
    if (sq) {
      val += G_K03_00 * do_sq(pixels[local_y][local_x - 1]);
      val += G_K03_01 * do_sq(pixels[local_y][local_x    ]);
      val += G_K03_02 * do_sq(pixels[local_y][local_x + 1]);
    } else {
      val += G_K03_00 * pixels[local_y][local_x - 1];
      val += G_K03_01 * pixels[local_y][local_x    ];
      val += G_K03_02 * pixels[local_y][local_x + 1];
    }
    opt[local_y][local_x - 1] = val;
  }
}

__device__ float do_separable_conv_y_K03(float pixels[CY_K03][CCX_K03], int H, int W, bool sq = false) {
  auto block = cg::this_thread_block();
  int local_y = block.thread_index().y + 1;
  int local_x = block.thread_index().x;
  float val = 0.0f;

  val += G_K03_00 * pixels[local_y - 1][local_x];
  val += G_K03_01 * pixels[local_y    ][local_x];
  val += G_K03_02 * pixels[local_y + 1][local_x];

  return val;
}

__global__ void fusedssim_K03_CUDA(
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
  __shared__ float buf1[SY_K03][SSX_K03];
  __shared__ float buf2[SY_K03][SSX_K03];
  __shared__ float buf3[CY_K03][CCX_K03];

  for (int i = 0; i < CH; ++i) {
    // load into shared
    load_into_shared_K03(buf1, img1, CH, H, W, i);
    block.sync();

    // calculate mu1
    flush_conv_scratch_K03(buf3);
    block.sync();
    do_separable_conv_x_K03(buf1, buf3, H, W);
    block.sync();
    float mu1 = do_separable_conv_y_K03(buf3, H, W);
    block.sync();

    // calculate sigma1_sq
    flush_conv_scratch_K03(buf3);
    block.sync();
    do_separable_conv_x_K03(buf1, buf3, H, W, true);
    block.sync();
    float sigma1_sq = do_separable_conv_y_K03(buf3, H, W) - mu1 * mu1;
    block.sync();

    // calculate mu2
    load_into_shared_K03(buf2, img2, CH, H, W, i);
    block.sync();
    flush_conv_scratch_K03(buf3);
    block.sync();
    do_separable_conv_x_K03(buf2, buf3, H, W);
    block.sync();
    float mu2 = do_separable_conv_y_K03(buf3, H, W);
    block.sync();

    // calculate sigma2_sq
    flush_conv_scratch_K03(buf3);
    block.sync();
    do_separable_conv_x_K03(buf2, buf3, H, W, true);
    block.sync();
    float sigma2_sq = do_separable_conv_y_K03(buf3, H, W) - mu2 * mu2;
    block.sync();

    // calculate sigma12
    multiply_shared_mem_K03(buf1, buf2);
    block.sync();
    flush_conv_scratch_K03(buf3);
    block.sync();
    do_separable_conv_x_K03(buf1, buf3, H, W);
    block.sync();
    float sigma12 = do_separable_conv_y_K03(buf3, H, W) - mu1 * mu2;
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

__global__ void fusedssim_K03_backwardCUDA(
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
  __shared__ float buf1[SY_K03][SSX_K03];
  __shared__ float buf2[SY_K03][SSX_K03];
  __shared__ float buf3[CY_K03][CCX_K03];

  for (int i = 0; i < CH; ++i) {
    float dL_dpix = 0.0f;
    float tmp = 0.0f;
    float pix1 = get_pix_value(img1, batch, i, pix_y, pix_x, CH, H, W);
    float pix2 = get_pix_value(img2, batch, i, pix_y, pix_x, CH, H, W);
    load_into_shared_K03(buf1, dL_dmap, CH, H, W, i);

    // gradient from mu1
    load_into_shared_K03(buf2, dm_dmu1, CH, H, W, i);
    block.sync();
    multiply_shared_mem_K03(buf2, buf1);
    block.sync();
    flush_conv_scratch_K03(buf3);
    block.sync();
    do_separable_conv_x_K03(buf2, buf3, H, W);
    block.sync();
    tmp = do_separable_conv_y_K03(buf3, H, W);
    block.sync();
    dL_dpix += tmp;

    // gradient from sigma1_sq
    load_into_shared_K03(buf2, dm_dsigma1_sq, CH, H, W, i);
    block.sync();
    multiply_shared_mem_K03(buf2, buf1);
    block.sync();
    flush_conv_scratch_K03(buf3);
    block.sync();
    do_separable_conv_x_K03(buf2, buf3, H, W);
    block.sync();
    tmp = pix1 * 2.0f * do_separable_conv_y_K03(buf3, H, W);
    block.sync();
    dL_dpix += tmp;

    // gradient from sigma12
    load_into_shared_K03(buf2, dm_dsigma12, CH, H, W, i);
    block.sync();
    multiply_shared_mem_K03(buf2, buf1);
    block.sync();
    flush_conv_scratch_K03(buf3);
    block.sync();
    do_separable_conv_x_K03(buf2, buf3, H, W);
    block.sync();
    tmp = pix2 * do_separable_conv_y_K03(buf3, H, W);
    block.sync();
    dL_dpix += tmp;

    if (pix_x < W && pix_y < H) {
      const int global_idx = batch * CH * num_pix + i * num_pix + pix_id;
      dL_dimg1[global_idx] = dL_dpix;
    }
  }
}

// K05
__device__ void load_into_shared_K05(float pixels[SY_K05][SSX_K05], const float *inp, const int CH, const int H, const int W, const int i) {
  auto block = cg::this_thread_block();
  const int batch = block.group_index().z;
  const int start_y = block.group_index().y * BY;
  const int start_x = block.group_index().x * BX;

  const int cnt = SY_K05 * SX_K05;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      int local_y = tid / SX_K05;
      int local_x = tid % SX_K05;
      int y = start_y + local_y;
      int x = start_x + local_x;
      float one = get_pix_value(inp, batch, i, y - 2, x - 2, CH, H, W);
      pixels[local_y][local_x] = one;
    }
  }
}

__device__ void multiply_shared_mem_K05(float pix1[SY_K05][SSX_K05], float pix2[SY_K05][SSX_K05]) {
  auto block = cg::this_thread_block();
  const int cnt = SY_K05 * SX_K05;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      int local_y = tid / SX_K05;
      int local_x = tid % SX_K05;
      float one = pix1[local_y][local_x];
      float two = pix2[local_y][local_x];
      pix1[local_y][local_x] = one * two;
    }
  }
}


__device__ void
flush_conv_scratch_K05(float buf[CY_K05][CCX_K05]) {
  auto block = cg::this_thread_block();
  const int cnt = CY_K05 * CX_K05;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    const int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      const int local_y = tid / CX_K05;
      const int local_x = tid % CX_K05;
      buf[local_y][local_x] = 0.0f;
    }
  }
}

__device__ void do_separable_conv_x_K05(float pixels[SY_K05][SSX_K05], float opt[CY_K05][CCX_K05], int H, int W, bool sq = false) {
  auto block = cg::this_thread_block();

  int local_y = block.thread_index().y;
  int local_x = block.thread_index().x + 2;
  float val = 0.0f;

  if (sq) {
    val += G_K05_00 * do_sq(pixels[local_y][local_x - 2]);
    val += G_K05_01 * do_sq(pixels[local_y][local_x - 1]);
    val += G_K05_02 * do_sq(pixels[local_y][local_x    ]);
    val += G_K05_03 * do_sq(pixels[local_y][local_x + 1]);
    val += G_K05_04 * do_sq(pixels[local_y][local_x + 2]);
  } else {
    val += G_K05_00 * pixels[local_y][local_x - 2];
    val += G_K05_01 * pixels[local_y][local_x - 1];
    val += G_K05_02 * pixels[local_y][local_x    ];
    val += G_K05_03 * pixels[local_y][local_x + 1];
    val += G_K05_04 * pixels[local_y][local_x + 2];
  }
  opt[local_y][local_x - 2] = val;

  val = 0.0f;
  local_y = block.thread_index().y + BY;
  if (local_y < SY_K05) {
    if (sq) {
      val += G_K05_00 * do_sq(pixels[local_y][local_x - 2]);
      val += G_K05_01 * do_sq(pixels[local_y][local_x - 1]);
      val += G_K05_02 * do_sq(pixels[local_y][local_x    ]);
      val += G_K05_03 * do_sq(pixels[local_y][local_x + 1]);
      val += G_K05_04 * do_sq(pixels[local_y][local_x + 2]);
    } else {
      val += G_K05_00 * pixels[local_y][local_x - 2];
      val += G_K05_01 * pixels[local_y][local_x - 1];
      val += G_K05_02 * pixels[local_y][local_x    ];
      val += G_K05_03 * pixels[local_y][local_x + 1];
      val += G_K05_04 * pixels[local_y][local_x + 2];
    }
    opt[local_y][local_x - 2] = val;
  }
}

__device__ float do_separable_conv_y_K05(float pixels[CY_K05][CCX_K05], int H, int W, bool sq = false) {
  auto block = cg::this_thread_block();
  int local_y = block.thread_index().y + 2;
  int local_x = block.thread_index().x;
  float val = 0.0f;

  val += G_K05_00 * pixels[local_y - 2][local_x];
  val += G_K05_01 * pixels[local_y - 1][local_x];
  val += G_K05_02 * pixels[local_y    ][local_x];
  val += G_K05_03 * pixels[local_y + 1][local_x];
  val += G_K05_04 * pixels[local_y + 2][local_x];

  return val;
}

__global__ void fusedssim_K05_CUDA(
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
  __shared__ float buf1[SY_K05][SSX_K05];
  __shared__ float buf2[SY_K05][SSX_K05];
  __shared__ float buf3[CY_K05][CCX_K05];

  for (int i = 0; i < CH; ++i) {
    // load into shared
    load_into_shared_K05(buf1, img1, CH, H, W, i);
    block.sync();

    // calculate mu1
    flush_conv_scratch_K05(buf3);
    block.sync();
    do_separable_conv_x_K05(buf1, buf3, H, W);
    block.sync();
    float mu1 = do_separable_conv_y_K05(buf3, H, W);
    block.sync();

    // calculate sigma1_sq
    flush_conv_scratch_K05(buf3);
    block.sync();
    do_separable_conv_x_K05(buf1, buf3, H, W, true);
    block.sync();
    float sigma1_sq = do_separable_conv_y_K05(buf3, H, W) - mu1 * mu1;
    block.sync();

    // calculate mu2
    load_into_shared_K05(buf2, img2, CH, H, W, i);
    block.sync();
    flush_conv_scratch_K05(buf3);
    block.sync();
    do_separable_conv_x_K05(buf2, buf3, H, W);
    block.sync();
    float mu2 = do_separable_conv_y_K05(buf3, H, W);
    block.sync();

    // calculate sigma2_sq
    flush_conv_scratch_K05(buf3);
    block.sync();
    do_separable_conv_x_K05(buf2, buf3, H, W, true);
    block.sync();
    float sigma2_sq = do_separable_conv_y_K05(buf3, H, W) - mu2 * mu2;
    block.sync();

    // calculate sigma12
    multiply_shared_mem_K05(buf1, buf2);
    block.sync();
    flush_conv_scratch_K05(buf3);
    block.sync();
    do_separable_conv_x_K05(buf1, buf3, H, W);
    block.sync();
    float sigma12 = do_separable_conv_y_K05(buf3, H, W) - mu1 * mu2;
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

__global__ void fusedssim_K05_backwardCUDA(
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
  __shared__ float buf1[SY_K05][SSX_K05];
  __shared__ float buf2[SY_K05][SSX_K05];
  __shared__ float buf3[CY_K05][CCX_K05];

  for (int i = 0; i < CH; ++i) {
    float dL_dpix = 0.0f;
    float tmp = 0.0f;
    float pix1 = get_pix_value(img1, batch, i, pix_y, pix_x, CH, H, W);
    float pix2 = get_pix_value(img2, batch, i, pix_y, pix_x, CH, H, W);
    load_into_shared_K05(buf1, dL_dmap, CH, H, W, i);

    // gradient from mu1
    load_into_shared_K05(buf2, dm_dmu1, CH, H, W, i);
    block.sync();
    multiply_shared_mem_K05(buf2, buf1);
    block.sync();
    flush_conv_scratch_K05(buf3);
    block.sync();
    do_separable_conv_x_K05(buf2, buf3, H, W);
    block.sync();
    tmp = do_separable_conv_y_K05(buf3, H, W);
    block.sync();
    dL_dpix += tmp;

    // gradient from sigma1_sq
    load_into_shared_K05(buf2, dm_dsigma1_sq, CH, H, W, i);
    block.sync();
    multiply_shared_mem_K05(buf2, buf1);
    block.sync();
    flush_conv_scratch_K05(buf3);
    block.sync();
    do_separable_conv_x_K05(buf2, buf3, H, W);
    block.sync();
    tmp = pix1 * 2.0f * do_separable_conv_y_K05(buf3, H, W);
    block.sync();
    dL_dpix += tmp;

    // gradient from sigma12
    load_into_shared_K05(buf2, dm_dsigma12, CH, H, W, i);
    block.sync();
    multiply_shared_mem_K05(buf2, buf1);
    block.sync();
    flush_conv_scratch_K05(buf3);
    block.sync();
    do_separable_conv_x_K05(buf2, buf3, H, W);
    block.sync();
    tmp = pix2 * do_separable_conv_y_K05(buf3, H, W);
    block.sync();
    dL_dpix += tmp;

    if (pix_x < W && pix_y < H) {
      const int global_idx = batch * CH * num_pix + i * num_pix + pix_id;
      dL_dimg1[global_idx] = dL_dpix;
    }
  }
}

// K07
__device__ void load_into_shared_K07(float pixels[SY_K07][SSX_K07], const float *inp, const int CH, const int H, const int W, const int i) {
  auto block = cg::this_thread_block();
  const int batch = block.group_index().z;
  const int start_y = block.group_index().y * BY;
  const int start_x = block.group_index().x * BX;

  const int cnt = SY_K07 * SX_K07;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      int local_y = tid / SX_K07;
      int local_x = tid % SX_K07;
      int y = start_y + local_y;
      int x = start_x + local_x;
      float one = get_pix_value(inp, batch, i, y - 3, x - 3, CH, H, W);
      pixels[local_y][local_x] = one;
    }
  }
}

__device__ void multiply_shared_mem_K07(float pix1[SY_K07][SSX_K07], float pix2[SY_K07][SSX_K07]) {
  auto block = cg::this_thread_block();
  const int cnt = SY_K07 * SX_K07;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      int local_y = tid / SX_K07;
      int local_x = tid % SX_K07;
      float one = pix1[local_y][local_x];
      float two = pix2[local_y][local_x];
      pix1[local_y][local_x] = one * two;
    }
  }
}


__device__ void
flush_conv_scratch_K07(float buf[CY_K07][CCX_K07]) {
  auto block = cg::this_thread_block();
  const int cnt = CY_K07 * CX_K07;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    const int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      const int local_y = tid / CX_K07;
      const int local_x = tid % CX_K07;
      buf[local_y][local_x] = 0.0f;
    }
  }
}

__device__ void do_separable_conv_x_K07(float pixels[SY_K07][SSX_K07], float opt[CY_K07][CCX_K07], int H, int W, bool sq = false) {
  auto block = cg::this_thread_block();

  int local_y = block.thread_index().y;
  int local_x = block.thread_index().x + 3;
  float val = 0.0f;

  if (sq) {
    val += G_K07_00 * do_sq(pixels[local_y][local_x - 3]);
    val += G_K07_01 * do_sq(pixels[local_y][local_x - 2]);
    val += G_K07_02 * do_sq(pixels[local_y][local_x - 1]);
    val += G_K07_03 * do_sq(pixels[local_y][local_x    ]);
    val += G_K07_04 * do_sq(pixels[local_y][local_x + 1]);
    val += G_K07_05 * do_sq(pixels[local_y][local_x + 2]);
    val += G_K07_06 * do_sq(pixels[local_y][local_x + 3]);
  } else {
    val += G_K07_00 * pixels[local_y][local_x - 3];
    val += G_K07_01 * pixels[local_y][local_x - 2];
    val += G_K07_02 * pixels[local_y][local_x - 1];
    val += G_K07_03 * pixels[local_y][local_x    ];
    val += G_K07_04 * pixels[local_y][local_x + 1];
    val += G_K07_05 * pixels[local_y][local_x + 2];
    val += G_K07_06 * pixels[local_y][local_x + 3];
  }
  opt[local_y][local_x - 3] = val;

  val = 0.0f;
  local_y = block.thread_index().y + BY;
  if (local_y < SY_K07) {
    if (sq) {
      val += G_K07_00 * do_sq(pixels[local_y][local_x - 3]);
      val += G_K07_01 * do_sq(pixels[local_y][local_x - 2]);
      val += G_K07_02 * do_sq(pixels[local_y][local_x - 1]);
      val += G_K07_03 * do_sq(pixels[local_y][local_x    ]);
      val += G_K07_04 * do_sq(pixels[local_y][local_x + 1]);
      val += G_K07_05 * do_sq(pixels[local_y][local_x + 2]);
      val += G_K07_06 * do_sq(pixels[local_y][local_x + 3]);
    } else {
      val += G_K07_00 * pixels[local_y][local_x - 3];
      val += G_K07_01 * pixels[local_y][local_x - 2];
      val += G_K07_02 * pixels[local_y][local_x - 1];
      val += G_K07_03 * pixels[local_y][local_x    ];
      val += G_K07_04 * pixels[local_y][local_x + 1];
      val += G_K07_05 * pixels[local_y][local_x + 2];
      val += G_K07_06 * pixels[local_y][local_x + 3];
    }
    opt[local_y][local_x - 3] = val;
  }
}

__device__ float do_separable_conv_y_K07(float pixels[CY_K07][CCX_K07], int H, int W, bool sq = false) {
  auto block = cg::this_thread_block();
  int local_y = block.thread_index().y + 3;
  int local_x = block.thread_index().x;
  float val = 0.0f;

  val += G_K07_00 * pixels[local_y - 3][local_x];
  val += G_K07_01 * pixels[local_y - 2][local_x];
  val += G_K07_02 * pixels[local_y - 1][local_x];
  val += G_K07_03 * pixels[local_y    ][local_x];
  val += G_K07_04 * pixels[local_y + 1][local_x];
  val += G_K07_05 * pixels[local_y + 2][local_x];
  val += G_K07_06 * pixels[local_y + 3][local_x];

  return val;
}

__global__ void fusedssim_K07_CUDA(
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
  __shared__ float buf1[SY_K07][SSX_K07];
  __shared__ float buf2[SY_K07][SSX_K07];
  __shared__ float buf3[CY_K07][CCX_K07];

  for (int i = 0; i < CH; ++i) {
    // load into shared
    load_into_shared_K07(buf1, img1, CH, H, W, i);
    block.sync();

    // calculate mu1
    flush_conv_scratch_K07(buf3);
    block.sync();
    do_separable_conv_x_K07(buf1, buf3, H, W);
    block.sync();
    float mu1 = do_separable_conv_y_K07(buf3, H, W);
    block.sync();

    // calculate sigma1_sq
    flush_conv_scratch_K07(buf3);
    block.sync();
    do_separable_conv_x_K07(buf1, buf3, H, W, true);
    block.sync();
    float sigma1_sq = do_separable_conv_y_K07(buf3, H, W) - mu1 * mu1;
    block.sync();

    // calculate mu2
    load_into_shared_K07(buf2, img2, CH, H, W, i);
    block.sync();
    flush_conv_scratch_K07(buf3);
    block.sync();
    do_separable_conv_x_K07(buf2, buf3, H, W);
    block.sync();
    float mu2 = do_separable_conv_y_K07(buf3, H, W);
    block.sync();

    // calculate sigma2_sq
    flush_conv_scratch_K07(buf3);
    block.sync();
    do_separable_conv_x_K07(buf2, buf3, H, W, true);
    block.sync();
    float sigma2_sq = do_separable_conv_y_K07(buf3, H, W) - mu2 * mu2;
    block.sync();

    // calculate sigma12
    multiply_shared_mem_K07(buf1, buf2);
    block.sync();
    flush_conv_scratch_K07(buf3);
    block.sync();
    do_separable_conv_x_K07(buf1, buf3, H, W);
    block.sync();
    float sigma12 = do_separable_conv_y_K07(buf3, H, W) - mu1 * mu2;
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

__global__ void fusedssim_K07_backwardCUDA(
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
  __shared__ float buf1[SY_K07][SSX_K07];
  __shared__ float buf2[SY_K07][SSX_K07];
  __shared__ float buf3[CY_K07][CCX_K07];

  for (int i = 0; i < CH; ++i) {
    float dL_dpix = 0.0f;
    float tmp = 0.0f;
    float pix1 = get_pix_value(img1, batch, i, pix_y, pix_x, CH, H, W);
    float pix2 = get_pix_value(img2, batch, i, pix_y, pix_x, CH, H, W);
    load_into_shared_K07(buf1, dL_dmap, CH, H, W, i);

    // gradient from mu1
    load_into_shared_K07(buf2, dm_dmu1, CH, H, W, i);
    block.sync();
    multiply_shared_mem_K07(buf2, buf1);
    block.sync();
    flush_conv_scratch_K07(buf3);
    block.sync();
    do_separable_conv_x_K07(buf2, buf3, H, W);
    block.sync();
    tmp = do_separable_conv_y_K07(buf3, H, W);
    block.sync();
    dL_dpix += tmp;

    // gradient from sigma1_sq
    load_into_shared_K07(buf2, dm_dsigma1_sq, CH, H, W, i);
    block.sync();
    multiply_shared_mem_K07(buf2, buf1);
    block.sync();
    flush_conv_scratch_K07(buf3);
    block.sync();
    do_separable_conv_x_K07(buf2, buf3, H, W);
    block.sync();
    tmp = pix1 * 2.0f * do_separable_conv_y_K07(buf3, H, W);
    block.sync();
    dL_dpix += tmp;

    // gradient from sigma12
    load_into_shared_K07(buf2, dm_dsigma12, CH, H, W, i);
    block.sync();
    multiply_shared_mem_K07(buf2, buf1);
    block.sync();
    flush_conv_scratch_K07(buf3);
    block.sync();
    do_separable_conv_x_K07(buf2, buf3, H, W);
    block.sync();
    tmp = pix2 * do_separable_conv_y_K07(buf3, H, W);
    block.sync();
    dL_dpix += tmp;

    if (pix_x < W && pix_y < H) {
      const int global_idx = batch * CH * num_pix + i * num_pix + pix_id;
      dL_dimg1[global_idx] = dL_dpix;
    }
  }
}

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
      float one = get_pix_value(inp, batch, i, y - 4, x - 4, CH, H, W);
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
  int local_x = block.thread_index().x + 4;
  float val = 0.0f;

  if (sq) {
    val += G_K09_00 * do_sq(pixels[local_y][local_x - 4]);
    val += G_K09_01 * do_sq(pixels[local_y][local_x - 3]);
    val += G_K09_02 * do_sq(pixels[local_y][local_x - 2]);
    val += G_K09_03 * do_sq(pixels[local_y][local_x - 1]);
    val += G_K09_04 * do_sq(pixels[local_y][local_x    ]);
    val += G_K09_05 * do_sq(pixels[local_y][local_x + 1]);
    val += G_K09_06 * do_sq(pixels[local_y][local_x + 2]);
    val += G_K09_07 * do_sq(pixels[local_y][local_x + 3]);
    val += G_K09_08 * do_sq(pixels[local_y][local_x + 4]);
  } else {
    val += G_K09_00 * pixels[local_y][local_x - 4];
    val += G_K09_01 * pixels[local_y][local_x - 3];
    val += G_K09_02 * pixels[local_y][local_x - 2];
    val += G_K09_03 * pixels[local_y][local_x - 1];
    val += G_K09_04 * pixels[local_y][local_x    ];
    val += G_K09_05 * pixels[local_y][local_x + 1];
    val += G_K09_06 * pixels[local_y][local_x + 2];
    val += G_K09_07 * pixels[local_y][local_x + 3];
    val += G_K09_08 * pixels[local_y][local_x + 4];
  }
  opt[local_y][local_x - 4] = val;

  val = 0.0f;
  local_y = block.thread_index().y + BY;
  if (local_y < SY_K09) {
    if (sq) {
      val += G_K09_00 * do_sq(pixels[local_y][local_x - 4]);
      val += G_K09_01 * do_sq(pixels[local_y][local_x - 3]);
      val += G_K09_02 * do_sq(pixels[local_y][local_x - 2]);
      val += G_K09_03 * do_sq(pixels[local_y][local_x - 1]);
      val += G_K09_04 * do_sq(pixels[local_y][local_x    ]);
      val += G_K09_05 * do_sq(pixels[local_y][local_x + 1]);
      val += G_K09_06 * do_sq(pixels[local_y][local_x + 2]);
      val += G_K09_07 * do_sq(pixels[local_y][local_x + 3]);
      val += G_K09_08 * do_sq(pixels[local_y][local_x + 4]);
    } else {
      val += G_K09_00 * pixels[local_y][local_x - 4];
      val += G_K09_01 * pixels[local_y][local_x - 3];
      val += G_K09_02 * pixels[local_y][local_x - 2];
      val += G_K09_03 * pixels[local_y][local_x - 1];
      val += G_K09_04 * pixels[local_y][local_x    ];
      val += G_K09_05 * pixels[local_y][local_x + 1];
      val += G_K09_06 * pixels[local_y][local_x + 2];
      val += G_K09_07 * pixels[local_y][local_x + 3];
      val += G_K09_08 * pixels[local_y][local_x + 4];
    }
    opt[local_y][local_x - 4] = val;
  }
}

__device__ float do_separable_conv_y_K09(float pixels[CY_K09][CCX_K09], int H, int W, bool sq = false) {
  auto block = cg::this_thread_block();
  int local_y = block.thread_index().y + 4;
  int local_x = block.thread_index().x;
  float val = 0.0f;

  val += G_K09_00 * pixels[local_y - 4][local_x];
  val += G_K09_01 * pixels[local_y - 3][local_x];
  val += G_K09_02 * pixels[local_y - 2][local_x];
  val += G_K09_03 * pixels[local_y - 1][local_x];
  val += G_K09_04 * pixels[local_y    ][local_x];
  val += G_K09_05 * pixels[local_y + 1][local_x];
  val += G_K09_06 * pixels[local_y + 2][local_x];
  val += G_K09_07 * pixels[local_y + 3][local_x];
  val += G_K09_08 * pixels[local_y + 4][local_x];

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

// K11
__device__ void load_into_shared_K11(float pixels[SY_K11][SSX_K11], const float *inp, const int CH, const int H, const int W, const int i) {
  auto block = cg::this_thread_block();
  const int batch = block.group_index().z;
  const int start_y = block.group_index().y * BY;
  const int start_x = block.group_index().x * BX;

  const int cnt = SY_K11 * SX_K11;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      int local_y = tid / SX_K11;
      int local_x = tid % SX_K11;
      int y = start_y + local_y;
      int x = start_x + local_x;
      float one = get_pix_value(inp, batch, i, y - 5, x - 5, CH, H, W);
      pixels[local_y][local_x] = one;
    }
  }
}

__device__ void multiply_shared_mem_K11(float pix1[SY_K11][SSX_K11], float pix2[SY_K11][SSX_K11]) {
  auto block = cg::this_thread_block();
  const int cnt = SY_K11 * SX_K11;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      int local_y = tid / SX_K11;
      int local_x = tid % SX_K11;
      float one = pix1[local_y][local_x];
      float two = pix2[local_y][local_x];
      pix1[local_y][local_x] = one * two;
    }
  }
}


__device__ void
flush_conv_scratch_K11(float buf[CY_K11][CCX_K11]) {
  auto block = cg::this_thread_block();
  const int cnt = CY_K11 * CX_K11;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    const int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      const int local_y = tid / CX_K11;
      const int local_x = tid % CX_K11;
      buf[local_y][local_x] = 0.0f;
    }
  }
}

__device__ void do_separable_conv_x_K11(float pixels[SY_K11][SSX_K11], float opt[CY_K11][CCX_K11], int H, int W, bool sq = false) {
  auto block = cg::this_thread_block();

  int local_y = block.thread_index().y;
  int local_x = block.thread_index().x + 5;
  float val = 0.0f;

  if (sq) {
    val += G_K11_00 * do_sq(pixels[local_y][local_x - 5]);
    val += G_K11_01 * do_sq(pixels[local_y][local_x - 4]);
    val += G_K11_02 * do_sq(pixels[local_y][local_x - 3]);
    val += G_K11_03 * do_sq(pixels[local_y][local_x - 2]);
    val += G_K11_04 * do_sq(pixels[local_y][local_x - 1]);
    val += G_K11_05 * do_sq(pixels[local_y][local_x    ]);
    val += G_K11_06 * do_sq(pixels[local_y][local_x + 1]);
    val += G_K11_07 * do_sq(pixels[local_y][local_x + 2]);
    val += G_K11_08 * do_sq(pixels[local_y][local_x + 3]);
    val += G_K11_09 * do_sq(pixels[local_y][local_x + 4]);
    val += G_K11_10 * do_sq(pixels[local_y][local_x + 5]);
  } else {
    val += G_K11_00 * pixels[local_y][local_x - 5];
    val += G_K11_01 * pixels[local_y][local_x - 4];
    val += G_K11_02 * pixels[local_y][local_x - 3];
    val += G_K11_03 * pixels[local_y][local_x - 2];
    val += G_K11_04 * pixels[local_y][local_x - 1];
    val += G_K11_05 * pixels[local_y][local_x    ];
    val += G_K11_06 * pixels[local_y][local_x + 1];
    val += G_K11_07 * pixels[local_y][local_x + 2];
    val += G_K11_08 * pixels[local_y][local_x + 3];
    val += G_K11_09 * pixels[local_y][local_x + 4];
    val += G_K11_10 * pixels[local_y][local_x + 5];
  }
  opt[local_y][local_x - 5] = val;

  val = 0.0f;
  local_y = block.thread_index().y + BY;
  if (local_y < SY_K11) {
    if (sq) {
      val += G_K11_00 * do_sq(pixels[local_y][local_x - 5]);
      val += G_K11_01 * do_sq(pixels[local_y][local_x - 4]);
      val += G_K11_02 * do_sq(pixels[local_y][local_x - 3]);
      val += G_K11_03 * do_sq(pixels[local_y][local_x - 2]);
      val += G_K11_04 * do_sq(pixels[local_y][local_x - 1]);
      val += G_K11_05 * do_sq(pixels[local_y][local_x    ]);
      val += G_K11_06 * do_sq(pixels[local_y][local_x + 1]);
      val += G_K11_07 * do_sq(pixels[local_y][local_x + 2]);
      val += G_K11_08 * do_sq(pixels[local_y][local_x + 3]);
      val += G_K11_09 * do_sq(pixels[local_y][local_x + 4]);
      val += G_K11_10 * do_sq(pixels[local_y][local_x + 5]);
    } else {
      val += G_K11_00 * pixels[local_y][local_x - 5];
      val += G_K11_01 * pixels[local_y][local_x - 4];
      val += G_K11_02 * pixels[local_y][local_x - 3];
      val += G_K11_03 * pixels[local_y][local_x - 2];
      val += G_K11_04 * pixels[local_y][local_x - 1];
      val += G_K11_05 * pixels[local_y][local_x    ];
      val += G_K11_06 * pixels[local_y][local_x + 1];
      val += G_K11_07 * pixels[local_y][local_x + 2];
      val += G_K11_08 * pixels[local_y][local_x + 3];
      val += G_K11_09 * pixels[local_y][local_x + 4];
      val += G_K11_10 * pixels[local_y][local_x + 5];
    }
    opt[local_y][local_x - 5] = val;
  }
}

__device__ float do_separable_conv_y_K11(float pixels[CY_K11][CCX_K11], int H, int W, bool sq = false) {
  auto block = cg::this_thread_block();
  int local_y = block.thread_index().y + 5;
  int local_x = block.thread_index().x;
  float val = 0.0f;

  val += G_K11_00 * pixels[local_y - 5][local_x];
  val += G_K11_01 * pixels[local_y - 4][local_x];
  val += G_K11_02 * pixels[local_y - 3][local_x];
  val += G_K11_03 * pixels[local_y - 2][local_x];
  val += G_K11_04 * pixels[local_y - 1][local_x];
  val += G_K11_05 * pixels[local_y    ][local_x];
  val += G_K11_06 * pixels[local_y + 1][local_x];
  val += G_K11_07 * pixels[local_y + 2][local_x];
  val += G_K11_08 * pixels[local_y + 3][local_x];
  val += G_K11_09 * pixels[local_y + 4][local_x];
  val += G_K11_10 * pixels[local_y + 5][local_x];

  return val;
}

__global__ void fusedssim_K11_CUDA(
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
  __shared__ float buf1[SY_K11][SSX_K11];
  __shared__ float buf2[SY_K11][SSX_K11];
  __shared__ float buf3[CY_K11][CCX_K11];

  for (int i = 0; i < CH; ++i) {
    // load into shared
    load_into_shared_K11(buf1, img1, CH, H, W, i);
    block.sync();

    // calculate mu1
    flush_conv_scratch_K11(buf3);
    block.sync();
    do_separable_conv_x_K11(buf1, buf3, H, W);
    block.sync();
    float mu1 = do_separable_conv_y_K11(buf3, H, W);
    block.sync();

    // calculate sigma1_sq
    flush_conv_scratch_K11(buf3);
    block.sync();
    do_separable_conv_x_K11(buf1, buf3, H, W, true);
    block.sync();
    float sigma1_sq = do_separable_conv_y_K11(buf3, H, W) - mu1 * mu1;
    block.sync();

    // calculate mu2
    load_into_shared_K11(buf2, img2, CH, H, W, i);
    block.sync();
    flush_conv_scratch_K11(buf3);
    block.sync();
    do_separable_conv_x_K11(buf2, buf3, H, W);
    block.sync();
    float mu2 = do_separable_conv_y_K11(buf3, H, W);
    block.sync();

    // calculate sigma2_sq
    flush_conv_scratch_K11(buf3);
    block.sync();
    do_separable_conv_x_K11(buf2, buf3, H, W, true);
    block.sync();
    float sigma2_sq = do_separable_conv_y_K11(buf3, H, W) - mu2 * mu2;
    block.sync();

    // calculate sigma12
    multiply_shared_mem_K11(buf1, buf2);
    block.sync();
    flush_conv_scratch_K11(buf3);
    block.sync();
    do_separable_conv_x_K11(buf1, buf3, H, W);
    block.sync();
    float sigma12 = do_separable_conv_y_K11(buf3, H, W) - mu1 * mu2;
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

__global__ void fusedssim_K11_backwardCUDA(
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
  __shared__ float buf1[SY_K11][SSX_K11];
  __shared__ float buf2[SY_K11][SSX_K11];
  __shared__ float buf3[CY_K11][CCX_K11];

  for (int i = 0; i < CH; ++i) {
    float dL_dpix = 0.0f;
    float tmp = 0.0f;
    float pix1 = get_pix_value(img1, batch, i, pix_y, pix_x, CH, H, W);
    float pix2 = get_pix_value(img2, batch, i, pix_y, pix_x, CH, H, W);
    load_into_shared_K11(buf1, dL_dmap, CH, H, W, i);

    // gradient from mu1
    load_into_shared_K11(buf2, dm_dmu1, CH, H, W, i);
    block.sync();
    multiply_shared_mem_K11(buf2, buf1);
    block.sync();
    flush_conv_scratch_K11(buf3);
    block.sync();
    do_separable_conv_x_K11(buf2, buf3, H, W);
    block.sync();
    tmp = do_separable_conv_y_K11(buf3, H, W);
    block.sync();
    dL_dpix += tmp;

    // gradient from sigma1_sq
    load_into_shared_K11(buf2, dm_dsigma1_sq, CH, H, W, i);
    block.sync();
    multiply_shared_mem_K11(buf2, buf1);
    block.sync();
    flush_conv_scratch_K11(buf3);
    block.sync();
    do_separable_conv_x_K11(buf2, buf3, H, W);
    block.sync();
    tmp = pix1 * 2.0f * do_separable_conv_y_K11(buf3, H, W);
    block.sync();
    dL_dpix += tmp;

    // gradient from sigma12
    load_into_shared_K11(buf2, dm_dsigma12, CH, H, W, i);
    block.sync();
    multiply_shared_mem_K11(buf2, buf1);
    block.sync();
    flush_conv_scratch_K11(buf3);
    block.sync();
    do_separable_conv_x_K11(buf2, buf3, H, W);
    block.sync();
    tmp = pix2 * do_separable_conv_y_K11(buf3, H, W);
    block.sync();
    dL_dpix += tmp;

    if (pix_x < W && pix_y < H) {
      const int global_idx = batch * CH * num_pix + i * num_pix + pix_id;
      dL_dimg1[global_idx] = dL_dpix;
    }
  }
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
fusedssim(
  const float C1,
  const float C2,
  const torch::Tensor &img1,
  const torch::Tensor &img2,
  const int window_size,
  const bool train
) {
  int B = img1.size(0);
  int CH = img1.size(1);
  int H = img1.size(2);
  int W = img1.size(3);
  dim3 grid((W + BX - 1) / BX, (H + BY - 1) / BY, B);
  dim3 block(BX, BY, 1);

  torch::Tensor target = torch::zeros_like(img1).contiguous();
  torch::Tensor dm_dmu1 = train ? torch::zeros_like(img1).contiguous() : torch::empty(0);
  torch::Tensor dm_dsigma1_sq = train ? torch::zeros_like(img1).contiguous() : torch::empty(0);
  torch::Tensor dm_dsigma12 = train ? torch::zeros_like(img1).contiguous() : torch::empty(0);

  if (window_size == 11) {
    fusedssim_K11_CUDA<<<grid,block>>>(
      H, W, CH, C1, C2,
      img1.contiguous().data<float>(),
      img2.contiguous().data<float>(),
      target.contiguous().data<float>(),
      dm_dmu1.contiguous().data<float>(),
      dm_dsigma1_sq.contiguous().data<float>(),
      dm_dsigma12.contiguous().data<float>()
    );
  }
  else if (window_size == 9){
    fusedssim_K09_CUDA<<<grid,block>>>(
      H, W, CH, C1, C2,
      img1.contiguous().data<float>(),
      img2.contiguous().data<float>(),
      target.contiguous().data<float>(),
      dm_dmu1.contiguous().data<float>(),
      dm_dsigma1_sq.contiguous().data<float>(),
      dm_dsigma12.contiguous().data<float>()
    );
  }
  else if (window_size == 7){
    fusedssim_K07_CUDA<<<grid,block>>>(
      H, W, CH, C1, C2,
      img1.contiguous().data<float>(),
      img2.contiguous().data<float>(),
      target.contiguous().data<float>(),
      dm_dmu1.contiguous().data<float>(),
      dm_dsigma1_sq.contiguous().data<float>(),
      dm_dsigma12.contiguous().data<float>()
    );
  }
  else if (window_size == 5){
    fusedssim_K05_CUDA<<<grid,block>>>(
      H, W, CH, C1, C2,
      img1.contiguous().data<float>(),
      img2.contiguous().data<float>(),
      target.contiguous().data<float>(),
      dm_dmu1.contiguous().data<float>(),
      dm_dsigma1_sq.contiguous().data<float>(),
      dm_dsigma12.contiguous().data<float>()
    );
  }
  else if (window_size == 3) {
    fusedssim_K03_CUDA<<<grid,block>>>(
      H, W, CH, C1, C2,
      img1.contiguous().data<float>(),
      img2.contiguous().data<float>(),
      target.contiguous().data<float>(),
      dm_dmu1.contiguous().data<float>(),
      dm_dsigma1_sq.contiguous().data<float>(),
      dm_dsigma12.contiguous().data<float>()
    );
  }

  return std::make_tuple(target, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
}

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
) {
  int B = img1.size(0);
  int CH = img1.size(1);
  int H = img1.size(2);
  int W = img1.size(3);

  torch::Tensor dL_dimg1 = torch::zeros_like(img1).contiguous();

  dim3 grid((W + BX - 1) / BX, (H + BY - 1) / BY, B);
  dim3 block(BX, BY, 1);

  if (window_size == 11) {
    fusedssim_K11_backwardCUDA<<<grid,block>>>(
      H, W, CH, C1, C2,
      img1.contiguous().data<float>(),
      img2.contiguous().data<float>(),
      dL_dmap.contiguous().data<float>(),
      dL_dimg1.contiguous().data<float>(),
      dm_dmu1.contiguous().data<float>(),
      dm_dsigma1_sq.contiguous().data<float>(),
      dm_dsigma12.contiguous().data<float>()
    );
  }
  else if (window_size == 9) {
    fusedssim_K09_backwardCUDA<<<grid,block>>>(
      H, W, CH, C1, C2,
      img1.contiguous().data<float>(),
      img2.contiguous().data<float>(),
      dL_dmap.contiguous().data<float>(),
      dL_dimg1.contiguous().data<float>(),
      dm_dmu1.contiguous().data<float>(),
      dm_dsigma1_sq.contiguous().data<float>(),
      dm_dsigma12.contiguous().data<float>()
    );
  }
  else if (window_size == 7) {
    fusedssim_K07_backwardCUDA<<<grid,block>>>(
      H, W, CH, C1, C2,
      img1.contiguous().data<float>(),
      img2.contiguous().data<float>(),
      dL_dmap.contiguous().data<float>(),
      dL_dimg1.contiguous().data<float>(),
      dm_dmu1.contiguous().data<float>(),
      dm_dsigma1_sq.contiguous().data<float>(),
      dm_dsigma12.contiguous().data<float>()
    );
  }
  else if (window_size == 5) {
    fusedssim_K05_backwardCUDA<<<grid,block>>>(
      H, W, CH, C1, C2,
      img1.contiguous().data<float>(),
      img2.contiguous().data<float>(),
      dL_dmap.contiguous().data<float>(),
      dL_dimg1.contiguous().data<float>(),
      dm_dmu1.contiguous().data<float>(),
      dm_dsigma1_sq.contiguous().data<float>(),
      dm_dsigma12.contiguous().data<float>()
    );
  }
  else if (window_size == 3) {
    fusedssim_K03_backwardCUDA<<<grid,block>>>(
      H, W, CH, C1, C2,
      img1.contiguous().data<float>(),
      img2.contiguous().data<float>(),
      dL_dmap.contiguous().data<float>(),
      dL_dimg1.contiguous().data<float>(),
      dm_dmu1.contiguous().data<float>(),
      dm_dsigma1_sq.contiguous().data<float>(),
      dm_dsigma12.contiguous().data<float>()
    );
  }

  return dL_dimg1;
}
