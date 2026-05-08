
#include "fusion.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multi_view_fusion", &multi_view_fusion);
}