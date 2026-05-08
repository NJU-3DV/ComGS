
#include <torch/extension.h>
#include "src/render_equation.h"
#include "src/d2n.h"
#include "src/sample.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render_equation", &pbr::RenderingEquation);
  m.def("render_equation_backward", &pbr::RenderingEquationBackward);
  m.def("depth2normal", &d2n::Depth2Normal);
  m.def("depth2normal_backward", &d2n::Depth2NormalBackward);
  m.def("uniform_sample", &sample::uniform_sample);
}