// Copyright 2024 The tf.opt Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tf_opt/tensor/clif_double_tensor.h"

#include <cstdint>

#include "third_party/clif/python/postconv.h"
#include "third_party/clif/python/types.h"
#include "third_party/py/numpy/core/include/numpy/ndarrayobject.h"
#include "third_party/py/numpy/core/include/numpy/ndarraytypes.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {

namespace {

constexpr int kBytesPerDouble = 8;

template <typename... Args>
void PyRaiseTypeError(Args&&... args) {
  PyErr_SetString(PyExc_TypeError,
                  absl::StrCat(std::forward<Args>(args)...).c_str());
}

template <typename... Args>
void PyRaiseValueError(Args&&... args) {
  PyErr_SetString(PyExc_ValueError,
                  absl::StrCat(std::forward<Args>(args)...).c_str());
}

Shape NdarrayToShape(PyArrayObject* input) {
  int num_dim = PyArray_NDIM(input);
  std::vector<int64_t> dimensions;
  dimensions.reserve(num_dim);
  for (int i = 0; i < num_dim; ++i) {
    dimensions.push_back(PyArray_SHAPE(input)[i]);
  }
  return Shape(dimensions);
}

}  // namespace

bool Clif_PyObjAs(PyObject* numpy_array, DoubleTensor* tensor) {
  CHECK(tensor != nullptr);
  if (!PyArray_Check(numpy_array)) {
    PyRaiseTypeError("The input is not a Numpy array.");
    return false;
  }

  PyArrayObject* input = reinterpret_cast<PyArrayObject*>(numpy_array);
  if (!PyArray_IS_C_CONTIGUOUS(input) || !PyArray_ISALIGNED(input)) {
    PyRaiseValueError("Ndarray must be C contiguous and aligned");
    return false;
  }

  NPY_TYPES np_type = static_cast<NPY_TYPES>(PyArray_TYPE(input));
  if (np_type != NPY_DOUBLE) {
    PyRaiseValueError("Ndarray must be double valued.");
  }

  CHECK_EQ(kBytesPerDouble, PyArray_ITEMSIZE(input));

  Shape shape = NdarrayToShape(input);
  CHECK_EQ(shape.size(), PyArray_SIZE(input));
  int64_t size_in_bytes = shape.size() * kBytesPerDouble;

  DoubleTensor result(shape);
  auto p = reinterpret_cast<uint8_t*>(result.mutable_flat_values()->data());
  std::memcpy(p, PyArray_DATA(input), size_in_bytes);
  *tensor = std::move(result);
  return true;
}

namespace {

std::vector<npy_intp> ShapeToNdarrayShape(const Shape& shape) {
  std::vector<npy_intp> dims;
  dims.reserve(shape.num_dimensions());
  for (const int64_t dim : shape.dimension_sizes()) {
    dims.push_back(dim);
  }
  return dims;
}

PyArrayObject* NdarrayFromCopiedData(const DoubleTensor& tensor) {
  auto dims = ShapeToNdarrayShape(tensor.dimension());
  const NPY_TYPES np_type = NPY_DOUBLE;

  PyObject* obj =
      CHECK_NOTNULL(PyArray_SimpleNew(dims.size(), dims.data(), np_type));
  PyArrayObject* output = reinterpret_cast<PyArrayObject*>(obj);
  CHECK_EQ(PyArray_ITEMSIZE(output), kBytesPerDouble);
  CHECK_EQ(PyArray_SIZE(output), tensor.size());

  const void* data = CHECK_NOTNULL(
      reinterpret_cast<const char*>(tensor.flat_values().data()));
  int64_t size_in_bytes = tensor.size() * kBytesPerDouble;
  std::memcpy(PyArray_DATA(output), data, size_in_bytes);
  return output;
}

}  // namespace

PyObject* Clif_PyObjFrom(const DoubleTensor& tensor,
                         const ::clif::py::PostConv& pc) {
  PyArrayObject* output = NdarrayFromCopiedData(tensor);
  return pc.Apply(PyArray_Return(output));
}

}  // namespace tf_opt
