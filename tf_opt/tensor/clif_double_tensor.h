// Copyright 2023 The tf.opt Authors.
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

// CLIF bindings to associate NumPy arrays with tf_opt::DoubleTensor.
//
// See cs/third_party/clif/python/g3doc/ext.md
//
// Loosely based on:
//   cs/third_party/absl/python/numpy/span.h
//   cs/research/vision/piedpiper/brain/util/python/tensor_clif.h

#ifndef TF_OPT_TENSOR_CLIF_DOUBLE_TENSOR_H_
#define TF_OPT_TENSOR_CLIF_DOUBLE_TENSOR_H_

#include "third_party/clif/python/postconv.h"
#include "tf_opt/tensor/tensor.h"

// CLIF use `::tf_opt::DoubleTensor` as NumpyArray

// WARNING(rander): PyClif depends on ADL. So these methods must be in the same
// namespace as DoubleTensor.
namespace tf_opt {

bool Clif_PyObjAs(PyObject* numpy_array, DoubleTensor* tensor);

PyObject* Clif_PyObjFrom(const DoubleTensor& tensor,
                         const ::clif::py::PostConv& pc);

}  // namespace tf_opt

#endif  // TF_OPT_TENSOR_CLIF_DOUBLE_TENSOR_H_
