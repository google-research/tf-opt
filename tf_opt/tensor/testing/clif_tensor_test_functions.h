// Copyright 2025 The tf.opt Authors.
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

#ifndef TF_OPT_TENSOR_TESTING_CLIF_TENSOR_TEST_FUNCTIONS_H_
#define TF_OPT_TENSOR_TESTING_CLIF_TENSOR_TEST_FUNCTIONS_H_

#include <cstdint>
#include <numeric>
#include <vector>

#include "ortools/base/logging.h"
#include "tf_opt/tensor/reduce.h"
#include "tf_opt/tensor/tensor.h"

namespace tf_opt {

inline double TfOptTestReduceSum(const DoubleTensor& double_tensor) {
  std::vector<int64_t> axes(double_tensor.dimension().num_dimensions());
  std::iota(axes.begin(), axes.end(), 0);
  const DoubleTensor result = ReduceSum(double_tensor, axes);
  CHECK_EQ(result.dimension().num_dimensions(), 0);
  return result.flat_value(0);
}

inline DoubleTensor TwoByThree(double start_value) {
  DoubleTensor result(Shape({2, 3}));
  result.SetValueSpan({0, 0}, start_value++);
  result.SetValueSpan({0, 1}, start_value++);
  result.SetValueSpan({0, 2}, start_value++);
  result.SetValueSpan({1, 0}, start_value++);
  result.SetValueSpan({1, 1}, start_value++);
  result.SetValueSpan({1, 2}, start_value++);
  return result;
}

}  // namespace tf_opt

#endif  // TF_OPT_TENSOR_TESTING_CLIF_TENSOR_TEST_FUNCTIONS_H_
