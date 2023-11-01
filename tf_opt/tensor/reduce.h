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

#ifndef TF_OPT_TENSOR_REDUCE_H_
#define TF_OPT_TENSOR_REDUCE_H_

#include <cstdint>
#include <numeric>
#include <vector>

#include "absl/status/statusor.h"
#include "tf_opt/tensor/element_operations.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor.h"

namespace tf_opt {

// Returns the output shape for reducing the given 'input_shape' along the
// 'axes'. This requires the 'axes' vector to be sorted and not contain
// duplicates.
absl::StatusOr<Shape> ReduceOutputShape(const Shape& input_shape,
                                        const std::vector<int64_t>& axes);

template <typename T>
Tensor<T> ReduceMax(const Tensor<T>& input, const std::vector<int64_t>& axes);
template <typename T>
Tensor<T> ReduceMin(const Tensor<T>& input, const std::vector<int64_t>& axes);
template <typename T>
Tensor<T> ReduceMean(const Tensor<T>& input, const std::vector<int64_t>& axes);
template <typename T>
Tensor<T> ReduceSum(const Tensor<T>& input, const std::vector<int64_t>& axes);

template <typename T>
T ReduceMax(const Tensor<T>& input);
template <typename T>
T ReduceMin(const Tensor<T>& input);
template <typename T>
T ReduceMean(const Tensor<T>& input);
template <typename T>
T ReduceSum(const Tensor<T>& input);

// ////////////////////////// Implementation details ///////////////////////////

namespace internal {

template <typename T>
const Tensor<T> GetInputSliceForReduce(const Tensor<T>& input_tensor,
                                       const std::vector<int64_t>& reduce_axes,
                                       const Shape& output_tensor_shape,
                                       int output_flat_index) {
  // Coordinate of output.
  std::vector<int64_t> multi_index =
      output_tensor_shape.ExpandIndex(output_flat_index);
  const Shape& input_shape = input_tensor.dimension();
  std::vector<int64_t> begins(input_shape.num_dimensions(), -1);
  std::vector<int64_t> sizes(input_shape.dimension_sizes());

  // pass one: set begins if reduced
  for (const int reduce_axis : reduce_axes) {
    begins[reduce_axis] = 0;
  }

  // pass two: set begins and sizes if not reduced
  {
    int output_axis = 0;
    for (int input_index = 0; input_index < input_shape.num_dimensions();
         ++input_index) {
      if (begins[input_index] < 0) {
        begins[input_index] = multi_index.at(output_axis);
        sizes[input_index] = 1;
        ++output_axis;
      }
    }
  }

  // Everything in begins should be set.
  for (const int64_t begin : begins) {
    CHECK_GE(begin, 0);
  }

  return input_tensor.Slice(begins, sizes);
}

// This method assumes that the 'axes' is sorted and doesn't contain duplicates.
template <typename ResultType, typename InputType, typename ReduceOperator>
Tensor<ResultType> Reduce(const Tensor<InputType>& input,
                          const std::vector<int64_t>& axes,
                          const ReduceOperator& reduce_operator) {
  const Shape output_shape = ReduceOutputShape(input.dimension(), axes).value();
  Tensor<ResultType> result(output_shape);
  for (int i = 0; i < result.flat_values().size(); ++i) {
    const Tensor<InputType> input_slice =
        GetInputSliceForReduce(input, axes, output_shape, i);
    (*result.mutable_flat_values())[i] =
        reduce_operator(input_slice.flat_values(), i);
  }
  return result;
}

template <typename T>
std::vector<int64_t> AllDims(const Tensor<T> tensor) {
  std::vector<int64_t> result(tensor.dimension().num_dimensions());
  std::iota(result.begin(), result.end(), static_cast<int64_t>(0));
  return result;
}

}  // namespace internal

template <typename T>
Tensor<T> ReduceMax(const Tensor<T>& input, const std::vector<int64_t>& axes) {
  MaxAllElements<T> element;
  return internal::Reduce<T>(input, axes, element);
}

template <typename T>
Tensor<T> ReduceMin(const Tensor<T>& input, const std::vector<int64_t>& axes) {
  MinAllElements<T> element;
  return internal::Reduce<T>(input, axes, element);
}

template <typename T>
Tensor<T> ReduceMean(const Tensor<T>& input, const std::vector<int64_t>& axes) {
  AverageAllElements<T> element;
  return internal::Reduce<T>(input, axes, element);
}

template <typename T>
Tensor<T> ReduceSum(const Tensor<T>& input, const std::vector<int64_t>& axes) {
  AddAllElements<T> element;
  return internal::Reduce<T>(input, axes, element);
}

template <typename T>
T ReduceMax(const Tensor<T>& input) {
  return ReduceMax(input, internal::AllDims(input)).flat_value(0);
}
template <typename T>
T ReduceMin(const Tensor<T>& input) {
  return ReduceMin(input, internal::AllDims(input)).flat_value(0);
}
template <typename T>
T ReduceMean(const Tensor<T>& input) {
  return ReduceMean(input, internal::AllDims(input)).flat_value(0);
}
template <typename T>
T ReduceSum(const Tensor<T>& input) {
  return ReduceSum(input, internal::AllDims(input)).flat_value(0);
}

}  // namespace tf_opt

#endif  // TF_OPT_TENSOR_REDUCE_H_
