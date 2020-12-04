// Copyright 2021 The tf.opt Authors.
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

#ifndef TF_OPT_TENSOR_CONCAT_H_
#define TF_OPT_TENSOR_CONCAT_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor.h"

namespace tf_opt {

// The shape resulting from calling Concat() below on tensors with these shapes.
// Does not support broadcasting.
//
// Returns an error on invalid Concat() input:
//   * input_shapes.empty()
//   * input_shapes[i].num_dimensions() is not the same for all i
//   * axis < 0 or axis >= input_shapes[0].num_dimensions()
//   * for any i, for any j != axis:
//       input_shapes[i].dimension_size(j) != input_shapes[0].dimension_size(j)
absl::StatusOr<Shape> ConcatOutputShape(const std::vector<Shape>& input_shapes,
                                        int axis);

// Given a list of tensors t1, t2, ... and an "axis" to concatenate on,
// produces a single output tensor.  For every dimension except of "axis",
// input tensors must agree on shape.
//
// If each tensor i has shape [s_1, s_2, ..., axis_shape_i, ..., s_n],
// where the s_j are constant across the tensors i, letting,
//   output_size = sum_{i in input_tensors} axis_shape_i
// the shape of the output will be [s_1, s_2, ..., output_size, ..., s_n].
//
// Examples:
//
//   1. t1 = [[1, 2, 3], [4, 5, 6]]
//         t1.dimension() ==> [2, 3]
//      t2 = [[7, 8, 9], [10, 11, 12]]
//         t2.dimension() ==> [2, 3]
//
//      c1 = concat([t1, t2], axis=0)
//         c1 ==> [[1, 2, 3],[4, 5, 6], [7, 8, 9], [10, 11, 12]]
//         c1.dimension() ==> [4, 3]
//
//      c2 = concat([t1, t2], axis=1)
//         c2 ==> [[1, 2, 3, 7, 8, 9],[4, 5, 6, 10, 11, 12]]
//         c1.dimension() ==> [2, 6]
//
//   2. t1 shape = [3, 5]
//      t2 shape = [7, 5]
//      concat([t1, t2], axis=0) is legal, output shape [10, 5]
//      concat([t1, t2], axis=1) is illegal.
//
// Will CHECK fail if input shapes/axis are incompatible (i.e. if
// ConcatOutputShape above returns an error).
template <typename T>
Tensor<T> Concat(const std::vector<const Tensor<T>*>& inputs, int axis);

template <typename T>
Tensor<T> ConcatDirect(const std::vector<Tensor<T>>& inputs, int axis);

namespace internal {

// Exposed only for testing.
//
// Given several lists of different sizes that have been concatenated together,
// converts an index in the concatenated list into the (list, position) pair
// that was the original source of the data.
//
// Example: input is list_sizes = [3, 5, 4]
//
// concat_index|which_list|position_in_list|
//            0|         0|               0|
//            1|         0|               1|
//            2|         0|               2|
//            3|         1|               0|
//            4|         1|               1|
//            5|         1|               2|
//            6|         1|               3|
//            7|         1|               4|
//            8|         2|               0|
//            9|         2|               1|
//           10|         2|               2|
//           11|         2|               3|
class ConcatLookupTable {
 public:
  // list_sizes are the sizes of the original lists, in the order they were
  // concatenated.
  explicit ConcatLookupTable(const std::vector<int64_t>& list_sizes);

  // Given an index in the concatenated list, returns the pair:
  // (index of the list, position within list)
  // that was the source of the data.
  std::pair<int, int64_t> Lookup(int64_t concat_index) const;

 private:
  std::vector<int64_t> cumulative_list_sizes_;
  std::vector<int64_t> concat_index_to_init_list_index_;
};

}  // namespace internal

// //////////////// Template Function Implementations //////////////////////////

// TODO: Have a smarter person write a smarter algorithm for this.
//   * From a runtime perspective, it is asymptotically optimal.
//   * For memory allocations, the entire allocation by ConcatLookupTable is
//       unnecessary.  But it will never be bigger than the memory allocated
//       by the output, and usually much smaller.
//   * The first big problem is that we are not using bulk copy methods.
//   * The second big problem is that we are doing a bunch of repeated work
//     in our index calculations that could probably be pretty easily shared
//     between output_flat_index and output_flat_index + 1.  Further, we are
//     allocating multiple vectors per computation, which isn't needed.
template <typename T>
Tensor<T> Concat(const std::vector<const Tensor<T>*>& inputs, int axis) {
  std::vector<Shape> input_shapes;
  input_shapes.reserve(inputs.size());
  for (const Tensor<T>* const input : inputs) {
    input_shapes.push_back(input->dimension());
  }
  const Shape out_shape = ConcatOutputShape(input_shapes, axis).value();
  Tensor<T> result(out_shape);

  std::vector<int64_t> axis_sizes;
  axis_sizes.reserve(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    axis_sizes.push_back(inputs[i]->dimension().dimension_size(axis));
  }
  const internal::ConcatLookupTable axis_index_lookup(axis_sizes);

  for (int output_flat_index = 0; output_flat_index < out_shape.size();
       output_flat_index++) {
    // Position in the concatenated output
    std::vector<int64_t> multi_index = out_shape.ExpandIndex(output_flat_index);
    const int64_t output_axis_value = multi_index[axis];
    const std::pair<int, int64_t> which_tensor_and_position =
        axis_index_lookup.Lookup(output_axis_value);
    const int input_tensor_index = which_tensor_and_position.first;
    const int64_t position_in_tensor_on_axis = which_tensor_and_position.second;
    // WARNING:  The following line mutates multi_index to now give a
    // position from the select input tensor, rather than in the output
    // concatenated tensor.
    multi_index[axis] = position_in_tensor_on_axis;
    const Tensor<T>* const input_tensor = inputs[input_tensor_index];
    (*result.mutable_flat_values())[output_flat_index] =
        input_tensor->value(multi_index);
  }
  return result;
}

template <typename T>
Tensor<T> ConcatDirect(const std::vector<Tensor<T>>& inputs, const int axis) {
  std::vector<const Tensor<T>*> input_pointers;
  input_pointers.reserve(inputs.size());
  for (const Tensor<T>& input : inputs) {
    input_pointers.push_back(&input);
  }
  return Concat(input_pointers, axis);
}

}  // namespace tf_opt

#endif  // TF_OPT_TENSOR_CONCAT_H_
