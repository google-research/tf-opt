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

// This is a shared backend for math.h and mp_math.h, not a public API. Unless
// you are editing those files, you shouldn't need to be here.

#ifndef TF_OPT_TENSOR_MATH_IMPL_H_
#define TF_OPT_TENSOR_MATH_IMPL_H_

#include <algorithm>
#include <cstdint>
#include <vector>

#include "ortools/base/logging.h"
#include "absl/status/statusor.h"
#include "tf_opt/tensor/element_operations.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor.h"

namespace tf_opt {
namespace internal {

// UnaryElementOperator must have the operation
//   ResultType operator(InputType, int64_t) const;
// defined.
template <typename ResultType, typename InputType,
          typename UnaryElementOperator>
Tensor<ResultType> UnaryElementwiseOp(
    const Tensor<InputType>& input,
    const UnaryElementOperator& unary_element_operator) {
  Tensor<ResultType> result(input.dimension());
  for (int64_t i = 0; i < result.size(); i++) {
    (*result.mutable_flat_values())[i] =
        unary_element_operator(input.flat_value(i), i);
  }
  return result;
}

Shape BroadcastPadIfNeeded(const Shape& shape, int64_t target_num_dimensions);

int64_t MaxNumDimensions(const Shape& shape_left, const Shape& shape_right);

absl::StatusOr<Shape> ResultShape(const Shape& padded_left,
                                  const Shape& padded_right);

enum class MultiplicationPosition { kLeft, kRight };

// Relates the elements of r to the elements of t1 used to produce r.
class Broadcaster {
 public:
  // broadcast_shape: the shape of "r".
  // padded_true_shape: the shape of "t1" after padding with 1s at the
  //   start to match length of r.  See BroadcastPadIfNeeded.
  Broadcaster(const Shape& true_shape, const Shape& padded_true_shape,
              const Shape& broadcast_shape)
      : true_shape_(true_shape),
        padded_shape_(padded_true_shape),
        broadcast_shape_(broadcast_shape) {}

  // Given a flat index into the result "r", compute the flat index in "t1".
  int64_t BroadcastIndexToTrueIndex(int64_t broadcast_index) const;
  std::vector<int64_t> BroadcastIndexToMatmulSliceArg(
      int64_t broadcast_index, MultiplicationPosition mult_pos) const;

 private:
  std::vector<int64_t> PaddedMultiIndex(int64_t broadcast_index) const;

  const Shape true_shape_;
  const Shape padded_shape_;
  const Shape broadcast_shape_;
};

// Given input tensors left and right of broadcast compatible shapes, computes
// a new tensor that is in spirit:
//   [f(left[i], right[i]) for i in ResultDimension(left, right)].
// Above, i is a multi index and accesses left and right by broadcasting rules.
//
// BinaryElementOperator should take a LeftOperandType, a RightOperandType, and
// the OutputIndex to produce the ResultType. See element_operations.h for
// examples, e.g. AddElements<ResultType, LeftOperandType, RightOperandType>..
template <typename ResultType, typename LeftOperandType,
          typename RightOperandType, typename BinaryElementOperator>
Tensor<ResultType> BinaryElementwiseOp(const Tensor<LeftOperandType>& left,
                                       const Tensor<RightOperandType>& right,
                                       const BinaryElementOperator& f) {
  const int64_t num_dim = MaxNumDimensions(left.dimension(), right.dimension());
  const Shape padded_left_dim = BroadcastPadIfNeeded(left.dimension(), num_dim);
  const Shape padded_right_dim =
      BroadcastPadIfNeeded(right.dimension(), num_dim);
  const Shape result_shape =
      ResultShape(padded_left_dim, padded_right_dim).value();
  Tensor<ResultType> result(result_shape);
  const Broadcaster broadcast_left(left.dimension(), padded_left_dim,
                                   result_shape);
  const Broadcaster broadcast_right(right.dimension(), padded_right_dim,
                                    result_shape);
  for (int64_t i = 0; i < result.size(); i++) {
    const LeftOperandType& left_value =
        left.flat_values()[broadcast_left.BroadcastIndexToTrueIndex(i)];
    const RightOperandType& right_value =
        right.flat_values()[broadcast_right.BroadcastIndexToTrueIndex(i)];
    (*result.mutable_flat_values())[i] = f(left_value, right_value, i);
  }
  return result;
}

template <typename ResultType, typename LeftOperandType,
          typename RightOperandType>
Tensor<ResultType> Add(const Tensor<LeftOperandType>& left,
                       const Tensor<RightOperandType>& right) {
  return internal::BinaryElementwiseOp<ResultType, LeftOperandType,
                                       RightOperandType>(
      left, right,
      [](const LeftOperandType& left_element,
         const RightOperandType& right_element,
         const int64_t) { return left_element + right_element; });
}

template <typename ResultType, typename LeftOperandType,
          typename RightOperandType>
Tensor<ResultType> Subtract(const Tensor<LeftOperandType>& left,
                            const Tensor<RightOperandType>& right) {
  return internal::BinaryElementwiseOp<ResultType, LeftOperandType,
                                       RightOperandType>(
      left, right,
      [](const LeftOperandType& left_element,
         const RightOperandType& right_element,
         const int64_t) { return left_element - right_element; });
}

template <typename ResultType, typename LeftOperandType,
          typename RightOperandType>
Tensor<ResultType> Multiply(const Tensor<LeftOperandType>& left,
                            const Tensor<RightOperandType>& right) {
  return internal::BinaryElementwiseOp<ResultType, LeftOperandType,
                                       RightOperandType>(
      left, right,
      [](const LeftOperandType& left_element,
         const RightOperandType& right_element,
         const int64_t) { return left_element * right_element; });
}

template <typename ResultType, typename LeftOperandType,
          typename RightOperandType>
Tensor<ResultType> Divide(const Tensor<LeftOperandType>& left,
                          const Tensor<RightOperandType>& right) {
  return internal::BinaryElementwiseOp<ResultType, LeftOperandType,
                                       RightOperandType>(
      left, right,
      [](const LeftOperandType& left_element,
         const RightOperandType& right_element,
         const int64_t) { return left_element / right_element; });
}

absl::StatusOr<Shape> MatMulResultShape(const Shape& padded_left,
                                        const Shape& padded_right);

template <typename ResultType, typename LeftOperandType,
          typename RightOperandType>
Tensor<ResultType> MatMul(const Tensor<LeftOperandType>& left,
                          const Tensor<RightOperandType>& right) {
  const int64_t left_dimensions = left.dimension().num_dimensions();
  const int64_t right_dimensions = right.dimension().num_dimensions();
  CHECK_GE(left_dimensions, 2);
  CHECK_GE(right_dimensions, 2);

  const int64_t num_dim = MaxNumDimensions(left.dimension(), right.dimension());
  const Shape padded_left_dim = BroadcastPadIfNeeded(left.dimension(), num_dim);
  const Shape padded_right_dim =
      BroadcastPadIfNeeded(right.dimension(), num_dim);
  const Shape result_shape =
      MatMulResultShape(padded_left_dim, padded_right_dim).value();
  Tensor<ResultType> result(result_shape);
  Broadcaster broadcast_left(left.dimension(), padded_left_dim, result_shape);
  Broadcaster broadcast_right(right.dimension(), padded_right_dim,
                              result_shape);
  for (int64_t i = 0; i < result.size(); i++) {
    const std::vector<LeftOperandType> left_row =
        left.VectorSlice(broadcast_left.BroadcastIndexToMatmulSliceArg(
            i, MultiplicationPosition::kLeft));
    const std::vector<RightOperandType> right_col =
        right.VectorSlice(broadcast_right.BroadcastIndexToMatmulSliceArg(
            i, MultiplicationPosition::kRight));
    CHECK_EQ(left_row.size(), right_col.size());
    ResultType inner_prod = result.flat_values()[i];  // zero.
    for (int j = 0; j < left_row.size(); j++) {
      inner_prod += left_row[j] * right_col[j];
    }
    (*result.mutable_flat_values())[i] = inner_prod;
  }
  return result;
}

}  // namespace internal

}  // namespace tf_opt

#endif  // TF_OPT_TENSOR_MATH_IMPL_H_
