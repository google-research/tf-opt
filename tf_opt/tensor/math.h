// Copyright 2020 The tf.opt Authors.
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

// Throughout, we use the following notation.  We have a binary op "o" in
// {+,-,*,/} performing r = t1 o t2 for Tensors "t1", "t2", and "r"
// (all operations are element-wise). Let "s1" be the shape of
// "t1" and "s2" be the shape of "t2".
//
// In determining what inputs are of legal shape, we follow NumPy broadcasting
// rules.  The output has m = max(s1.num_dimensions(), s2.num_dimensions())
// dimensions.  The shorter of s1 or s2 is then padded with ones at the front.
// The shapes are compatible if for every i = 0, ..., m-1:
//   s1.dimension_size(i) == s2.dimension_size(i) OR
//   s1.dimension_size(i) == 1 OR
//   s2.dimension_size(i) == 1.
//
// The output shape is, for every i,
//   max(s1.dimension_size(i), s2.dimension_size(i)).
//
// This is the same behavior as NumPy and TensorFlow.
//
// The functions:
// * Add
// * Subtract
// * Multiply
// * Divide
// * ElementwiseMaximum
// * ElementwiseMinimum
// take two Tensors of type T and produce a result Tensor of type T by
// applying the binary operation element-wise. If the operation is not defined
// for type T, (e.g. multiplication is not defined for a pair of LinearExprs,
// and thus it is not possible to multiply or divide two MPTensors), then the
// function will not compile.  For ElementwiseMaximum, std::max is the binary
// operator (and std::min for ElementwiseMinimum).
//
// Example use:
//   DoubleTensor st1({{1.0, 2.0, 3.0}, {10.0, 20.0, 30.0}});
//   Multiply(st1, st1);  // Returns {{1.0, 4.0, 9.0}, {100.0, 400.0, 900.0}}
//
//   DoubleTensor s2({2.0, 4.0, 6.0});
//   Multiply(st1, s2);  // Returns {{2.0, 8.0, 18.0}, {20.0, 80.0, 180.0}}
//
//   DoubleTensor s3({{2.0, 4.0, 6.0}, {2.0, 4.0, 6.0}});
//   Multiply(st1, s3);  // Equivalent to Multiply(st1, s2)
//
//   MPTensor t(std::vector<LinearExpr>({1.0}))
//   Multiply(t,t);  // Will not compile, not linear!
//   Divide(t, t);  // Will not compile, not linear!
//
// Note that LinearExpr operator*(LinearExpr, LinearExpr) is (intentionally)
// not defined, preventing compilation of the nonsensical expression above.

#ifndef TF_OPT_TENSOR_MATH_H_
#define TF_OPT_TENSOR_MATH_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "tf_opt/tensor/element_operations.h"
#include "tf_opt/tensor/math_impl.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor.h"

namespace tf_opt {

// Compute the output tensor shape of the tensor that would be generated from
// applying a binary op (e.g. Add, Subtract, ...) to tensors of shape left and
// right.  Returns error if the shapes are incompatible.
absl::StatusOr<Shape> BinaryOpOutputShape(const Shape& left,
                                          const Shape& right);

// Compute the output tensor shape if tensors of shape left and right are matrix
// multiplied together via MatMul.  Returns error if shapes are incompatible.
absl::StatusOr<Shape> MatMulOutputShape(const Shape& left, const Shape& right);

// Returns tensor with shape of input and each element negated.
//
// T requirement: T::operator-() is defined.
template <typename T>
Tensor<T> ElementwiseNegate(const Tensor<T>& input) {
  return internal::UnaryElementwiseOp<T>(
      input, [](const T& element, const int64_t) { return -element; });
}

// Returns tensor with shape of input and Relu(x) = max(x, 0) applied to each
// element.
//
// T requirement: TfOptMax(T, T) is defined.
template <typename T>
Tensor<T> ElementwiseRelu(const Tensor<T>& input) {
  ReluElement<T> element;
  return internal::UnaryElementwiseOp<T>(input, element);
}

// Returns tensor with shape of input and
//   ClippedRelu(x, cap) = min(cap, max(x, 0))
// applied to each element.
//
// T requirement: TfOptMax(T, T) and TfOptMin(T, T) is defined.
template <typename T>
Tensor<T> ElementwiseClippedRelu(const Tensor<T>& input, double cap) {
  ClippedReluElement<T> element(cap);
  return internal::UnaryElementwiseOp<T>(input, element);
}

// Returns left + right, CHECK fails on shape error.
//
// T requirement: operator+(T, T) is defined.
template <typename T>
Tensor<T> Add(const Tensor<T>& left, const Tensor<T>& right) {
  return internal::Add<T, T, T>(left, right);
}

// Returns left - right, CHECK fails on shape error.
//
// T requirement: operator-(T, T) is defined.
template <typename T>
Tensor<T> Subtract(const Tensor<T>& left, const Tensor<T>& right) {
  return internal::Subtract<T, T, T>(left, right);
}

// Returns left * right (componentwise), CHECK fails on shape error.
//
// T requirement: operator*(T, T) is defined.
template <typename T>
Tensor<T> Multiply(const Tensor<T>& left, const Tensor<T>& right) {
  return internal::Multiply<T, T, T>(left, right);
}

// Returns left / right (componentwise), CHECK fails on shape error.
//
// T requirement: operator/(T, T) is defined.
template <typename T>
Tensor<T> Divide(const Tensor<T>& left, const Tensor<T>& right) {
  return internal::Divide<T, T, T>(left, right);
}

// Returns left * right (matrix multiplicaiton), CHECK fails on shape error.
//
// T requirement: operator*(T, T) is defined.
template <typename T>
Tensor<T> MatMul(const Tensor<T>& left, const Tensor<T>& right) {
  return internal::MatMul<T, T, T>(left, right);
}

// Returns max(left, right) (componentwise), CHECK fails on shape error.
//
// T requirement: TfOptMax(T, T) is defined.
template <typename T>
Tensor<T> ElementwiseMaximum(const Tensor<T>& left, const Tensor<T>& right) {
  MaxElements<T> element;
  return internal::BinaryElementwiseOp<T, T, T>(left, right, element);
}

// Returns min(left, right) (componentwise), CHECK fails on shape error.
//
// T requirement: TfOptMin(T, T) is defined.
template <typename T>
Tensor<T> ElementwiseMinimum(const Tensor<T>& left, const Tensor<T>& right) {
  MinElements<T> element;
  return internal::BinaryElementwiseOp<T, T, T>(left, right, element);
}

}  // namespace tf_opt

#endif  // TF_OPT_TENSOR_MATH_H_
