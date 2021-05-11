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

#ifndef TF_OPT_SHARED_OPS_ARITHMETIC_OPERATIONS_H_
#define TF_OPT_SHARED_OPS_ARITHMETIC_OPERATIONS_H_

#include <string>
#include <utility>
#include <vector>

#include "ortools/base/logging.h"
#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_validator.h"
#include "tf_opt/neural_net/operation_visitor.h"
#include "tf_opt/neural_net/ops/operation_types.h"
#include "tf_opt/open_source/status_macros.h"
#include "tf_opt/tensor/math.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {

template <BinaryArithmeticOpType OpType>
class BinaryArithmeticOperation : public Operation {
 public:
  ~BinaryArithmeticOperation() override {}

  const Shape& left() const { return input_shape(0); }

  const Shape& right() const { return input_shape(1); }

  void Accept(OperationVisitor* visitor) const override {
    visitor->Visit(*this);
  }
  static absl::StatusOr<BinaryArithmeticOperation<OpType>> Create(
      std::string op_name, Shape left_shape, Shape right_shape);

  // TODO: replace this by a variadic template function.
  static MaybeForGraph<BinaryArithmeticOperation<OpType>> CreateForGraph(
      std::string op_name, const Operation* left, const Operation* right);

  // Expected input format:
  //   input_shapes: Shapes of the input tensors, input_shapes.size() == 2.
  //   output_shape: The shape to produce, follows broadcasting rules.
  //   options: Empty.
  static absl::StatusOr<BinaryArithmeticOperation<OpType>> GenericCreate(
      std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
      const Options& options);

  proto::TensorNode ToProto(
      const std::vector<std::string>& inputs) const override;

 private:
  BinaryArithmeticOperation(std::string op_name,
                            std::vector<Shape> input_shapes,
                            Shape output_shape);
};

// Add two tensors element-wise.
using AddOperation = BinaryArithmeticOperation<BinaryArithmeticOpType::kAdd>;

// Subtract two tensors element-wise.
using SubtractOperation =
    BinaryArithmeticOperation<BinaryArithmeticOpType::kSubtract>;

// Multiply two tensors element-wise.
using MultiplyOperation =
    BinaryArithmeticOperation<BinaryArithmeticOpType::kMultiply>;

// Divide two tensors element-wise.
using DivideOperation =
    BinaryArithmeticOperation<BinaryArithmeticOpType::kDivide>;

// ///////////////////////////// Implementation ////////////////////////////////

template <BinaryArithmeticOpType OpType>
BinaryArithmeticOperation<OpType>::BinaryArithmeticOperation(
    std::string op_name, std::vector<Shape> input_shapes, Shape output_shape)
    : Operation(std::move(op_name), std::move(input_shapes),
                std::move(output_shape)) {}

template <BinaryArithmeticOpType OpType>
absl::StatusOr<BinaryArithmeticOperation<OpType>>
BinaryArithmeticOperation<OpType>::Create(std::string op_name, Shape left_shape,
                                          Shape right_shape) {
  TFOPT_ASSIGN_OR_RETURN(Shape output_shape,
                         BinaryOpOutputShape(left_shape, right_shape));
  return BinaryArithmeticOperation<OpType>(
      std::move(op_name), {std::move(left_shape), std::move(right_shape)},
      std::move(output_shape));
}

template <BinaryArithmeticOpType OpType>
MaybeForGraph<BinaryArithmeticOperation<OpType>>
BinaryArithmeticOperation<OpType>::CreateForGraph(std::string op_name,
                                                  const Operation* left,
                                                  const Operation* right) {
  return FromMaybeCreated(
      BinaryArithmeticOperation<OpType>::Create(
          std::move(op_name), left->output_shape(), right->output_shape()),
      {left, right});
}

template <BinaryArithmeticOpType OpType>
absl::StatusOr<BinaryArithmeticOperation<OpType>>
BinaryArithmeticOperation<OpType>::GenericCreate(
    std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
    const Options& options) {
  OperationValidator validator("BinaryArithmeticOperation", op_name);
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectInputSizeEquals(input_shapes.size(), 2));
  TFOPT_RETURN_IF_ERROR(validator.ExpectOptionsEmpty(options.size()));
  TFOPT_ASSIGN_OR_RETURN(BinaryArithmeticOperation<OpType> result,
                         BinaryArithmeticOperation<OpType>::Create(
                             std::move(op_name), std::move(input_shapes[0]),
                             std::move(input_shapes[1])),
                         _ << validator.base_error_message());
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectOutputShapeEquals(output_shape, result.output_shape()));
  return std::move(result);
}

template <BinaryArithmeticOpType OpType>
proto::TensorNode BinaryArithmeticOperation<OpType>::ToProto(
    const std::vector<std::string>& inputs) const {
  CHECK_EQ(inputs.size(), 2);
  proto::TensorNode result;
  result.set_name(name());
  switch (OpType) {
    case BinaryArithmeticOpType::kAdd:
      result.set_op_type(proto::OpType::ADD);
      break;
    case BinaryArithmeticOpType::kSubtract:
      result.set_op_type(proto::OpType::SUBTRACT);
      break;
    case BinaryArithmeticOpType::kMultiply:
      result.set_op_type(proto::OpType::MULTIPLY);
      break;
    case BinaryArithmeticOpType::kDivide:
      result.set_op_type(proto::OpType::DIVIDE);
      break;
  }
  *result.mutable_out_dimension() = output_shape().AsProto();
  for (const std::string& input : inputs) {
    result.add_input_names(input);
  }
  result.set_output_type(proto::TensorNode::FLOAT32);
  return result;
}

}  // namespace tf_opt

#endif  // TF_OPT_SHARED_OPS_ARITHMETIC_OPERATIONS_H_
