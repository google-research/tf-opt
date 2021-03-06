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

#include "tf_opt/neural_net/ops/conv2d_operation.h"

#include <utility>

#include "ortools/base/logging.h"
#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation_validator.h"
#include "tf_opt/open_source/status_macros.h"
#include "tf_opt/tensor/convolve.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {

constexpr const char Conv2dOperation::kOptionsStrideRowKey[];
constexpr const char Conv2dOperation::kOptionsStrideColKey[];
constexpr const char Conv2dOperation::kOptionsPaddingKey[];

Conv2dOperation::Conv2dOperation(std::string op_name, Shape input_value_shape,
                                 Shape filter_shape, Shape output_shape,
                                 Position2D stride, const PaddingType padding)
    : Operation(std::move(op_name),
                {std::move(input_value_shape), std::move(filter_shape)},
                std::move(output_shape)),
      stride_(stride),
      padding_(padding) {}

absl::StatusOr<Conv2dOperation> Conv2dOperation::Create(
    std::string op_name, Shape input_value_shape, Shape filter_shape,
    Position2D stride, const PaddingType padding) {
  TFOPT_ASSIGN_OR_RETURN(
      Shape output_shape,
      Conv2dOutputShape(input_value_shape, filter_shape, stride, padding));
  return Conv2dOperation(std::move(op_name), std::move(input_value_shape),
                         std::move(filter_shape), std::move(output_shape),
                         stride, padding);
}

MaybeForGraph<Conv2dOperation> Conv2dOperation::CreateForGraph(
    std::string op_name, const Operation* input_value, const Operation* filter,
    Position2D stride, const PaddingType padding) {
  return FromMaybeCreated(
      Create(std::move(op_name), input_value->output_shape(),
             filter->output_shape(), stride, padding),
      {input_value, filter});
}

absl::StatusOr<Conv2dOperation> Conv2dOperation::GenericCreate(
    std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
    const Options& options) {
  OperationValidator validator("Conv1dOperation", op_name);
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectInputSizeEquals(input_shapes.size(), 2));
  TFOPT_RETURN_IF_ERROR(validator.ExpectOptionsSizeAtMost(options.size(), 3));
  TFOPT_ASSIGN_OR_RETURN(
      const int stride_row,
      validator.IntegerOption(options, kOptionsStrideRowKey));
  TFOPT_ASSIGN_OR_RETURN(
      const int stride_column,
      validator.IntegerOption(options, kOptionsStrideColKey));
  TFOPT_ASSIGN_OR_RETURN(const std::string& padding_name,
                         validator.StringOption(options, kOptionsPaddingKey));
  PaddingType padding_type;
  if (!PaddingTypeFromString(padding_name, &padding_type)) {
    return validator.OperationValidationError("Invalid padding string");
  }

  TFOPT_ASSIGN_OR_RETURN(
      auto op,
      Create(op_name, input_shapes[0], input_shapes[1],
             Position2D(stride_row, stride_column), padding_type),
      _ << validator.base_error_message());
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectOutputShapeEquals(output_shape, op.output_shape()));
  return std::move(op);
}

}  // namespace tf_opt
