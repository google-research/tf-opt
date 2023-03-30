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

#include "tf_opt/neural_net/ops/maxpool_operation.h"

#include "ortools/base/logging.h"
#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation_validator.h"
#include "tf_opt/open_source/status_macros.h"
#include "tf_opt/tensor/pooling.h"
#include "ortools/base/map_util.h"

namespace tf_opt {

constexpr char MaxpoolOperation::kOptionsFormulationKey[];
constexpr char MaxpoolOperation::kOptionsFormulationDefault[];
constexpr char MaxpoolOperation::kOptionsStrideRowKey[];
constexpr char MaxpoolOperation::kOptionsStrideColKey[];
constexpr char MaxpoolOperation::kOptionsWindowHeightKey[];
constexpr char MaxpoolOperation::kOptionsWindowWidthKey[];
constexpr char MaxpoolOperation::kOptionsPaddingKey[];

const char* MaxpoolOperation::OptionsFormulation(
    MaximumImplementationType max_impl) {
  return ToString(max_impl);
}

std::vector<std::string> MaxpoolOperation::AllMaxPoolImplementations() {
  std::vector<std::string> max_pool_impls;
  for (const auto& max_impl : AllMaximumImplementations()) {
    max_pool_impls.push_back(ToString(max_impl));
  }
  return max_pool_impls;
}

absl::StatusOr<Shape> MaxpoolOperation::OutputShape(
    const Shape& input_shape, const Position2D& window_size,
    const Position2D& strides, const PaddingType& padding) {
  return Pool2dOutputShape(input_shape, window_size, strides, padding);
}

absl::StatusOr<MaxpoolOperation> MaxpoolOperation::Create(
    std::string op_name, Shape input_shape, const Position2D ksize,
    const Position2D strides, const PaddingType padding,
    const MaximumImplementationType formulation) {
  TFOPT_ASSIGN_OR_RETURN(Shape output_shape,
                         OutputShape(input_shape, ksize, strides, padding));
  return MaxpoolOperation(std::move(op_name), std::move(input_shape),
                          std::move(output_shape), ksize, strides, padding,
                          formulation);
}

absl::StatusOr<MaxpoolOperation> MaxpoolOperation::GenericCreate(
    std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
    const Options& options) {
  OperationValidator validator("MaxpoolOperation", op_name);
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectInputSizeEquals(input_shapes.size(), 1));
  TFOPT_RETURN_IF_ERROR(validator.ExpectOptionsSizeAtMost(options.size(), 6));

  TFOPT_ASSIGN_OR_RETURN(
      const int stride_row,
      validator.IntegerOption(options, kOptionsStrideRowKey));
  TFOPT_ASSIGN_OR_RETURN(
      const int stride_col,
      validator.IntegerOption(options, kOptionsStrideColKey));
  TFOPT_ASSIGN_OR_RETURN(
      const int window_height,
      validator.IntegerOption(options, kOptionsWindowHeightKey));
  TFOPT_ASSIGN_OR_RETURN(
      const int window_width,
      validator.IntegerOption(options, kOptionsWindowWidthKey));
  TFOPT_ASSIGN_OR_RETURN(const std::string& padding_name,
                         validator.StringOption(options, kOptionsPaddingKey));

  MaximumImplementationType formulation = kDefaultMaximum;
  {
    const std::string formulation_name =
        ::gtl::FindWithDefault(options.string_options, kOptionsFormulationKey,
                               kOptionsFormulationDefault);
    if (formulation_name != kOptionsFormulationDefault &&
        !formulation_name.empty()) {
      if (!MaximumImplFromString(formulation_name, &formulation)) {
        return validator.OperationValidationError(absl::StrCat(
            "Unrecognized formulation name for maximum: ", formulation_name));
      }
    }
  }
  PaddingType padding;
  if (!PaddingTypeFromString(padding_name, &padding)) {
    return validator.OperationValidationError(
        absl::StrCat("Invalid padding string", padding_name));
  }

  TFOPT_ASSIGN_OR_RETURN(
      MaxpoolOperation op,
      Create(std::move(op_name), input_shapes[0],
             Position2D(window_height, window_width),
             Position2D(stride_row, stride_col), padding, formulation),
      _ << validator.base_error_message());
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectOutputShapeEquals(op.output_shape(), output_shape));
  return op;
}

proto::TensorNode MaxpoolOperation::ToProto(
    const std::vector<std::string>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  proto::TensorNode result;
  result.set_name(name());
  result.set_op_type(proto::OpType::MAX_POOL);
  *result.mutable_out_dimension() = output_shape().AsProto();
  result.add_input_names(inputs[0]);
  proto::Options::StringOption& padding_option =
      *result.mutable_options()->add_string_options();
  padding_option.set_name(kOptionsPaddingKey);
  padding_option.set_value(ToString(padding()));

  proto::Options::IntegerOption& window_height_option =
      *result.mutable_options()->add_integer_options();
  window_height_option.set_name(kOptionsWindowHeightKey);
  window_height_option.set_value(ksize_.row);

  proto::Options::IntegerOption& window_width_option =
      *result.mutable_options()->add_integer_options();
  window_width_option.set_name(kOptionsWindowWidthKey);
  window_width_option.set_value(ksize_.col);

  proto::Options::IntegerOption& stride_row_option =
      *result.mutable_options()->add_integer_options();
  stride_row_option.set_name(kOptionsStrideRowKey);
  stride_row_option.set_value(stride_.row);

  proto::Options::IntegerOption& stride_col_option =
      *result.mutable_options()->add_integer_options();
  stride_col_option.set_name(kOptionsStrideColKey);
  stride_col_option.set_value(stride_.col);

  if (formulation_ != kDefaultMaximum) {
    proto::Options::StringOption* formulation_option =
        result.mutable_options()->add_string_options();
    formulation_option->set_name(kOptionsFormulationKey);
    formulation_option->set_value(ToString(formulation_));
  }

  result.set_output_type(proto::TensorNode::FLOAT32);
  return result;
}

}  // namespace tf_opt
