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

#include "tf_opt/neural_net/ops/squeeze_operation.h"

#include "ortools/base/logging.h"
#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation_validator.h"
#include "tf_opt/open_source/status_macros.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor.h"
#include "ortools/base/map_util.h"

namespace tf_opt {
namespace {

std::vector<int> ConvertToInts(const std::vector<int64_t>& int64s) {
  std::vector<int> ints;
  ints.reserve(int64s.size());
  for (const int64_t i64 : int64s) {
    ints.push_back(static_cast<int>(i64));
  }
  return ints;
}
}  // namespace

constexpr const char SqueezeOperation::kOptionsAxesKey[];

absl::StatusOr<SqueezeOperation> SqueezeOperation::Create(
    std::string op_name, Shape input_shape, std::vector<int> axes) {
  OperationValidator validator("SqueezeOperation", op_name);
  Shape result_shape;
  if (axes.empty()) {
    // Never fails
    result_shape = internal::SqueezeShape(input_shape);
  } else {
    TFOPT_ASSIGN_OR_RETURN(result_shape,
                           internal::SqueezeShape(input_shape, axes),
                           _ << validator.base_error_message());
  }
  return SqueezeOperation(std::move(op_name), std::move(input_shape),
                          std::move(result_shape), std::move(axes));
}

absl::StatusOr<SqueezeOperation> SqueezeOperation::GenericCreate(
    std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
    const Options& options) {
  OperationValidator validator("SqueezeOperation", op_name);
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectInputSizeEquals(input_shapes.size(), 1));
  TFOPT_RETURN_IF_ERROR(validator.ExpectOptionsSizeAtMost(options.size(), 1));
  std::vector<int> axes = ConvertToInts(
      ::gtl::FindWithDefault(options.integer_list_options, kOptionsAxesKey));
  TFOPT_ASSIGN_OR_RETURN(
      SqueezeOperation op,
      Create(std::move(op_name), std::move(input_shapes[0]), std::move(axes)));
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectOutputShapeEquals(op.output_shape(), output_shape));
  return op;
}

proto::TensorNode SqueezeOperation::ToProto(
    const std::vector<std::string>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  proto::TensorNode result;
  result.set_name(name());
  result.set_op_type(proto::OpType::SQUEEZE);
  *result.mutable_out_dimension() = output_shape().AsProto();
  result.add_input_names(inputs[0]);

  if (!axes_.empty()) {
    proto::Options::IntegerListOption& axes_option =
        *result.mutable_options()->add_integer_list_options();
    axes_option.set_name(kOptionsAxesKey);
    for (const int v : axes_) {
      axes_option.add_value(v);
    }
  }

  result.set_output_type(proto::TensorNode::FLOAT32);
  return result;
}

}  // namespace tf_opt
