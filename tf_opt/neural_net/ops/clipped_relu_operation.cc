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

#include "tf_opt/neural_net/ops/clipped_relu_operation.h"

#include <utility>

#include "ortools/base/logging.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tf_opt/neural_net/operation_validator.h"
#include "tf_opt/open_source/status_macros.h"
#include "ortools/base/map_util.h"

namespace tf_opt {

constexpr const char ClippedReluOperation::kOptionsCapKey[];
constexpr const char ClippedReluOperation::kOptionsFormulationKey[];
constexpr const char ClippedReluOperation::kOptionsFormulationDefault[];

const char* ClippedReluOperation::OptionsFormulationCompositeDirect() {
  return ToString(ClippedReluImplementationType::kCompositeDirect);
}
const char* ClippedReluOperation::OptionsFormulationCompositeExtended() {
  return ToString(ClippedReluImplementationType::kCompositeExtended);
}
const char* ClippedReluOperation::OptionsFormulationExtendedXExclusion() {
  return ToString(ClippedReluImplementationType::kExtendedXExclusion);
}
const char* ClippedReluOperation::OptionsFormulationExtendedYExclusion() {
  return ToString(ClippedReluImplementationType::kExtendedYExclusion);
}

const char* ClippedReluOperation::OptionsFormulationUnaryBigM() {
  return ToString(ClippedReluImplementationType::kUnaryBigM);
}

const char* ClippedReluOperation::OptionsFormulationIncrementalBigM() {
  return ToString(ClippedReluImplementationType::kIncrementalBigM);
}

ClippedReluOperation::ClippedReluOperation(
    std::string op_name, Shape input_shape, const double cap,
    const ClippedReluImplementationType formulation)
    : Operation(std::move(op_name), {input_shape}, input_shape),
      cap_(cap),
      formulation_(formulation) {}

absl::StatusOr<ClippedReluOperation> ClippedReluOperation::Create(
    std::string op_name, Shape input_shape, const double cap,
    const ClippedReluImplementationType formulation) {
  if (cap < 0) {
    OperationValidator validator("ClippedReluOperation", op_name);
    return validator.OperationValidationError(
        "Option cap must be nonnegative.");
  }
  return ClippedReluOperation(std::move(op_name), std::move(input_shape), cap,
                              formulation);
}

absl::StatusOr<ClippedReluOperation> ClippedReluOperation::GenericCreate(
    std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
    const Options& options) {
  OperationValidator validator("ClippedReluOperation", op_name);
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectInputSizeEquals(input_shapes.size(), 1));
  TFOPT_RETURN_IF_ERROR(validator.ExpectOptionsSizeAtMost(options.size(), 2));
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectOutputShapeEquals(output_shape, input_shapes[0]));
  TFOPT_ASSIGN_OR_RETURN(const double cap,
                         validator.DoubleOption(options, kOptionsCapKey));
  ClippedReluImplementationType formulation = kDefaultClippedRelu;
  {
    // If the formulation is set in options, override the default
    const std::string formulation_name =
        ::gtl::FindWithDefault(options.string_options, kOptionsFormulationKey,
                               kOptionsFormulationDefault);
    if (formulation_name != kOptionsFormulationDefault &&
        !formulation_name.empty()) {
      if (!ClippedReluImplFromString(formulation_name, &formulation)) {
        return validator.OperationValidationError(
            absl::StrCat("Unrecognized formulation name for clipped relu: ",
                         formulation_name));
      }
    }
  }
  return Create(std::move(op_name), std::move(input_shapes[0]), cap,
                formulation);
}

proto::TensorNode ClippedReluOperation::ToProto(
    const std::vector<std::string>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  proto::TensorNode result;
  result.set_name(name());
  result.set_op_type(proto::OpType::CLIPPED_RELU);
  *result.mutable_out_dimension() = output_shape().AsProto();
  result.add_input_names(inputs[0]);
  if (formulation() != kDefaultClippedRelu) {
    proto::Options::StringOption* formulation_option =
        result.mutable_options()->add_string_options();
    formulation_option->set_name(kOptionsFormulationKey);
    formulation_option->set_value(ToString(formulation()));
  }
  proto::Options::DoubleOption& cap_option =
      *result.mutable_options()->add_double_options();
  cap_option.set_name(kOptionsCapKey);
  cap_option.set_value(cap_);
  result.set_output_type(proto::TensorNode::FLOAT32);
  return result;
}

}  // namespace tf_opt
