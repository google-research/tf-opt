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

#include "tf_opt/neural_net/ops/relu_operation.h"

#include <utility>

#include "ortools/base/logging.h"
#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation_validator.h"
#include "tf_opt/open_source/status_macros.h"
#include "ortools/base/map_util.h"

namespace tf_opt {

constexpr const char ReluOperation::kOptionsFormulationKey[];
constexpr const char ReluOperation::kOptionsFormulationDefault[];

const char* ReluOperation::OptionsFormulationBigM() {
  return ToString(ReluImplementationType::kBigM);
}

const char* ReluOperation::OptionsFormulationMultipleChoice() {
  return ToString(ReluImplementationType::kMultipleChoice);
}

const char* ReluOperation::OptionsFormulationMultipleChoiceSimplified() {
  return ToString(ReluImplementationType::kMultipleChoiceSimplified);
}

ReluOperation::ReluOperation(std::string op_name, Shape input_shape,
                             const ReluImplementationType formulation)
    : Operation(std::move(op_name), {input_shape}, input_shape),
      formulation_(formulation) {}

absl::StatusOr<ReluOperation> ReluOperation::Create(
    std::string op_name, Shape input_shape,
    const ReluImplementationType formulation) {
  return ReluOperation(op_name, input_shape, formulation);
}

absl::StatusOr<ReluOperation> ReluOperation::GenericCreate(
    std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
    const Options& options) {
  OperationValidator validator("ReluOperation", op_name);
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectInputSizeEquals(input_shapes.size(), 1));
  TFOPT_RETURN_IF_ERROR(validator.ExpectOptionsSizeAtMost(options.size(), 1));
  TFOPT_RETURN_IF_ERROR(
      validator.ExpectOutputShapeEquals(output_shape, input_shapes[0]));
  ReluImplementationType formulation = kDefaultRelu;
  {
    // If the formulation is set in options, override the default
    const std::string formulation_name =
        ::gtl::FindWithDefault(options.string_options, kOptionsFormulationKey,
                               kOptionsFormulationDefault);
    if (formulation_name != kOptionsFormulationDefault &&
        !formulation_name.empty()) {
      if (!ReluImplFromString(formulation_name, &formulation)) {
        return validator.OperationValidationError(
            absl::StrCat("Unrecognized formulation name for clipped relu: ",
                         formulation_name));
      }
    }
  }
  return Create(std::move(op_name), std::move(input_shapes[0]), formulation);
}

proto::TensorNode ReluOperation::ToProto(
    const std::vector<std::string>& inputs) const {
  CHECK_EQ(inputs.size(), 1);
  proto::TensorNode result;
  result.set_name(name());
  result.set_op_type(proto::OpType::RELU);
  *result.mutable_out_dimension() = output_shape().AsProto();
  result.add_input_names(inputs[0]);
  if (formulation() != kDefaultRelu) {
    proto::Options::StringOption* formulation_option =
        result.mutable_options()->add_string_options();
    formulation_option->set_name(kOptionsFormulationKey);
    formulation_option->set_value(ToString(formulation()));
  }
  result.set_output_type(proto::TensorNode::FLOAT32);
  return result;
}

}  // namespace tf_opt
