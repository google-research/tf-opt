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

#include "tf_opt/neural_net/ops/constant_operation.h"

#include <utility>

#include "ortools/base/logging.h"
#include "absl/status/statusor.h"
#include "tf_opt/open_source/status_macros.h"

namespace tf_opt {

ConstantOperation::ConstantOperation(std::string op_name, DoubleTensor value)
    : Operation(std::move(op_name), {}, value.dimension()),
      value_(std::move(value)) {}

absl::StatusOr<ConstantOperation> ConstantOperation::Create(
    std::string op_name, DoubleTensor value) {
  return ConstantOperation(std::move(op_name), std::move(value));
}

MaybeForGraph<ConstantOperation> ConstantOperation::CreateForGraph(
    std::string op_name, DoubleTensor value) {
  return FromMaybeCreated(Create(std::move(op_name), std::move(value)), {});
}

absl::StatusOr<ConstantOperation> ConstantOperation::GenericCreate(
    std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
    const Options& options) {
  LOG(FATAL) << "Cannot do generic initialization for constants, but attempted "
                "so for constant: "
             << op_name;
}

proto::TensorNode ConstantOperation::ToProto(
    const std::vector<std::string>& inputs) const {
  LOG(FATAL) << "Constant operations serialize to ParameterValue instead.";
}

proto::ParameterValue ConstantOperation::ToProto() const {
  proto::ParameterValue result;
  DoubleTensorToProto(value_, &result);
  result.set_name(name());
  return result;
}

}  // namespace tf_opt
