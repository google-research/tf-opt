// Copyright 2022 The tf.opt Authors.
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

#ifndef TF_OPT_SHARED_OPS_CONSTANT_OPERATION_H_
#define TF_OPT_SHARED_OPS_CONSTANT_OPERATION_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tf_opt/neural_net/neural_net.pb.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_visitor.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor.h"

namespace tf_opt {

class ConstantOperation : public Operation {
 public:
  ConstantOperation(std::string op_name, DoubleTensor value);

  static absl::StatusOr<ConstantOperation> Create(std::string op_name,
                                                  DoubleTensor value);

  // Not supported for ConstantOperation, will CHECK fail.  As implemented, a
  // ConstantOperation owns the DoubleTensor that is its data, but there is no
  // way to wire this in through GenericInitialize.
  static absl::StatusOr<ConstantOperation> GenericCreate(
      std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
      const Options& options);

  const DoubleTensor& value() const { return value_; }

  void Accept(OperationVisitor* visitor) const override {
    visitor->Visit(*this);
  }

  // Always fails, ConstantOperation writes to ParameterValue instead.
  // TODO: ConstantOperation should just serialize to a TensorNode.
  proto::TensorNode ToProto(
      const std::vector<std::string>& inputs) const override;
  proto::ParameterValue ToProto() const;

 private:
  DoubleTensor value_;
};

}  // namespace tf_opt

#endif  // TF_OPT_SHARED_OPS_CONSTANT_OPERATION_H_
