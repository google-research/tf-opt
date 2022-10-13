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

#ifndef TF_OPT_SHARED_OPS_VARIABLE_OPERATION_H_
#define TF_OPT_SHARED_OPS_VARIABLE_OPERATION_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tf_opt/neural_net/neural_net.pb.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_visitor.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {

// Creates an input to the function, with a value that can be plugged in later
// or optimized over.
class VariableOperation : public Operation {
 public:
  VariableOperation(std::string op_name, Shape shape);

  static absl::StatusOr<VariableOperation> Create(std::string op_name,
                                                  Shape shape);

  // Expected input format:
  //   input_shapes: Should be empty.
  //   output_shape: A single shape, the shape of the variable to create.
  //   options: Should be empty.
  static absl::StatusOr<VariableOperation> GenericCreate(
      std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
      const Options& options);

  void Accept(OperationVisitor* visitor) const override {
    visitor->Visit(*this);
  }

  proto::TensorNode ToProto(
      const std::vector<std::string>& inputs) const override;
};

}  // namespace tf_opt

#endif  // TF_OPT_SHARED_OPS_VARIABLE_OPERATION_H_
