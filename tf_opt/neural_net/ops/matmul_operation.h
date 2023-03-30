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

#ifndef TF_OPT_SHARED_OPS_MATMUL_OPERATION_H_
#define TF_OPT_SHARED_OPS_MATMUL_OPERATION_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "tf_opt/neural_net/neural_net.pb.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_visitor.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {

// Matrix multiplies two tensors.
class MatmulOperation : public Operation {
 public:
  static absl::StatusOr<MatmulOperation> Create(std::string op_name,
                                                Shape left_shape,
                                                Shape right_shape);

  // Expected input format:
  //   input_shapes: The shapes of the tensors to multiply (two of them).
  //   output_shape: The shape to produce, follows broadcasting rules.
  //   options: Should be empty.
  static absl::StatusOr<MatmulOperation> GenericCreate(
      std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
      const Options& options);

  const Shape& left() const { return input_shape(0); }

  const Shape& right() const { return input_shape(1); }

  void Accept(OperationVisitor* visitor) const override {
    visitor->Visit(*this);
  }

  proto::TensorNode ToProto(
      const std::vector<std::string>& inputs) const override;

 private:
  MatmulOperation(std::string op_name, std::vector<Shape> input_shapes,
                  Shape output_shape);
};

}  // namespace tf_opt

#endif  // TF_OPT_SHARED_OPS_MATMUL_OPERATION_H_
