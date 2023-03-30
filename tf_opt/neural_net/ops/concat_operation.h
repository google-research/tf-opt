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

#ifndef TF_OPT_SHARED_OPS_CONCAT_OPERATION_H_
#define TF_OPT_SHARED_OPS_CONCAT_OPERATION_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_visitor.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {

class ConcatOperation : public Operation {
 public:
  static constexpr const char kOptionsAxisKey[] = "axis";

  static absl::StatusOr<ConcatOperation> Create(std::string op_name,
                                                std::vector<Shape> input_shapes,
                                                int axis);

  // Expected input format:
  //   input_shapes: The shapes of a non-empty list of tensors that have the
  //     same number of dimensions, and the same size in all dimensions except
  //     for the dimension 'axis'.
  //   output_shape: The shape of the output tensor, as described in
  //     tf_opt/optimize/tensor/concat.h.
  //   options: Contains a single key, kOptionsAxisKey with value "axis".
  static absl::StatusOr<ConcatOperation> GenericCreate(
      std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
      const Options& options);

  void Accept(OperationVisitor* visitor) const override {
    visitor->Visit(*this);
  }

  int axis() const { return axis_; }

  proto::TensorNode ToProto(
      const std::vector<std::string>& inputs) const override;

 private:
  ConcatOperation(std::string op_name, std::vector<Shape> input_shapes,
                  Shape output_shape, int axis);

  int axis_;
};

}  // namespace tf_opt

#endif  // TF_OPT_SHARED_OPS_CONCAT_OPERATION_H_
