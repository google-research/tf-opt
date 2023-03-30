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

#ifndef TF_OPT_SHARED_OPS_SLICE_OPERATION_H_
#define TF_OPT_SHARED_OPS_SLICE_OPERATION_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_visitor.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {

// Extracts a rectangular subtensor from the input tensor.
//
// E.g. for
//   input =  [[10, 11, 12],
//             [13, 14, 15],
//             [16, 17, 18]]
//   begin = [1, 1]
//   size = [2, 2]
// the output is:
//   [[14, 15],
//    [17, 18]].
class SliceOperation : public Operation {
 public:
  static constexpr const char kOptionsBeginKey[] = "begin";
  static constexpr const char kOptionsSizeKey[] = "size";

  static absl::StatusOr<SliceOperation> Create(std::string op_name,
                                               Shape input_shape,
                                               std::vector<int64_t> begin,
                                               std::vector<int64_t> size);

  // Expected input format:
  //   input_shapes: The shape of the tensor to slice,
  //       input_shapes.size() == 1.
  //   output_shape: The shape to produce, must match options[size].
  //   options: Must contain two IntegerLists, keyed on "begin" and "size",
  //      and the size of each list must be equal to the rank of the input.
  static absl::StatusOr<SliceOperation> GenericCreate(
      std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
      const Options& options);

  const Shape& input() const { return input_shape(0); }
  const std::vector<int64_t>& begin() const { return begin_; }
  const std::vector<int64_t>& sizes() const { return sizes_; }

  void Accept(OperationVisitor* visitor) const override {
    visitor->Visit(*this);
  }

  proto::TensorNode ToProto(
      const std::vector<std::string>& inputs) const override;

 private:
  SliceOperation(std::string op_name, Shape input_shape, Shape output_shape,
                 std::vector<int64_t> begin, std::vector<int64_t> sizes)
      : Operation(std::move(op_name), {std::move(input_shape)},
                  std::move(output_shape)),
        begin_(std::move(begin)),
        sizes_(std::move(sizes)) {}

  std::vector<int64_t> begin_;
  std::vector<int64_t> sizes_;
};

}  // namespace tf_opt

#endif  // TF_OPT_SHARED_OPS_SLICE_OPERATION_H_
