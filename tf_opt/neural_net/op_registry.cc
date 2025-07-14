// Copyright 2025 The tf.opt Authors.
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

#include "tf_opt/neural_net/op_registry.h"

#include <memory>
#include <utility>

#include "ortools/base/logging.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "tf_opt/neural_net/ops/all_operations.h"
#include "tf_opt/open_source/status_macros.h"

namespace tf_opt {
namespace op_registry {

namespace {

template <typename ReturnType, typename Transformer, typename... Args>
ReturnType ApplyForOpType(const proto::OpType op_type, const Transformer& t,
                          Args... args) {
#define TFOPT_OP_CASE(op_type_val, op_class) \
  case op_type_val:                          \
    return t.template apply<op_class>(std::forward<Args>(args)...)

  switch (op_type) {
    TFOPT_OP_CASE(proto::ADD, AddOperation);
    TFOPT_OP_CASE(proto::SUBTRACT, SubtractOperation);
    TFOPT_OP_CASE(proto::MULTIPLY, MultiplyOperation);
    TFOPT_OP_CASE(proto::DIVIDE, DivideOperation);
    TFOPT_OP_CASE(proto::CLIPPED_RELU, ClippedReluOperation);
    TFOPT_OP_CASE(proto::CONCAT, ConcatOperation);
    TFOPT_OP_CASE(proto::CONV1D, Conv1dOperation);
    TFOPT_OP_CASE(proto::CONV2D, Conv2dOperation);
    TFOPT_OP_CASE(proto::EXPAND_DIMS, ExpandDimsOperation);
    TFOPT_OP_CASE(proto::MAT_MUL, MatmulOperation);
    TFOPT_OP_CASE(proto::MAX_POOL, MaxpoolOperation);
    TFOPT_OP_CASE(proto::EMBEDDING_LOOKUP, EmbeddingLookupOperation);
    TFOPT_OP_CASE(proto::RELU, ReluOperation);
    TFOPT_OP_CASE(proto::RESHAPE, ReshapeOperation);
    TFOPT_OP_CASE(proto::REDUCE_MAX, ReduceMaxOperation);
    TFOPT_OP_CASE(proto::REDUCE_MIN, ReduceMinOperation);
    TFOPT_OP_CASE(proto::REDUCE_MEAN, ReduceMeanOperation);
    TFOPT_OP_CASE(proto::REDUCE_SUM, ReduceSumOperation);
    TFOPT_OP_CASE(proto::SLICE, SliceOperation);
    TFOPT_OP_CASE(proto::SQUEEZE, SqueezeOperation);
    TFOPT_OP_CASE(proto::INPUT, VariableOperation);
    default:
      LOG(FATAL) << "No implementation found for op_type: " << op_type;
  }
#undef TFOPT_OP_CASE
  CHECK(false);
}

struct OpFactory {
  template <typename OpClass>
  absl::StatusOr<std::unique_ptr<Operation>> apply(
      std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
      const Operation::Options& options) const {
    TFOPT_ASSIGN_OR_RETURN(
        auto op,
        OpClass::GenericCreate(std::move(op_name), std::move(input_shapes),
                               std::move(output_shape), options));
    std::unique_ptr<Operation> result =
        std::make_unique<OpClass>(std::move(op));
    return result;
  }
};

}  // namespace

absl::StatusOr<std::unique_ptr<Operation>> MakeOperation(
    proto::OpType op_type, std::string op_name, std::vector<Shape> input_shapes,
    Shape output_shape, const Operation::Options& options) {
  OpFactory factory;
  return ApplyForOpType<absl::StatusOr<std::unique_ptr<Operation>>>(
      op_type, factory, std::move(op_name), std::move(input_shapes),
      std::move(output_shape), options);
}

}  // namespace op_registry
}  // namespace tf_opt
