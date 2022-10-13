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

#ifndef TF_OPT_SHARED_OPS_EMBEDDING_LOOKUP_OPERATION_H_
#define TF_OPT_SHARED_OPS_EMBEDDING_LOOKUP_OPERATION_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_visitor.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {

// Inputs are:
//   (1) params: The weights to look up from. The first dimension should be
//       equal to the number of classes in the embedding.
//   (2) ids: The final dimension is the number of classes in the embedding.
//       The tensor should be one-hot in this dimension. Every such "vector" of
//       size "number of classes" is an embedding lookup to do. The input must
//       be at least of rank two.
//
// In typical use, you will have:
//   * params shape: [num_classes, embedding_dimension],
//   * ids shape: [1, num_lookups, num_classes],
//   * result shape: [1, num_lookups, embedding_dimension],
// where:
//   num_lookups: How many in elements to lookup from the embedding,
//   num_classes: The number of different elements that can be looked up,
//   embedding_dimension: The size of the output vector from each lookup,
//   (and 1 above is the batch size).
class EmbeddingLookupOperation : public Operation {
 public:
  static absl::StatusOr<Shape> OutputShape(const Shape& params_shape,
                                           const Shape& ids_shape);

  static absl::StatusOr<EmbeddingLookupOperation> Create(std::string op_name,
                                                         Shape params_shape,
                                                         Shape ids_shape);

  // Input format:
  //   input_shapes: Contains the shapes of:
  //     (1) the params tensor (the weights to look up)
  //     (2) the ids tensor (indices in the params tensor to get values from).
  //   output_shape: Of shape ids[:-1] + params[1:].
  //   options: Empty.
  static absl::StatusOr<EmbeddingLookupOperation> GenericCreate(
      std::string op_name, std::vector<Shape> input_shapes, Shape output_shape,
      const Options& options);

  const Shape& params() const { return input_shape(0); }

  const Shape& ids() const { return input_shape(1); }

  void Accept(OperationVisitor* visitor) const override {
    visitor->Visit(*this);
  }

  proto::TensorNode ToProto(
      const std::vector<std::string>& inputs) const override;

 private:
  EmbeddingLookupOperation(std::string op_name, std::vector<Shape> input_shapes,
                           Shape output_shape);
};

}  // namespace tf_opt

#endif  // TF_OPT_SHARED_OPS_EMBEDDING_LOOKUP_OPERATION_H_
