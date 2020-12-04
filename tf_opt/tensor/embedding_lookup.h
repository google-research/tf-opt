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

#ifndef TF_OPT_TENSOR_EMBEDDING_LOOKUP_H_
#define TF_OPT_TENSOR_EMBEDDING_LOOKUP_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor.h"

namespace tf_opt {

// For params_shape [num_classes, x1, ..., xm] and ids_shape
// [y1,...,yn, num_classes], returns the shape [y1,...,yn, x1,..., xm].
// Produces an error if either input has rank < 2, or if the inputs disagree on
// num_classes.
absl::StatusOr<Shape> EmbeddingLookupOutputShape(const Shape& params_shape,
                                                 const Shape& ids_shape);

// Inputs are:
//   (1) An embedding weights DoubleTensor.  The first dimension should be equal
//       to the number of classes in the embedding.
//   (2) The ids tensor, of rank >= 2, with the final dimension is the number of
//       classes in the embedding. Typically, this tensor is one-hot in the
//       final dimension (but this is not required). Multiple elements can be
//       looked up by having additional dimensions before the final dimension.
//
// Returns:
//    For weights with shape [num_classes, x1, ..., xm] and ids with shape
//    [y1,...,yn, num_classes], a new tensor with shape [y1,...,yn, x1,..., xm]
//    where:
//      result[i1,...,in, :] = sum_{k in classes} ids[i1,...in,k] * weights[k,:]
//
// In typical use, you will have:
//   * DoubleTensor weights shape: [num_classes, embedding_dimension],
//   * MPTensor shape: [1, num_lookups, num_classes],
//   * Result shape: [1, num_lookups, embedding_dimension],
// where:
//   num_lookups: How many in elements to lookup from the embedding,
//   num_classes: The number of different elements that can be looked up,
//   embedding_dimension: The size of the output vector from each lookup.
//
//
// Type requirements:
//   ResultType operator*(WeightType, IdType) must be defined.
//   ResultType operator+(ResultType, ResultType) must be defined.
// double, Bounds, and operations_research::LinearExpr meet all these criteria.
template <typename ResultType, typename WeightType, typename IdType>
Tensor<ResultType> EmbeddingLookup(const Tensor<WeightType>& embedding_weights,
                                   const Tensor<IdType>& ids);

// //////////////////////// Implementation Details /////////////////////////////

template <typename ResultType, typename WeightType, typename IdType>
Tensor<ResultType> EmbeddingLookup(const Tensor<WeightType>& embedding_weights,
                                   const Tensor<IdType>& ids) {
  const Shape out_shape =
      std::move(EmbeddingLookupOutputShape(embedding_weights.dimension(),
                                           ids.dimension()))
          .value();
  Tensor<ResultType> result(out_shape);
  // TODO: Rewrite this once we have generic tensor slicing, it will
  // be cleaner.
  const int64_t rank_from_ids = ids.dimension().num_dimensions() - 1;
  for (int64_t i = 0; i < result.size(); ++i) {
    const std::vector<int64_t> out_coords = out_shape.ExpandIndex(i);
    std::vector<int64_t> ids_slice_coords(out_coords.begin(),
                                          out_coords.begin() + rank_from_ids);
    ids_slice_coords.push_back(-1);
    std::vector<int64_t> weight_slice_coords = {-1};
    weight_slice_coords.insert(weight_slice_coords.end(),
                               out_coords.begin() + rank_from_ids,
                               out_coords.end());
    const std::vector<IdType> ids_slice = ids.VectorSlice(ids_slice_coords);
    const std::vector<WeightType> weight_slice =
        embedding_weights.VectorSlice(weight_slice_coords);
    CHECK_EQ(ids_slice.size(), weight_slice.size());
    for (int j = 0; j < weight_slice.size(); ++j) {
      (*result.mutable_flat_values())[i] += weight_slice[j] * ids_slice[j];
    }
  }
  return result;
}

}  // namespace tf_opt

#endif  // TF_OPT_TENSOR_EMBEDDING_LOOKUP_H_
