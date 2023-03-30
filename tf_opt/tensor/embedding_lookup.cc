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

#include "tf_opt/tensor/embedding_lookup.h"

#include <cstdint>

#include "absl/status/statusor.h"

namespace tf_opt {

absl::StatusOr<Shape> EmbeddingLookupOutputShape(const Shape& params_shape,
                                                 const Shape& ids_shape) {
  if (params_shape.num_dimensions() <= 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Rank of params must be at least two, found: ",
                     params_shape.num_dimensions()));
  }
  if (ids_shape.num_dimensions() <= 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Rank of ids must be at least two, found: ",
                     ids_shape.num_dimensions()));
  }
  const int64_t num_classes = ids_shape.dimension_size(
      static_cast<int>(ids_shape.num_dimensions() - 1));

  if (num_classes != params_shape.dimension_size(0)) {
    return absl::InvalidArgumentError("Incompatible ids and params shapes");
  }
  std::vector<int64_t> result_shape;
  result_shape.reserve(ids_shape.num_dimensions() +
                       params_shape.num_dimensions() - 2);
  for (int i = 0; i < ids_shape.num_dimensions() - 1; ++i) {
    result_shape.push_back(ids_shape.dimension_size(i));
  }
  for (int i = 1; i < params_shape.num_dimensions(); ++i) {
    result_shape.push_back(params_shape.dimension_size(i));
  }
  return Shape(result_shape);
}

}  // namespace tf_opt
