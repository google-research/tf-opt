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

#ifndef TF_OPT_TENSOR_SHAPE_H_
#define TF_OPT_TENSOR_SHAPE_H_

#include <cstdint>
#include <iostream>
#include <vector>

#include "ortools/base/logging.h"
#include "absl/types/span.h"
#include "tf_opt/neural_net/neural_net.pb.h"
#include "tf_opt/tensor/tensor.pb.h"

namespace tf_opt {

// The shape of a rectangular multidimensional array.
//
// Convertible from/to equivalent proto::Dimension.
//
// The empty shape is interpreted as a scalar.
class Shape {
 public:
  // An empty shape (a scalar).
  Shape();

  // Required:
  //  * dimension_sizes[i] >= 0 for all i,
  //  * prod_i dimension_sizes[i] does not overflow.
  explicit Shape(std::vector<int64_t> dimension_sizes);

  // Creates a shape from the dimension proto.
  // Same requirements as above on proto_dimension.dim_sizes().
  //
  // Deprecated, prefer ShapeProto below.
  explicit Shape(const proto::Dimension& proto_dimension);

  explicit Shape(const ShapeProto& shape_proto);

  // Converts from C++ object to equivalent proto.
  // Deprecated prefer ShapeProto version below.
  proto::Dimension AsProto() const;
  ShapeProto AsShapeProto() const;

  bool MultiIndexIsValid(absl::Span<const int64_t> multi_index) const;

  // Given an index into each component "multi_index", computes the equivalent
  // single index for a flat array data structure (in row-major order).
  //
  // "multi_index" must be within bounds for this, i.e. for all i:
  // 0 <= multi_index[i] < dim_value(i)
  //
  // TODO: Investigate replacing with absl::Span (warning, is SWIGed).
  int64_t FlattenIndex(const std::vector<int64_t>& multi_index) const {
    return FlattenIndexSpan(multi_index);
  }

  // TODO: Merge this with FlattenIndex.
  int64_t FlattenIndexSpan(absl::Span<const int64_t> multi_index) const;

  // Inverse operation of FlattenIndex.
  std::vector<int64_t> ExpandIndex(int64_t flat_index) const;

  int64_t num_dimensions() const { return dimension_sizes_.size(); }
  const std::vector<int64_t>& dimension_sizes() const {
    return dimension_sizes_;
  }
  int64_t dimension_size(int i) const { return dimension_sizes_[i]; }

  // The number of possible values of the multi-dimensional index.
  int64_t size() const { return size_; }

  // True when dimension_sizes_ == other.dimension_sizes_
  bool operator==(const Shape& other) const;
  bool operator!=(const Shape& other) const;

  // A human readable representation of this.
  std::string ToString() const;

  template <typename T>
  static Shape FromVector(const std::vector<T>& vector) {
    return Shape({static_cast<int64_t>(vector.size())});
  }

  // Will CHECK fail if vector2d is ragged.
  template <typename T>
  static Shape FromVector2D(const std::vector<std::vector<T>>& vector2d) {
    Shape result(
        {static_cast<int64_t>(vector2d.size()),
         vector2d.empty() ? 0 : static_cast<int64_t>(vector2d[0].size())});
    for (const std::vector<T>& row : vector2d) {
      CHECK_EQ(row.size(), result.dimension_size(1));
    }
    return result;
  }

  // Will CHECK fail if vector3d is ragged.
  template <typename T>
  static Shape FromVector3D(
      const std::vector<std::vector<std::vector<T>>>& vector3d) {
    Shape result(
        {static_cast<int64_t>(vector3d.size()),
         vector3d.empty() ? 0 : static_cast<int64_t>(vector3d[0].size()),
         vector3d.empty() || vector3d[0].empty()
             ? 0
             : static_cast<int64_t>(vector3d[0][0].size())});
    for (const std::vector<std::vector<T>>& matrix : vector3d) {
      CHECK_EQ(matrix.size(), result.dimension_size(1));
      for (const std::vector<T>& row : matrix) {
        CHECK_EQ(row.size(), result.dimension_size(2));
      }
    }
    return result;
  }

 private:
  std::vector<int64_t> dimension_sizes_;
  int64_t size_;
};

std::ostream& operator<<(std::ostream& stream, const Shape& dimension);

}  // namespace tf_opt

#endif  // TF_OPT_TENSOR_SHAPE_H_
