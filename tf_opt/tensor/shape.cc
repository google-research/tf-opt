// Copyright 2020 The tf.opt Authors.
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

#include "tf_opt/tensor/shape.h"

#include <cstdint>
#include <limits>

#include "glog/logging.h"
#include "absl/strings/str_join.h"

namespace tf_opt {

Shape::Shape() : size_(1) {}

Shape::Shape(std::vector<int64_t> dimension_sizes)
    : dimension_sizes_(std::move(dimension_sizes)) {
  size_ = 1;
  for (const int64_t dim_size : dimension_sizes_) {
    CHECK_GE(dim_size, 0);
    // Check for overflow.
    CHECK(dim_size == 0 ||
          size_ <= std::numeric_limits<int64_t>::max() / dim_size);
    size_ *= dim_size;
  }
}

Shape::Shape(const proto::Dimension& proto_dimension)
    : Shape(std::vector<int64_t>(proto_dimension.dim_sizes().begin(),
                                 proto_dimension.dim_sizes().end())) {}

Shape::Shape(const ShapeProto& shape_proto)
    : Shape(std::vector<int64_t>(shape_proto.dimensions().begin(),
                                 shape_proto.dimensions().end())) {}

proto::Dimension Shape::AsProto() const {
  proto::Dimension proto;
  for (const int64_t dim_size : dimension_sizes_) {
    proto.add_dim_sizes(dim_size);
  }
  return proto;
}

ShapeProto Shape::AsShapeProto() const {
  ShapeProto result;
  *result.mutable_dimensions() = {dimension_sizes_.begin(),
                                  dimension_sizes_.end()};
  return result;
}

bool Shape::MultiIndexIsValid(absl::Span<const int64_t> multi_index) const {
  if (multi_index.size() != dimension_sizes_.size()) {
    return false;
  }
  for (int i = 0; i < multi_index.size(); ++i) {
    if (multi_index[i] < 0 || multi_index[i] >= dimension_sizes_[i]) {
      return false;
    }
  }
  return true;
}

int64_t Shape::FlattenIndexSpan(absl::Span<const int64_t> multi_index) const {
  CHECK_EQ(multi_index.size(), dimension_sizes_.size());
  int64_t ans = 0;
  int64_t multiplier = 1;
  int i = multi_index.size() - 1;
  for (auto it = multi_index.rbegin(); it != multi_index.rend(); ++it) {
    const int64_t value = *it;
    CHECK_GE(value, 0);
    CHECK_LT(value, dimension_sizes_[i]);
    ans += multiplier * value;
    multiplier *= dimension_sizes_[i];
    i--;
  }
  return ans;
}

std::vector<int64_t> Shape::ExpandIndex(const int64_t flat_index) const {
  CHECK_GE(flat_index, 0);
  CHECK_LT(flat_index, size_);
  std::vector<int64_t> multi_index;
  multi_index.reserve(dimension_sizes_.size());
  int flat_index_remaining = flat_index;
  for (auto it = dimension_sizes_.rbegin(); it != dimension_sizes_.rend();
       ++it) {
    const int64_t dimension_size = *it;
    multi_index.push_back(flat_index_remaining % dimension_size);
    flat_index_remaining /= dimension_size;
  }
  std::reverse(multi_index.begin(), multi_index.end());
  return multi_index;
}

bool Shape::operator==(const Shape& other) const {
  return dimension_sizes_ == other.dimension_sizes();
}

bool Shape::operator!=(const Shape& other) const { return !(*this == other); }

std::string Shape::ToString() const {
  return absl::StrJoin(dimension_sizes_, ",");
}

std::ostream& operator<<(std::ostream& stream, const Shape& dimension) {
  stream << dimension.ToString();
  return stream;
}

}  // namespace tf_opt
