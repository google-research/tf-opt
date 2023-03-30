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

#include "tf_opt/tensor/tensor.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"

namespace tf_opt {

using absl::StrCat;

DoubleTensor ProtoToDoubleTensor(
    const proto::ParameterValue& double_tensor_proto) {
  DoubleTensor result(Shape(double_tensor_proto.dimension()));
  CHECK_EQ(result.size(), double_tensor_proto.value_size());
  result.mutable_flat_values()->assign(double_tensor_proto.value().begin(),
                                       double_tensor_proto.value().end());
  return result;
}

DoubleTensor ProtoToDoubleTensor(const DoubleTensorProto& double_tensor_proto) {
  DoubleTensor result(Shape(double_tensor_proto.shape()));
  CHECK_EQ(result.size(), double_tensor_proto.values_size());
  result.mutable_flat_values()->assign(double_tensor_proto.values().begin(),
                                       double_tensor_proto.values().end());
  return result;
}

void DoubleTensorToProto(const DoubleTensor& double_tensor,
                         proto::ParameterValue* double_tensor_proto) {
  CHECK_NE(double_tensor_proto, nullptr);
  *double_tensor_proto->mutable_dimension() =
      double_tensor.dimension().AsProto();
  double_tensor_proto->clear_value();
  for (const double v : double_tensor.flat_values()) {
    double_tensor_proto->add_value(v);
  }
}

DoubleTensorProto DoubleTensorToProto(const DoubleTensor& double_tensor) {
  DoubleTensorProto result;
  *result.mutable_shape() = double_tensor.dimension().AsShapeProto();
  *result.mutable_values() = {double_tensor.flat_values().begin(),
                              double_tensor.flat_values().end()};
  return result;
}

BoundsTensor DoubleTensorToBoundsTensor(const DoubleTensor& double_tensor) {
  BoundsTensor result(double_tensor.dimension());
  for (int i = 0; i < double_tensor.size(); i++) {
    (*result.mutable_flat_values())[i] = Bounds(double_tensor.flat_values()[i]);
  }
  return result;
}

namespace internal {

Shape SqueezeShape(const Shape& input_shape) {
  std::vector<int64_t> result_shape;
  for (const int64_t dim_size : input_shape.dimension_sizes()) {
    if (dim_size != 1) {
      result_shape.push_back(dim_size);
    }
  }
  return Shape(result_shape);
}

absl::StatusOr<Shape> SqueezeShape(const Shape& input_shape,
                                   absl::Span<const int> axes) {
  if (axes.empty()) {
    return absl::InvalidArgumentError(
        "Cannot call Squeeze(axes) with an empty axes list.");
  }
  for (const int axis : axes) {
    if (axis < 0 || axis >= input_shape.num_dimensions()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Cannot squeeze shape ", input_shape.ToString(), " on axes: [",
          absl::StrJoin(axes, ", "), "], all squeezed axes must fall in [0, ",
          input_shape.num_dimensions(), "), but found axis: ", axis));
    }
    if (input_shape.dimension_size(axis) != 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot squeeze shape ", input_shape.ToString(),
                       " on axes: [", absl::StrJoin(axes, ", "),
                       "], all squeezed axes must have dimension size of 1, "
                       "but dimension size of axis ",
                       axis, " is ", input_shape.dimension_size(axis)));
    }
  }
  std::vector<bool> retained(input_shape.num_dimensions(), true);
  for (const int a : axes) {
    retained[a] = false;
  }
  std::vector<int64_t> result_shape;
  for (int d = 0; d < input_shape.num_dimensions(); ++d) {
    if (retained[d]) {
      result_shape.push_back(input_shape.dimension_size(d));
    }
  }
  return Shape(result_shape);
}

absl::StatusOr<Shape> ExpandDimsShape(const Shape& input_shape, int axis) {
  if (axis < 0 || axis > input_shape.num_dimensions()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "To call ExpandDims on a tensor of shape: ", input_shape.ToString(),
        ", axis must lie in [0, ", input_shape.num_dimensions(),
        "], but found: ", axis));
  }
  std::vector<int64_t> result_shape;
  if (axis == 0) {
    result_shape.push_back(1);
  }
  for (int d = 0; d < input_shape.num_dimensions(); ++d) {
    result_shape.push_back(input_shape.dimension_size(d));
    if (d + 1 == axis) {
      result_shape.push_back(1);
    }
  }
  return Shape(result_shape);
}

absl::StatusOr<Shape> SliceShape(const Shape& input_shape,
                                 absl::Span<const int64_t> begin_indices,
                                 absl::Span<const int64_t> sizes) {
  if (begin_indices.size() != input_shape.num_dimensions()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "begin_indices has ", begin_indices.size(), " dimensions != ",
        input_shape.num_dimensions(), " dimensions on input_dimension."));
  }
  if (sizes.size() != input_shape.num_dimensions()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "sizes has ", sizes.size(), " dimensions != ",
        input_shape.num_dimensions(), " dimensions on input_dimension."));
  }
  for (int d = 0; d < input_shape.num_dimensions(); ++d) {
    if (begin_indices[d] < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("begin_indices[", d, "] = ", begin_indices[d],
                       " < 0, must be nonnegative."));
    }
    if (sizes[d] < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "sizes[", d, "] = ", sizes[d], " < 0, must be nonnegative."));
    }
    if (begin_indices[d] + sizes[d] > input_shape.dimension_size(d)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "begin_indices[", d, "] + sizes[", d,
          "] = ", begin_indices[d] + sizes[d], " > input_dimension[", d,
          "] = ", input_shape.dimension_size(d),
          " requesting out of bounds indices in tensor slice."));
    }
  }
  return Shape(std::vector<int64_t>(sizes.begin(), sizes.end()));
}

Shape SubTensorShape(const Shape& input_shape, int start, int size) {
  CHECK_GE(input_shape.num_dimensions(), 1)
      << "SubTensor() cannot be called on scalars.";
  CHECK_GE(input_shape.dimension_size(0), start + size)
      << "start=" << start << " + size= " << size
      << " exceeds first dimension of tensor with shape: "
      << input_shape.ToString();
  std::vector<int64_t> output_dimensions = input_shape.dimension_sizes();
  output_dimensions[0] = size;
  return Shape(output_dimensions);
}

}  // namespace internal

bool HasInfiniteOrNan(const DoubleTensor& tensor) {
  for (const double d : tensor.flat_values()) {
    if (!std::isfinite(d)) {
      return true;
    }
  }
  return false;
}

}  // namespace tf_opt
