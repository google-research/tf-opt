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

#ifndef TF_OPT_TENSOR_TENSOR_H_
#define TF_OPT_TENSOR_TENSOR_H_

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "ortools/base/logging.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tf_opt/bounds/bounds.h"
#include "tf_opt/neural_net/neural_net.pb.h"
#include "tf_opt/tensor/shape.h"

namespace tf_opt {

// A multi-dimension rectangular array of type T. T must be default
// constructible and copyable. Is typically either double or LinearExpr.
//
// Implementation note: the underlying data structure is a single flat array
// with elements stored in row major order.
template <typename T>
class Tensor {
 public:
  // A scalar (rank 0) tensor with a single value that is the default for T.
  Tensor() : values_(1) {}

  // All values are initialized to the default value of T.
  explicit Tensor(Shape shape)
      : shape_(std::move(shape)), values_(shape_.size()) {}

  // Create a rank 0 zero with value "scalar_value".
  explicit Tensor(const T& scalar_value) : shape_(), values_({scalar_value}) {}

  // All values are initialized to fill_value.
  Tensor(Shape shape, const T& fill_value)
      : shape_(std::move(shape)), values_(shape_.size(), fill_value) {}

  // Make a vector of shape (values.size()).
  explicit Tensor(std::vector<T> value_vector)
      : shape_(Shape::FromVector(value_vector)),
        values_(std::move(value_vector)) {}

  // Make a matrix of shape (value_matrix.size(), value_matrix[0].size()). Will
  // CHECK fail unless value_matrix[i].size() is the same for all i.
  explicit Tensor(const std::vector<std::vector<T>>& value_matrix)
      : shape_(Shape::FromVector2D(value_matrix)), values_() {
    values_.reserve(shape_.size());
    for (const std::vector<T>& row : value_matrix) {
      values_.insert(values_.end(), row.begin(), row.end());
    }
  }

  static Tensor<T> CreateVector(const std::vector<T>& value_vector) {
    return Tensor<T>(value_vector);
  }

  static Tensor<T> CreateMatrix(
      const std::vector<std::vector<T>>& value_matrix) {
    return Tensor<T>(value_matrix);
  }

  // Create a tensor of shape "shape" with values in row major order of
  // "flat_data".  Pass by value of "flat_data" is intentional.
  static Tensor<T> FromFlatData(const Shape& shape, std::vector<T> flat_data) {
    CHECK_EQ(flat_data.size(), shape.size());
    Tensor result(std::move(flat_data));
    result.ReshapeInPlace(shape);
    return result;
  }

  // Make a tensor of shape (value_tensor.size(), value_tensor[0].size(),
  // value_tensor[0][0].size()). Will CHECK fail if input vector is ragged.
  explicit Tensor(const std::vector<std::vector<std::vector<T>>>& value_tensor)
      : shape_(Shape::FromVector3D(value_tensor)), values_() {
    values_.reserve(shape_.size());
    for (const std::vector<std::vector<T>>& matrix : value_tensor) {
      for (const std::vector<T>& row : matrix) {
        values_.insert(values_.end(), row.begin(), row.end());
      }
    }
  }

  // TODO: rename to shape().  This will break some template functions.
  const Shape& dimension() const { return shape_; }

  // The number of elements in the multidimensional array.
  int64_t size() const { return shape_.size(); }

  // Getter and setter for individual values in the array.
  // TODO: Investigate replacing with absl::Span (warning, is SWIGed).
  const T& value(const std::vector<int64_t>& index) const {
    return ValueSpan(index);
  }

  // TODO: Merge this with value.
  const T& ValueSpan(absl::Span<const int64_t> index) const {
    return values_.at(shape_.FlattenIndexSpan(index));
  }

  const T& flat_value(int64_t flat_index) const {
    return values_.at(flat_index);
  }

  void set_value(const std::vector<int64_t>& index, T value) {
    SetValueSpan(index, std::move(value));
  }

  // TODO: Merge this with set_value.
  void SetValueSpan(absl::Span<const int64_t> index, T value) {
    values_.at(shape_.FlattenIndexSpan(index)) = std::move(value);
  }

  void set_flat_value(int64_t flat_index, T value) {
    values_.at(flat_index) = std::move(value);
  }

  // A flat representation of the multi-dimensional array.
  const std::vector<T>& flat_values() const { return values_; }
  std::vector<T>* mutable_flat_values() { return &values_; }

  // Extracts a 1d vector from the tensor along one dimension. The argument
  // 'fixed_indices' should:
  //   * have length equal to shape_.num_dimensions(),
  //   * have exactly one "free dimension" indicated by a value of -1,
  //   * have the remaining values in [0, shape_.dimension_size(i)).
  //
  // Examples:
  // x = [[3, 5], [6, 8]]
  // x.VectorSlice([0, -1]) => [3, 5]
  // x.VectorSlice([1, -1]) => [6, 8]
  // x.VectorSlice([-1, 0]) => [3, 6]
  // x.VectorSlice([-1, 1]) => [5, 8]
  std::vector<T> VectorSlice(std::vector<int64_t> fixed_indices) const;

  // True if shape is the same and component-wise elements are ==.
  bool operator==(const Tensor<T>& other) const {
    return shape_ == other.dimension() && values_ == other.flat_values();
  }

  bool operator!=(const Tensor<T>& other) const { return !((*this) == other); }

  std::string ToString() const {
    return absl::StrCat("shape: ", shape_.ToString(), ", values: [",
                        absl::StrJoin(values_, ", ", absl::StreamFormatter()),
                        "]");
  }

  // Modify this to have shape "replacement_shape".  The size (number of
  // elements) of the initial and final shape must be the same, or else this
  // function will CHECK fail.
  void ReshapeInPlace(const Shape& replacement_shape) {
    CHECK_EQ(shape_.size(), replacement_shape.size());
    shape_ = replacement_shape;
  }

  // Create a new tensor with the same data as the current tensor, but shape
  // "replacement_shape".  The size (number of elements) of the new tensor
  // and this must be the same, or will CHECK fail.
  Tensor<T> Reshape(const Shape& replacement_shape) const {
    Tensor<T> result(*this);
    result.ReshapeInPlace(replacement_shape);
    return result;
  }

  // Reshapes this to remove any dimensions with dimension size 1.  E.g.
  //   input shape [1, 3, 1, 2] => output shape [3, 2]
  // Like numpy.squeeze(a, axis=None).
  //
  // The operation is done "in place", modifying this (see also Squeeze()).
  void SqueezeInPlace();

  // Creates and returns a copy of this with a new shape, where all size one
  // dimensions have been removed.  See SqueezeInPlace() for details.
  Tensor<T> Squeeze() const {
    Tensor<T> result = *this;
    result.SqueezeInPlace();
    return result;
  }

  // Checks if a Squeeze(absl::Span<const int> axes) operation is valid:
  //   1. axes is non-empty
  //   2. The elements a in axes satisfy:
  //       i. a in [0, this->dimension().NumDimensions())
  //      ii. this->dimension().dim_size(a) == 1
  absl::Status ValidateSqueeze(absl::Span<const int> axes) const;

  // Reshapes this to remove the dimensions of size one listed in axes.
  //
  // Similar to numpy.squeeze(a, axis=axes).
  //
  // Will CHECK fail if !ValidateSqueeze(axes).ok().
  //
  // E.g. if t.shape = [1, 3, 1, 2], then
  //   t.Squeeze([0]).shape = [3, 1, 2]
  //   t.Squeeze([2]).shape = [1, 3, 2]
  //   t.Squeeze([0, 2]).shape = [1, 3, 1, 2]
  //   t.Squeeze([]) => error
  //   t.Squeeze([1]) => error
  //   t.Squeeze([0, 1]) => error
  //
  // The operation is done "in place", modifying this (see also
  // Squeeze(absl::Span<const int> axes)).
  void SqueezeInPlace(absl::Span<const int> axes);

  // Creates and returns a copy of this with a new shape, where the size one
  // dimensions listed in axes have been removed.  See
  // SqueezeInPlace(absl::Span<const int> axes) for details.
  Tensor<T> Squeeze(absl::Span<const int> axes) const {
    Tensor<T> result = *this;
    result.SqueezeInPlace(axes);
    return result;
  }

  // Checks if an ExpandDims() operation is valid:
  //   * axis is in [0, this->dimension().NumDimensions()]
  absl::Status ValidateExpandDims(int axis) const;

  // Reshapes this add a dimension of size one at index axis.
  //
  // Similar to numpy.expand_dims(a, axis).
  //
  // Will CHECK fail if !ValidateExpandDims(axis).ok().
  //
  // E.g. if t.shape = [1, 3, 2], then
  //   t.ExpandDims(0).shape = [1, 1, 3, 2]
  //   t.ExpandDims(1).shape = [1, 1, 3, 2]
  //   t.ExpandDims(2).shape = [1, 3, 1, 2]
  //   t.ExpandDims(3).shape = [1, 3, 2, 1]
  //   t.ExpandDims(4) => error
  //
  // The operation is done "in place", modifying this (see also ExpandDims()).
  void ExpandDimsInPlace(int axis);

  // Creates and returns a copy of this with a new shape, where a dimension of
  // dimensions listed in axis have been removed.  See
  // SqueezeInPlace(absl::Span<const int> axis) for details.
  Tensor<T> ExpandDims(int axis) const {
    Tensor<T> result = *this;
    result.ExpandDimsInPlace(axis);
    return result;
  }

  // Checks if this->Slice(begin_indices, sizes) is a valid operation.
  //
  // Specifically, the slice is valid when:
  //  * begin_indices and sizes both have length equal to the rank of this,
  //  * 0 <= begin_indices elementwise,
  //  * 0 <= sizes elementwise,
  //  * begin_indices + sizes <= this->dimension() elementwise.
  absl::Status ValidateSlice(absl::Span<const int64_t> begin_indices,
                             absl::Span<const int64_t> sizes) const;

  // Creates a subtensor of this tensor.
  //
  // Similar to tf.slice(this, begin_indices, sizes).
  //
  // Creates the rectangular subtensor starting at begin_indices (inclusive) and
  // ending at begin_indices + sizes (elementwise addition, exclusive).  The
  // length of begin_indices and sizes must both be equal to the rank of this.
  //
  // Will CHECK fail if !ValidateSlice(axis).ok().
  //
  // E.g. if t =  [[10, 11, 12],
  //               [13, 14, 15],
  //               [16, 17, 18]] then
  //   t.Slice([0, 0], [1, 1]) = [[10]]
  //   t.Slice([0, 0], [1, 2]) = [[10, 11]]
  //   t.Slice([0, 0], [3, 1]) = [[10], [13], [16]]
  //   t.Slice([1, 2], [1, 1]) = [[17]]
  //   t.Slice([1, 1], [2, 2]) = [[14, 15], [17, 18]]
  //   t.Slice([1, 1], [2, 0]) = [[]]
  //   t.Slice([0, 0], [4, 4]) => error
  //   t.Slice([2, 2], [2, 2]) => error
  //   t.Slice([-1, 0], [1, 1]) => error
  Tensor<T> Slice(absl::Span<const int64_t> begin_indices,
                  absl::Span<const int64_t> sizes) const;

  // These methods all require that shape_.num_dimensions >= 1.
  //
  // Extracts the sub-tensor with elements having the first dimension in
  // [start_index, start_index + size). E.g.
  //
  // t = [[2, 3, 4], [5, 6, 7], [8, 9, 10]]
  // t.SubTensor(0, 2) => [[2, 3, 4], [5, 6, 7]]
  //
  // For variants with only an index, size=1, e.g.
  //
  // t.SubTensor(1, keep_dims=true) => [[5, 6, 7]]
  // t.SubTensor(1, keep_dims=false) => [5, 6, 7]
  //
  // Variants with an output argument will avoid allocating memory if
  // result.size() is already correct. result cannot be this.
  Tensor<T> SubTensor(int start_index, int size) const;
  Tensor<T> SubTensor(int index, bool keep_dims = false) const;
  void SubTensor(int start_index, int size, Tensor<T>* result) const;
  void SubTensor(int index, Tensor<T>* result, bool keep_dims = false) const;

 private:
  Shape shape_;
  std::vector<T> values_;
};

template <typename T>
std::ostream& operator<<(std::ostream& stream, const Tensor<T>& tensor) {
  stream << tensor.ToString();
  return stream;
}

using DoubleTensor = Tensor<double>;
using BoundsTensor = Tensor<Bounds>;

// Deprecated, prefer DoubleTensorProto version below.
DoubleTensor ProtoToDoubleTensor(
    const proto::ParameterValue& double_tensor_proto);

DoubleTensor ProtoToDoubleTensor(const DoubleTensorProto& double_tensor_proto);

// NOTE: This leaves double_tensor_proto.name unchanged.
// Deprecated, prefer DoubleTensorProto version below.
void DoubleTensorToProto(const DoubleTensor& double_tensor,
                         proto::ParameterValue* double_tensor_proto);

DoubleTensorProto DoubleTensorToProto(const DoubleTensor& double_tensor);

BoundsTensor DoubleTensorToBoundsTensor(const DoubleTensor& double_tensor);

bool HasInfiniteOrNan(const DoubleTensor& tensor);

// Returns a string representing a tensor of Bounds. Calls ToString on each
// bound before joining them.
template <>
inline std::string Tensor<Bounds>::ToString() const {
  return absl::StrCat("shape: ", shape_.ToString(), ", values: [",
                      absl::StrJoin(values_, ", ",
                                    [](std::string* out, const Bounds& bounds) {
                                      absl::StrAppend(out, bounds.ToString());
                                    }),
                      "]");
}

// Methods that can be applied to any Tensor type as free functions.

// Warning: makes a copy!
// TODO: rename GetTensorShape()
template <typename T>
Shape TensorDimension(const Tensor<T>& tensor) {
  return tensor.dimension();
}

template <typename T>
int64_t TensorSize(const Tensor<T>& tensor) {
  return tensor.size();
}

template <typename T>
void TensorReshapeInPlace(Tensor<T>* tensor, const Shape& replacement_shape) {
  tensor->ReshapeInPlace(replacement_shape);
}

namespace internal {

Shape SqueezeShape(const Shape& input_shape);
absl::StatusOr<Shape> SqueezeShape(const Shape& input_shape,
                                   absl::Span<const int> axes);
absl::StatusOr<Shape> ExpandDimsShape(const Shape& input_shape, int axis);
absl::StatusOr<Shape> SliceShape(const Shape& input_shape,
                                 absl::Span<const int64_t> begin_indices,
                                 absl::Span<const int64_t> sizes);

}  // namespace internal

// Template implementations

template <typename T>
void Tensor<T>::SqueezeInPlace() {
  ReshapeInPlace(internal::SqueezeShape(shape_));
}

template <typename T>
absl::Status Tensor<T>::ValidateSqueeze(absl::Span<const int> axes) const {
  return std::move(internal::SqueezeShape(shape_, axes)).status();
}

template <typename T>
void Tensor<T>::SqueezeInPlace(absl::Span<const int> axes) {
  const Shape result_shape = internal::SqueezeShape(shape_, axes).value();
  ReshapeInPlace(result_shape);
}

template <typename T>
absl::Status Tensor<T>::ValidateExpandDims(int axis) const {
  return std::move(internal::ExpandDimsShape(shape_, axis)).status();
}

template <typename T>
void Tensor<T>::ExpandDimsInPlace(int axis) {
  const Shape result_shape = internal::ExpandDimsShape(shape_, axis).value();
  ReshapeInPlace(result_shape);
}

template <typename T>
absl::Status Tensor<T>::ValidateSlice(absl::Span<const int64_t> begin_indices,
                                      absl::Span<const int64_t> sizes) const {
  return std::move(internal::SliceShape(shape_, begin_indices, sizes)).status();
}

template <typename T>
std::vector<T> Tensor<T>::VectorSlice(
    std::vector<int64_t> fixed_indices) const {
  CHECK_EQ(fixed_indices.size(), shape_.num_dimensions());
  int free_index = -1;
  for (int i = 0; i < fixed_indices.size(); ++i) {
    if (fixed_indices[i] < 0) {
      CHECK_LT(free_index, 0)
          << "Found two free indices: " << free_index << " and " << i << ".";
      free_index = i;
    } else {
      CHECK_LT(fixed_indices[i], shape_.dimension_size(i));
    }
  }
  CHECK_GE(free_index, 0);
  std::vector<T> slice;
  const int64_t size = shape_.dimension_size(free_index);
  slice.reserve(size);
  // Because fixed indices is passed by value, we can just reuse it as the
  // index vector accessing values from the tensor (we must modify it).
  for (int64_t i = 0; i < size; ++i) {
    fixed_indices[free_index] = i;
    slice.push_back(value(fixed_indices));
  }
  return slice;
}

template <typename T>
Tensor<T> Tensor<T>::Slice(absl::Span<const int64_t> begin_indices,
                           absl::Span<const int64_t> sizes) const {
  const Shape result_shape =
      internal::SliceShape(shape_, begin_indices, sizes).value();
  Tensor result(result_shape);
  // TODO: This could have a MUCH faster implementation
  for (int out_i = 0; out_i < result_shape.size(); ++out_i) {
    // multi_i has the multi index in the output space of the next element.
    std::vector<int64_t> multi_i = result_shape.ExpandIndex(out_i);
    for (int j = 0; j < multi_i.size(); ++j) {
      multi_i[j] += begin_indices[j];
    }
    // Now multi_i has the multi index in the input space of the next element.
    (*result.mutable_flat_values())[out_i] = value(multi_i);
  }
  return result;
}

namespace internal {
Shape SubTensorShape(const Shape& input_shape, int start, int size);
}  // namespace internal

template <typename T>
void Tensor<T>::SubTensor(int start_index, int size, Tensor<T>* result) const {
  CHECK(result != nullptr);
  CHECK(result != this);
  const Shape output_shape =
      internal::SubTensorShape(shape_, start_index, size);
  if (result->size() == output_shape.size()) {
    result->ReshapeInPlace(output_shape);
  } else {
    *result = Tensor<T>(output_shape);
  }
  std::vector<int64_t> multi_index(shape_.num_dimensions());
  multi_index[0] = start_index;
  const int64_t first_element_flat_index = shape_.FlattenIndexSpan(multi_index);
  const int64_t flat_size = first_element_flat_index + result->size();
  std::copy(values_.begin() + first_element_flat_index,
            values_.begin() + flat_size,
            result->mutable_flat_values()->begin());
}

template <typename T>
void Tensor<T>::SubTensor(int index, Tensor<T>* result, bool keep_dims) const {
  SubTensor(index, 1, result);
  if (!keep_dims) {
    result->SqueezeInPlace({0});
  }
}

template <typename T>
Tensor<T> Tensor<T>::SubTensor(int start_index, int size) const {
  Tensor<T> result;
  SubTensor(start_index, size, &result);
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::SubTensor(int index, bool keep_dims) const {
  Tensor<T> result;
  SubTensor(index, &result, keep_dims);
  return result;
}

}  // namespace tf_opt

#endif  // TF_OPT_TENSOR_TENSOR_H_
