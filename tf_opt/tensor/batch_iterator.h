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

#ifndef TF_OPT_TENSOR_BATCH_ITERATOR_H_
#define TF_OPT_TENSOR_BATCH_ITERATOR_H_

#include <cstdint>
#include <string>

#include "ortools/base/logging.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/die_if_null.h"
#include "absl/status/statusor.h"
#include "tf_opt/tensor/tensor.h"

namespace tf_opt {

// An iterator for parallel tensors that iterates though subtensors of a given
// batch size. The input tensors must all have >= 1 dimension and must agree
// on the first dimension.
//
// Example:
//   features = {{"x", DoubleTensor({1.0, 2.0, 3.0, 4.0, 5.0})},
//               {"y", DoubleTensor({1.1, 2.1, 3.1, 4.1, 5.1})}}
//   BatchIterator iterator(&features, 2);
//   iterator.Advance() => true
//   iterator.current_batch() => {{"x", DoubleTensor({1.0, 2.0})},
//                                {"y", DoubleTensor({1.1, 2.1})}}
//   iterator.Advance() => true
//   iterator.current_batch() => {{"x", DoubleTensor({3.0, 4.0})},
//                                {"y", DoubleTensor({3.1, 4.1})}}
//   iterator.Advance() => true
//   iterator.current_batch() => {{"x", DoubleTensor({5.0})},
//                                {"y", DoubleTensor({5.1})}}
//   iterator.Advance() => false
//
// Implementation note: mallocs O(1) memory per pass through a dataset of size
// n. Copies O(n*sum of tensor sizes) memory, with one call to std::copy per
// feature tensor per batch.
template <typename T>
class BatchIterator {
 public:
  // Arguments:
  //   features: the data to be iterated over. NOT OWNED. The string key is an
  //       identifier used subsequently when extracting data in current_batch().
  //   batch_size: the subtensors extracted with have shape with batch_size as
  //       the first dimension (or less in the final iteration).
  BatchIterator(const absl::flat_hash_map<std::string, Tensor<T>>* features,
                int64_t batch_size);

  // Returns false when there is no data left.
  bool Advance();

  // Goes back to the initial state.
  void Reset() { position_ = -1; }

  // Call only after calling Advance() when the returned value is true. Note:
  // after construction or Reset(), Advance() must be called before
  // current_batch().
  const absl::flat_hash_map<std::string, Tensor<T>>& current_batch() const {
    return current_batch_;
  }
  int64_t current_batch_size() const {
    return current_batch().begin()->second.dimension().dimension_size(0);
  }
  int64_t dataset_size() const { return dataset_size_; }

  // Tests if the input is valid for BatchIterator (all tensors in features must
  // have at least one dimension and the first dimension must match), and
  // returns the matching first dimension on a success.
  static absl::StatusOr<int64_t> CanBatchAndDatasetSize(
      const absl::flat_hash_map<std::string, Tensor<T>>& features);

 private:
  // NOT OWNED.
  const absl::flat_hash_map<std::string, Tensor<T>>* features_;
  int64_t position_;
  int64_t batch_size_;
  int64_t dataset_size_;
  absl::flat_hash_map<std::string, Tensor<T>> current_batch_;
};

// ////////////////////////// Template Implementations /////////////////////////

template <typename T>
absl::StatusOr<int64_t> BatchIterator<T>::CanBatchAndDatasetSize(
    const absl::flat_hash_map<std::string, Tensor<T>>& features) {
  if (features.empty()) {
    return 0;
  }
  const auto first_feature = *features.begin();
  if (first_feature.second.dimension().num_dimensions() == 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Feature ", first_feature.first,
        " was a scalar (had the empty shape), but all features should have at "
        "least one dimension."));
  }
  const int64_t dataset_size =
      first_feature.second.dimension().dimension_size(0);
  const std::string first_feature_name = features.begin()->first;

  for (const auto& name_feature_pair : features) {
    const Shape& feature_shape = name_feature_pair.second.dimension();
    if (feature_shape.num_dimensions() == 0 ||
        feature_shape.dimension_size(0) != dataset_size) {
      return absl::InvalidArgumentError(absl::StrCat(
          "On feature: ", name_feature_pair.first,
          "Expected first dimension of: ", dataset_size, " (to match ",
          first_feature_name, ") but had shape: ", feature_shape.ToString()));
    }
  }
  return dataset_size;
}

template <typename T>
BatchIterator<T>::BatchIterator(
    const absl::flat_hash_map<std::string, Tensor<T>>* features,
    const int64_t batch_size)
    : features_(CHECK_NOTNULL(features)),
      position_(-1),
      batch_size_(batch_size),
      dataset_size_(
          BatchIterator<T>::CanBatchAndDatasetSize(*features).value()) {}

template <typename T>
bool BatchIterator<T>::Advance() {
  if (position_ >= dataset_size_) {
    return false;
  }
  if (position_ < 0) {
    position_ = 0;
  } else {
    position_ += batch_size_;
  }
  if (position_ >= dataset_size_) {
    return false;
  }
  const int current_batch_size =
      std::min(batch_size_, dataset_size_ - position_);
  for (const auto& data_feature_pair : *features_) {
    data_feature_pair.second.SubTensor(
        position_, current_batch_size,
        &current_batch_[data_feature_pair.first]);
  }
  return true;
}

}  // namespace tf_opt

#endif  // TF_OPT_TENSOR_BATCH_ITERATOR_H_
