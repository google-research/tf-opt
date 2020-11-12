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

// Defines functors to be used with math_impl.h.
//
// This file is generally an implementation detail and typical users should not
// need to look at it.
//
// The unary and binary operations in math_impl.h require as input a functor
// that takes in the elements to operate on and the output index as input and
// produce the elementwise output. Many of these functors are defined inline
// as lambdas, but a few are defined here because either:
//  (a) They are reused.
//  (b) They are nontrivial and warrant testing.
#ifndef TF_OPT_TENSOR_ELEMENT_OPERATIONS_H_
#define TF_OPT_TENSOR_ELEMENT_OPERATIONS_H_

#include <cstdint>

#include "absl/types/span.h"

namespace tf_opt {

// Takes the maximum between two elements.
// NOTE: This enables ElementwiseMaximum and TfOptMaxElement to be
// used with Bounds. Otherwise it behaves as std::max except that it returns by
// value. We cannot use std::max + comparator for Bounds because the maximum of
// two Bounds may be a new Bounds object.
template <class T>
T TfOptMax(const T& left, const T& right) {
  return Max(left, right);
}

// Takes the minimum between two elements. See NOTE on TfOptMax function.
template <class T>
T TfOptMin(const T& left, const T& right) {
  return Min(left, right);
}

template <>
inline double TfOptMax(const double& left, const double& right) {
  return std::max(left, right);
}

template <>
inline double TfOptMin(const double& left, const double& right) {
  return std::min(left, right);
}

// TODO: this should return std::numeric_limits<T>::lowest() for types
// where infinity does not exist.
template <class T>
T TfOptLowest() {
  return -std::numeric_limits<T>::infinity();
}
template <class T>
T TfOptHighest() {
  return std::numeric_limits<T>::infinity();
}

// Unary element operations.

template <typename T>
struct ReluElement {
 public:
  T operator()(const T& input, const int64_t output_index) const {
    return TfOptMax(T(0.0), input);
  }
};

template <typename T>
struct ClippedReluElement {
 public:
  explicit ClippedReluElement(const double cap) : cap(cap) {}

  T operator()(const T& input, const int64_t output_index) const {
    return TfOptMin(T(cap), TfOptMax(T(0.0), input));
  }

  double cap;
};

// Binary element operations.

template <typename T>
struct MaxElements {
 public:
  T operator()(const T& left, const T& right,
               const int64_t output_index) const {
    return TfOptMax(left, right);
  }
};

template <typename T>
struct MinElements {
 public:
  T operator()(const T& left, const T& right,
               const int64_t output_index) const {
    return TfOptMin(left, right);
  }
};

// Bulk element operations.

template <typename T>
struct AddAllElements {
 public:
  T operator()(absl::Span<const T> elements, const int64_t output_index) const {
    T result(0.0);
    for (const T& t : elements) {
      result += t;
    }
    return result;
  }
};

template <typename T>
struct AverageAllElements {
 public:
  T operator()(absl::Span<const T> elements, const int64_t output_index) const {
    if (elements.empty()) return T(0.0);
    return AddAllElements<T>()(elements, output_index) / elements.size();
  }
};

template <typename T>
struct MaxAllElements {
 public:
  T operator()(absl::Span<const T> elements, const int64_t output_index) const {
    T result = TfOptLowest<T>();
    for (const T& t : elements) {
      result = TfOptMax(result, t);
    }
    return result;
  }
};

template <typename T>
struct MinAllElements {
 public:
  T operator()(absl::Span<const T> elements, const int64_t output_index) const {
    T result = TfOptHighest<T>();
    for (const T& t : elements) {
      result = TfOptMin(result, t);
    }
    return result;
  }
};

}  // namespace tf_opt

#endif  // TF_OPT_TENSOR_ELEMENT_OPERATIONS_H_
