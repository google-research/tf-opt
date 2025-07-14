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

// Example use:
//
// DoubleTensor t1({1.0, 2.0});
// DoubleTensor t2({1.0, 2.0});
// DoubleTensor t2({1.001, 2.001});
// EXPECT_THAT(t1, DoubleTensorEqual(t2));
// EXPECT_THAT(t1, DoubleTensorNear(t3), .01);
#ifndef TF_OPT_TENSOR_TENSOR_TESTING_H_
#define TF_OPT_TENSOR_TENSOR_TESTING_H_

#include <cstdint>

#include "gmock/gmock.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tf_opt/bounds/bounds_testing.h"
#include "tf_opt/tensor/tensor.h"

namespace tf_opt {

template <typename T>
bool NumericAreNear(const T& left, const T& right, double tolerance,
                    std::string* difference_out) {
  CHECK_NE(difference_out, nullptr);
  const double diff = std::abs(left - right);
  if (diff > tolerance) {
    *difference_out = absl::StrCat("Expected left: ", left, " and ", right,
                                   " to be within tolerance ", tolerance,
                                   ", but difference was ", diff);
    return false;
  }
  return true;
}

template <typename T>
class TensorMatcher : public ::testing::MatcherInterface<Tensor<T>> {
 public:
  TensorMatcher(const Tensor<T>& rhs,
                std::function<bool(const T&, const T&, double, std::string*)>
                    is_near_function,
                double tolerance)
      : rhs_(rhs), is_near_function_(is_near_function), tolerance_(tolerance) {}

  bool MatchAndExplain(Tensor<T> lhs,
                       ::testing::MatchResultListener* listener) const override;

  void DescribeTo(std::ostream* os) const override;

  void DescribeNegationTo(std::ostream* os) const override;

 private:
  const Tensor<T> rhs_;
  std::function<bool(T, T, double, std::string*)> is_near_function_;
  const double tolerance_;
};

template <typename T>
::testing::Matcher<Tensor<T>> TensorNear(const Tensor<T>& rhs,
                                         double tolerance) {
  return MakeMatcher(new TensorMatcher<T>(rhs, NumericAreNear<T>, tolerance));
}

template <typename T>
::testing::Matcher<Tensor<T>> TensorEquals(const Tensor<T>& rhs) {
  return TensorNear(rhs, 0.0);
}

inline ::testing::Matcher<DoubleTensor> DoubleTensorNear(
    const DoubleTensor& rhs, double tolerance = 1e-5) {
  return TensorNear(rhs, tolerance);
}

inline ::testing::Matcher<DoubleTensor> DoubleTensorEquals(
    const DoubleTensor& rhs) {
  return TensorNear(rhs, 0.0);
}

inline ::testing::Matcher<BoundsTensor> BoundsTensorNear(
    const BoundsTensor& rhs, double tolerance = 1e-5) {
  return MakeMatcher(new TensorMatcher<Bounds>(rhs, BoundsAreNear, tolerance));
}

inline ::testing::Matcher<BoundsTensor> BoundsTensorEquals(
    const BoundsTensor& rhs) {
  return BoundsTensorNear(rhs, 0);
}

template <typename T>
bool TensorMatcher<T>::MatchAndExplain(
    Tensor<T> lhs, ::testing::MatchResultListener* listener) const {
  if (lhs.dimension() != rhs_.dimension()) {
    *listener << "Tensors should have same shapes, but on left found "
              << lhs.dimension() << " and on right found " << rhs_.dimension();
    return false;
  }
  bool matches = true;
  for (int64_t i = 0; i < lhs.flat_values().size(); i++) {
    const T& left_value = lhs.flat_values()[i];
    const T& right_value = rhs_.flat_values()[i];
    std::string error = "";
    if (!is_near_function_(left_value, right_value, tolerance_, &error)) {
      matches = false;
      const std::string pos =
          absl::StrJoin(lhs.dimension().ExpandIndex(i), ", ");
      *listener << "At [" << pos << "]: " << error;
    }
  }
  return matches;
}

template <typename T>
void TensorMatcher<T>::DescribeTo(std::ostream* os) const {
  *os << "tensor entries are all within " << tolerance_ << " of "
      << rhs_.ToString();
}

template <typename T>
void TensorMatcher<T>::DescribeNegationTo(std::ostream* os) const {
  *os << "a tensor entry differs by " << tolerance_ << " from "
      << rhs_.ToString();
}

// Tests that a DoubleTensor looks approximately iid Normal(mean, stddev). The
// current implementation checks the min, max, and sum of all entries are in
// a typical range, this may be improved in the future. On iid random input,
// the test will pass with ~P(-4 <= N(0,1) <= 4), e.g. 0.9999.
::testing::Matcher<DoubleTensor> IsIIDRandomNormal(const Shape& shape,
                                                   double mean, double stddev);

}  // namespace tf_opt

#endif  // TF_OPT_TENSOR_TENSOR_TESTING_H_
