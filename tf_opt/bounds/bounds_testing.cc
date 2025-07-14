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

#include "tf_opt/bounds/bounds_testing.h"

#include "ortools/base/logging.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "tf_opt/bounds/bounds.h"

namespace tf_opt {

namespace {

class BoundsMatcher : public ::testing::MatcherInterface<Bounds> {
 public:
  BoundsMatcher(const Bounds& rhs, double tolerance)
      : rhs_(rhs), tolerance_(tolerance) {}

  bool MatchAndExplain(
      Bounds lhs, ::testing::MatchResultListener* listener) const override {
    std::string diff;
    bool result = BoundsAreNear(lhs, rhs_, tolerance_, &diff);
    *listener << diff;
    return result;
  }

  void DescribeTo(std::ostream* os) const override {
    *os << "bounds are within " << tolerance_ << " of " << rhs_.ToString();
  }

  void DescribeNegationTo(std::ostream* os) const override {
    *os << "a bound differs by " << tolerance_ << " from " << rhs_.ToString();
  }

 private:
  const Bounds rhs_;
  const double tolerance_;
};

}  // namespace

bool BoundsAreNear(const Bounds& left, const Bounds& right, double tolerance,
                   std::string* difference_out) {
  CHECK_NE(difference_out, nullptr);
  const double diff_lb = std::abs(left.lb() - right.lb());
  const double diff_ub = std::abs(left.ub() - right.ub());
  if (diff_lb > tolerance) {
    *difference_out = absl::StrCat(
        "Expected left expression ", left.ToString(), " and right expression ",
        right.ToString(), " to be within tolerance ", tolerance,
        " but found difference of ", diff_lb, " at the lower bound");
    return false;
  }
  if (diff_ub > tolerance) {
    *difference_out = absl::StrCat(
        "Expected left expression ", left.ToString(), " and right expression ",
        right.ToString(), " to be within tolerance ", tolerance,
        " but found difference of ", diff_ub, " at the upper bound");
    return false;
  }
  return true;
}

testing::Matcher<Bounds> BoundsNear(const Bounds& rhs, double tolerance) {
  return MakeMatcher(new BoundsMatcher(rhs, tolerance));
}

testing::Matcher<Bounds> BoundsEquals(const Bounds& rhs) {
  return BoundsNear(rhs, 0);
}

}  // namespace tf_opt
