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

#include "tf_opt/tensor/tensor_testing.h"

#include <numeric>

namespace tf_opt {
namespace {

class DoubleTensorIIDRandomNormalMatcher
    : public ::testing::MatcherInterface<DoubleTensor> {
 public:
  DoubleTensorIIDRandomNormalMatcher(const Shape& shape, const double mean,
                                     const double stddev)
      : shape_(shape), mean_(mean), stddev_(stddev) {}

  bool MatchAndExplain(
      DoubleTensor lhs,
      ::testing::MatchResultListener* listener) const override {
    // check the maximum value, minimum value, and sum of the values.
    if (lhs.dimension() != shape_) {
      *listener << "Expected shape: " << shape_
                << ", but found shape: " << lhs.dimension();
      return false;
    }
    if (lhs.size() == 0) {
      return true;
    }
    const std::vector<double>& flat = lhs.flat_values();
    const double max_obs = *std::max_element(flat.begin(), flat.end());
    // The maximum should be ~ mean_ + stddev_ * sqrt(2 ln(n))
    // and the stddev of the maximum is less than the stddev of a single draw.
    // https://math.stackexchange.com/questions/89030/expectation-of-the-maximum-of-gaussian-random-variables
    // So w.h.p, the maximum will lie in
    //   target - C * stddev  <= max_obs <= target + C * stddev
    // where target = mean_ + stddev_ * sqrt(2 ln(n))
    const double max_target =
        mean_ + stddev_ * std::sqrt(2 * std::log(lhs.size()));
    // C above. The stddev of the maximum is less than the stddev of a single
    // draw (it concentrates as n gets large). The values 4 and 2 below were
    // chosen heuristically. TODO: get a sharper bound on the stddev of
    // the max.
    const double num_stddevs_width = lhs.size() < 10 ? 4 : 2;
    if (!PointInRange(max_obs, max_target, num_stddevs_width * stddev_, "max",
                      listener)) {
      return false;
    }

    // A symmetric argument to the above.
    const double min_obs = *std::min_element(flat.begin(), flat.end());
    const double min_target =
        mean_ - stddev_ * std::sqrt(2 * std::log(lhs.size()));
    if (!PointInRange(min_obs, min_target, num_stddevs_width * stddev_, "min",
                      listener)) {
      return false;
    }

    // The sum of the observations will be normal with:
    //   sum_mean = n * mean_
    //   sum_var = n * stddev_^2
    //   sum_stddev = sqrt(sum_var)
    // So w.h.p.,
    //   sum_mean - 4 * sum_stddev <= sum_obs <= sum + 4 * sum_stddev
    const double sum_obs = std::reduce(flat.begin(), flat.end());
    const double sum_target = lhs.size() * mean_;
    const double sum_stddev = std::sqrt(lhs.size() * stddev_ * stddev_);
    if (!PointInRange(sum_obs, sum_target, 4 * sum_stddev, "sum", listener)) {
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os) const override {
    *os << "tensor of shape: " << shape_.ToString()
        << " is approximately iid normal with mean: " << mean_
        << " and stddev: " << stddev_;
  }

  void DescribeNegationTo(std::ostream* os) const override {
    *os << "tensor of shape: " << shape_.ToString()
        << " is NOT approximately iid normal with mean: " << mean_
        << " and stddev: " << stddev_;
  }

 private:
  bool PointInRange(double point, double center, double half_width,
                    const std::string& name,
                    ::testing::MatchResultListener* listener) const {
    if (point > center + half_width) {
      *listener << "Expected " << name
                << " to be at most: " << center + half_width
                << ", but found: " << point;
      return false;
    }
    if (point < center - half_width) {
      *listener << "Expected " << name
                << " to be at least: " << center - half_width
                << ", but found: " << point;
      return false;
    }
    return true;
  }

  const Shape shape_;
  const double mean_;
  const double stddev_;
};

}  // namespace

::testing::Matcher<DoubleTensor> IsIIDRandomNormal(const Shape& shape,
                                                   const double mean,
                                                   const double stddev) {
  return ::testing::MakeMatcher(
      new DoubleTensorIIDRandomNormalMatcher(shape, mean, stddev));
}

}  // namespace tf_opt
