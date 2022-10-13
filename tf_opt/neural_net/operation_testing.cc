// Copyright 2022 The tf.opt Authors.
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

#include "tf_opt/neural_net/operation_testing.h"

#include "ortools/base/container_logging.h"

namespace tf_opt {
namespace {

using ::testing::MakeMatcher;
using ::testing::Matcher;
using ::testing::MatcherInterface;
using ::testing::MatchResultListener;

class OperationMatcher : public MatcherInterface<const Operation&> {
 public:
  OperationMatcher(std::string name, std::vector<Shape> inputs, Shape output)
      : name_(std::move(name)),
        input_shapes_(std::move(inputs)),
        output_shape_(std::move(output)) {}

  bool MatchAndExplain(const Operation& actual,
                       MatchResultListener* listener) const override {
    if (actual.name() != name_) {
      *listener << "expected name: " << name_
                << ", but found: " << actual.name();
      return false;
    }
    if (actual.input_shapes() != input_shapes_) {
      *listener << "expected input shapes: " << gtl::LogContainer(input_shapes_)
                << ", but found: " << gtl::LogContainer(actual.input_shapes());
      return false;
    }
    if (actual.output_shape() != output_shape_) {
      *listener << "expected output shape: " << output_shape_
                << ", but found: " << actual.output_shape();
      return false;
    }

    return true;
  }

  void DescribeTo(::std::ostream* os) const override {
    *os << "operation has name: " << name_
        << ", input shapes: " << gtl::LogContainer(input_shapes_)
        << ", and output shape: " << output_shape_ << ".";
  }

 private:
  std::string name_;
  std::vector<Shape> input_shapes_;
  Shape output_shape_;
};

}  // namespace

testing::Matcher<const Operation&> OperationArgsAre(
    absl::string_view name, absl::Span<const Shape> input_shapes,
    const Shape& output_shape) {
  return MakeMatcher(new OperationMatcher(
      std::string(name),
      std::vector<Shape>(input_shapes.begin(), input_shapes.end()),
      output_shape));
}
}  // namespace tf_opt
