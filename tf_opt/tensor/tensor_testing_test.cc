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

#include "tf_opt/tensor/tensor_testing.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tf_opt/tensor/tensor.h"

namespace tf_opt {
namespace {

using ::testing::Not;

TEST(DoubleTensorNear, Scalars) {
  const DoubleTensor t1(3.5);
  const DoubleTensor t2(3.52);
  EXPECT_THAT(t1, DoubleTensorNear(t2, 0.1));
  EXPECT_THAT(t1, Not(DoubleTensorNear(t2, 0.01)));
}

TEST(DoubleTensorEqual, Scalars) {
  const DoubleTensor t1(3.5);
  const DoubleTensor t2(3.5);
  const DoubleTensor t3(3.52);
  EXPECT_THAT(t1, DoubleTensorEquals(t2));
  EXPECT_THAT(t1, Not(DoubleTensorEquals(t3)));
}

TEST(IsIIDRandomNormal, Scalars) {
  EXPECT_THAT(DoubleTensor(5.1), IsIIDRandomNormal(Shape(), 5.0, 2.0));
  EXPECT_THAT(DoubleTensor(50.0), Not(IsIIDRandomNormal(Shape(), 5.0, 2.0)));
  EXPECT_THAT(DoubleTensor(-50.0), Not(IsIIDRandomNormal(Shape(), 5.0, 2.0)));
}

TEST(IsIIDRandomNormal, BadSum) {
  EXPECT_THAT(DoubleTensor(Shape({100}), 1.0),
              Not(IsIIDRandomNormal(Shape({100}), 0.0, 1.0)));
}

TEST(IsIIDRandomNormal, BadMaxTooBig) {
  DoubleTensor bad_input(Shape({100}));
  bad_input.set_flat_value(3, 7.0);
  bad_input.set_flat_value(5, -3.0);
  EXPECT_THAT(bad_input, Not(IsIIDRandomNormal(Shape({100}), 0.0, 1.0)));
}

TEST(IsIIDRandomNormal, BadMaxTooSmall) {
  // Expected max of of 100 N(0,1) is ~= sqrt(2 * ln 100) ~= 3.03
  // So the test below asserts that the observed max=0 is not in [1.03, 5.03].
  DoubleTensor bad_input(Shape({100}));
  bad_input.set_flat_value(5, -3.0);
  EXPECT_THAT(bad_input, Not(IsIIDRandomNormal(Shape({100}), 0.0, 1.0)));
}

TEST(IsIIDRandomNormal, BadMinTooBig) {
  DoubleTensor bad_input(Shape({100}));
  bad_input.set_flat_value(5, 3.0);
  EXPECT_THAT(bad_input, Not(IsIIDRandomNormal(Shape({100}), 0.0, 1.0)));
}

TEST(IsIIDRandomNormal, BadMinTooSmall) {
  DoubleTensor bad_input(Shape({100}));
  bad_input.set_flat_value(5, 3);
  bad_input.set_flat_value(3, -7.0);
  EXPECT_THAT(bad_input, Not(IsIIDRandomNormal(Shape({100}), 0.0, 1.0)));
}

TEST(IsIIDRandomNormal, TypicalInput) {
  DoubleTensor input(Shape({100}));
  input.set_flat_value(5, 2);
  input.set_flat_value(3, -2);
  EXPECT_THAT(input, IsIIDRandomNormal(Shape({100}), 0.0, 1.0));
}

}  // namespace
}  // namespace tf_opt
