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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tf_opt/bounds/bounds.h"

namespace tf_opt {

namespace {

using ::testing::Not;

TEST(BoundsNear, Bounds) {
  const Bounds t1(Bounds(3.5, 4.0));
  const Bounds t2(Bounds(3.52, 3.95));
  EXPECT_THAT(t1, BoundsNear(t2, 0.1));
  EXPECT_THAT(t1, Not(BoundsNear(t2, 0.01)));
}

TEST(BoundsEqual, Bounds) {
  const Bounds t1(Bounds(3.5, 4.0));
  const Bounds t2(Bounds(3.5, 4.0));
  const Bounds t3(Bounds(3.52, 4.0));
  EXPECT_THAT(t1, BoundsEquals(t2));
  EXPECT_THAT(t1, Not(BoundsEquals(t3)));
}

}  // namespace

}  // namespace tf_opt
