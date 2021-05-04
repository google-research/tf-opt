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

#include "tf_opt/bounds/bounds.h"

#include <limits>
#include <sstream>

#include "gtest/gtest.h"

namespace tf_opt {
namespace {

TEST(BoundsTest, Addition) {
  const Bounds bounds = Bounds(2, 6) + Bounds(-3, 4);
  EXPECT_DOUBLE_EQ(bounds.lb(), -1);
  EXPECT_DOUBLE_EQ(bounds.ub(), 10);
}

TEST(BoundsTest, AdditionInPlace) {
  Bounds bounds(2, 6);
  bounds += Bounds(-3, 4);
  EXPECT_DOUBLE_EQ(bounds.lb(), -1);
  EXPECT_DOUBLE_EQ(bounds.ub(), 10);
}

TEST(BoundsTest, AdditionInPlaceDouble) {
  Bounds bounds(2, 6);
  bounds += 2;
  EXPECT_DOUBLE_EQ(bounds.lb(), 4);
  EXPECT_DOUBLE_EQ(bounds.ub(), 8);
}

TEST(BoundsTest, AdditionLeftDouble) {
  const Bounds bounds = 2.0 + Bounds(-3, 4);
  EXPECT_DOUBLE_EQ(bounds.lb(), -1);
  EXPECT_DOUBLE_EQ(bounds.ub(), 6);
}

TEST(BoundsTest, AdditionRightDouble) {
  const Bounds bounds = Bounds(2, 6) + (-3.0);
  EXPECT_DOUBLE_EQ(bounds.lb(), -1);
  EXPECT_DOUBLE_EQ(bounds.ub(), 3);
}

TEST(BoundsTest, Subtraction) {
  const Bounds bounds = Bounds(2, 6) - Bounds(-3, 4);
  EXPECT_DOUBLE_EQ(bounds.lb(), -2);
  EXPECT_DOUBLE_EQ(bounds.ub(), 9);
}

TEST(BoundsTest, SubtractionInPlace) {
  Bounds bounds(2, 6);
  bounds -= Bounds(-3, 4);
  EXPECT_DOUBLE_EQ(bounds.lb(), -2);
  EXPECT_DOUBLE_EQ(bounds.ub(), 9);
}

TEST(BoundsTest, SubtractionInPlaceDouble) {
  Bounds bounds(2, 6);
  bounds -= 2;
  EXPECT_DOUBLE_EQ(bounds.lb(), 0);
  EXPECT_DOUBLE_EQ(bounds.ub(), 4);
}

TEST(BoundsTest, SubtractionLeftDouble) {
  const Bounds bounds = 2.0 - Bounds(-3, 4);
  EXPECT_DOUBLE_EQ(bounds.lb(), -2);
  EXPECT_DOUBLE_EQ(bounds.ub(), 5);
}

TEST(BoundsTest, SubtractionRightDouble) {
  const Bounds bounds = Bounds(2, 6) - (-3.0);
  EXPECT_DOUBLE_EQ(bounds.lb(), 5);
  EXPECT_DOUBLE_EQ(bounds.ub(), 9);
}

TEST(BoundsTest, Multiplication) {
  Bounds bounds = Bounds(2, 6) * Bounds(-3, 4);
  EXPECT_DOUBLE_EQ(bounds.lb(), -18);
  EXPECT_DOUBLE_EQ(bounds.ub(), 24);

  bounds = Bounds(2, 6) * Bounds(-3, -1);
  EXPECT_DOUBLE_EQ(bounds.lb(), -18);
  EXPECT_DOUBLE_EQ(bounds.ub(), -2);

  bounds = Bounds(-2, 6) * Bounds(3, 6);
  EXPECT_DOUBLE_EQ(bounds.lb(), -12);
  EXPECT_DOUBLE_EQ(bounds.ub(), 36);
}

TEST(BoundsTest, MultiplicationInPlace) {
  Bounds bounds(2, 6);
  bounds *= Bounds(-3, 4);
  EXPECT_DOUBLE_EQ(bounds.lb(), -18);
  EXPECT_DOUBLE_EQ(bounds.ub(), 24);
}

TEST(BoundsTest, MultiplicationInPlaceDouble) {
  Bounds bounds(2, 6);
  bounds *= 2;
  EXPECT_DOUBLE_EQ(bounds.lb(), 4);
  EXPECT_DOUBLE_EQ(bounds.ub(), 12);
}

TEST(BoundsTest, MultiplicationLeftDouble) {
  const Bounds bounds = 2.0 * Bounds(-3, 4);
  EXPECT_DOUBLE_EQ(bounds.lb(), -6);
  EXPECT_DOUBLE_EQ(bounds.ub(), 8);
}

TEST(BoundsTest, MultiplicationRightDouble) {
  const Bounds bounds = Bounds(2, 6) * (-3.0);
  EXPECT_DOUBLE_EQ(bounds.lb(), -18);
  EXPECT_DOUBLE_EQ(bounds.ub(), -6);
}

TEST(BoundsTest, Division) {
  const Bounds bounds = Bounds(3, 8) / Bounds(2, 6);
  EXPECT_DOUBLE_EQ(bounds.lb(), 0.5);
  EXPECT_DOUBLE_EQ(bounds.ub(), 4);
}

TEST(BoundsTest, DivisionZeroInNumerator) {
  const Bounds bounds = Bounds(0, 0) / Bounds(1, 2);
  EXPECT_DOUBLE_EQ(bounds.lb(), 0);
  EXPECT_DOUBLE_EQ(bounds.ub(), 0);
}

TEST(BoundsTest, DivisionByZero) {
  const Bounds bounds = Bounds(1, 2) / Bounds(0, 0);
  EXPECT_DOUBLE_EQ(bounds.lb(), -std::numeric_limits<double>::infinity());
  EXPECT_DOUBLE_EQ(bounds.ub(), std::numeric_limits<double>::infinity());
}

TEST(BoundsTest, DivisionByIntervalContainingZero) {
  const Bounds bounds = Bounds(1, 2) / Bounds(-1, 1);
  EXPECT_DOUBLE_EQ(bounds.lb(), -std::numeric_limits<double>::infinity());
  EXPECT_DOUBLE_EQ(bounds.ub(), std::numeric_limits<double>::infinity());
}

TEST(BoundsTest, DivisionInPlace) {
  Bounds bounds(3, 8);
  bounds /= Bounds(2, 6);
  EXPECT_DOUBLE_EQ(bounds.lb(), 0.5);
  EXPECT_DOUBLE_EQ(bounds.ub(), 4);
}

TEST(BoundsTest, DivisionInPlaceDouble) {
  Bounds bounds(2, 8);
  bounds /= 2;
  EXPECT_DOUBLE_EQ(bounds.lb(), 1);
  EXPECT_DOUBLE_EQ(bounds.ub(), 4);
}

TEST(BoundsTest, DivisionLeftDouble) {
  const Bounds bounds = 3.0 / Bounds(2, 6);
  EXPECT_DOUBLE_EQ(bounds.lb(), 0.5);
  EXPECT_DOUBLE_EQ(bounds.ub(), 1.5);
}

TEST(BoundsTest, DivisionRightDouble) {
  const Bounds bounds = Bounds(3, 8) / Bounds(2, 6);
  EXPECT_DOUBLE_EQ(bounds.lb(), 0.5);
  EXPECT_DOUBLE_EQ(bounds.ub(), 4);
}

TEST(BoundsTest, Negate) {
  const Bounds bounds = -Bounds(-2, 6);
  EXPECT_DOUBLE_EQ(bounds.lb(), -6);
  EXPECT_DOUBLE_EQ(bounds.ub(), 2);
}

TEST(BoundsTest, MaxTwo) {
  const Bounds max_bounds = Max(Bounds(-2, 6), Bounds(-3, 7));
  EXPECT_DOUBLE_EQ(max_bounds.lb(), -2);
  EXPECT_DOUBLE_EQ(max_bounds.ub(), 7);
}

TEST(BoundsTest, MaxList) {
  const Bounds max_bounds = Max({Bounds(-2, 6), Bounds(-3, 7), Bounds(-1, 5)});
  EXPECT_DOUBLE_EQ(max_bounds.lb(), -1);
  EXPECT_DOUBLE_EQ(max_bounds.ub(), 7);
}

TEST(BoundsTest, MinTwo) {
  const Bounds min_bounds = Min(Bounds(-2, 6), Bounds(-3, 7));
  EXPECT_DOUBLE_EQ(min_bounds.lb(), -3);
  EXPECT_DOUBLE_EQ(min_bounds.ub(), 6);
}

TEST(BoundsTest, Intersect) {
  const Bounds max_bounds = Intersect(Bounds(-2, 6), Bounds(-3, 7));
  EXPECT_DOUBLE_EQ(max_bounds.lb(), -2);
  EXPECT_DOUBLE_EQ(max_bounds.ub(), 6);
}

TEST(BoundsTest, OutputStream) {
  const Bounds bounds(2, 6);
  std::ostringstream ostr;
  ostr << bounds;
  EXPECT_EQ(ostr.str(), "[2, 6]");
}

TEST(BoundsTest, SingleConstructor) {
  const Bounds bounds(2, 2);
  EXPECT_DOUBLE_EQ(bounds.lb(), 2);
  EXPECT_DOUBLE_EQ(bounds.ub(), 2);
}

TEST(BoundsTest, Equality) { EXPECT_TRUE(Bounds(2, 4) == Bounds(2, 4)); }

TEST(BoundsTest, UnboundedOperation) {
  const Bounds bounds = Bounds(2, +std::numeric_limits<double>::infinity()) +
                        Bounds(-3, +std::numeric_limits<double>::infinity());
  EXPECT_DOUBLE_EQ(bounds.lb(), -1);
  EXPECT_DOUBLE_EQ(bounds.ub(), +std::numeric_limits<double>::infinity());
}

TEST(BoundsTest, Unbounded) {
  const Bounds bounds = Bounds::Unbounded();
  EXPECT_DOUBLE_EQ(bounds.lb(), -std::numeric_limits<double>::infinity());
  EXPECT_DOUBLE_EQ(bounds.ub(), +std::numeric_limits<double>::infinity());
}

}  // namespace
}  // namespace tf_opt
