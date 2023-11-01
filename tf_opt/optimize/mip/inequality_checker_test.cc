// Copyright 2024 The tf.opt Authors.
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

#include "tf_opt/optimize/mip/inequality_checker.h"

#include "gtest/gtest.h"
#include "ortools/linear_solver/linear_expr.h"
#include "ortools/linear_solver/linear_solver.h"

namespace tf_opt {
namespace {

using operations_research::LinearExpr;
using operations_research::MPSolver;

// Creates a feasibility model of the form:
//   x1 + x2 <= 2.5
//   0 <= x1 <= 3, 0 <= x2 <= 5
//   x1, x2 integer
//
// With coefficients (1,1), the following inequalities are tight:
//   x1 + x2 <= 2, x1 + x2 >= 0
class InequalityCheckerTest : public ::testing::Test {
 public:
  InequalityCheckerTest()
      : solver_("inequality_checker_test",
                MPSolver::SCIP_MIXED_INTEGER_PROGRAMMING),
        x1_(solver_.MakeIntVar(0, 3, "x1")),
        x2_(solver_.MakeIntVar(0, 5, "x2")) {
    solver_.MakeRowConstraint(x1_ + x2_ <= 2.5);
  }

 protected:
  MPSolver solver_;
  const LinearExpr x1_;
  const LinearExpr x2_;
};

TEST_F(InequalityCheckerTest, ComputeGapValid) {
  EXPECT_EQ(ComputeInequalityGap(solver_, x1_ + x2_ <= 3.0), 1.0);
  EXPECT_EQ(ComputeInequalityGap(solver_, x1_ + x2_ >= -1.0), 1.0);
}

TEST_F(InequalityCheckerTest, ComputeGapNegative) {
  EXPECT_EQ(ComputeInequalityGap(solver_, x1_ + x2_ <= 1.0), -1.0);
  EXPECT_EQ(ComputeInequalityGap(solver_, x1_ + x2_ >= 1.0), -1.0);
}

TEST_F(InequalityCheckerTest, CheckValidInequality) {
  EXPECT_TRUE(CheckValidInequality(solver_, x1_ + x2_ <= 3.0));
  EXPECT_FALSE(CheckValidInequality(solver_, x1_ + x2_ >= 1.0));
}

}  // namespace
}  // namespace tf_opt
