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

#ifndef TF_OPT_OPTIMIZE_MIP_INEQUALITY_CHECKER_H_
#define TF_OPT_OPTIMIZE_MIP_INEQUALITY_CHECKER_H_

#include "ortools/linear_solver/linear_expr.h"
#include "ortools/linear_solver/linear_solver.h"

namespace tf_opt {

// Returns the gap between a (one-sided) inequality and the tightest inequality
// modeled by solver with the same coefficients. If negative, this implies that
// the inequality is not valid. This solves a (copy of the) full model and may
// be slow, and it is intended to be used mainly for analysis and debugging.
double ComputeInequalityGap(const operations_research::MPSolver& solver,
                            const operations_research::LinearRange inequality);

// Returns true if inequality is valid with respect to model in solver. No
// tolerance is used. This solves (a copy of) the full model in the process.
bool CheckValidInequality(const operations_research::MPSolver& solver,
                          const operations_research::LinearRange inequality);

}  // namespace tf_opt

#endif  // TF_OPT_OPTIMIZE_MIP_INEQUALITY_CHECKER_H_
