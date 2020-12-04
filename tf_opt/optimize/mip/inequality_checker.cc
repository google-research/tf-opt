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

#include "tf_opt/optimize/mip/inequality_checker.h"

#include "glog/logging.h"
#include "ortools/linear_solver/linear_expr.h"
#include "ortools/linear_solver/linear_solver.h"
#include "ortools/linear_solver/linear_solver.pb.h"

namespace tf_opt {

using operations_research::LinearExpr;
using operations_research::LinearRange;
using operations_research::MPSolver;

double ComputeInequalityGap(const MPSolver& solver,
                            const LinearRange inequality) {
  CHECK(inequality.lower_bound() != -MPSolver::infinity() ||
        inequality.upper_bound() != +MPSolver::infinity());
  CHECK(inequality.lower_bound() == -MPSolver::infinity() ||
        inequality.upper_bound() == +MPSolver::infinity())
      << "Inequality to be checked must be one-sided";
  const bool less_or_equal =
      (inequality.upper_bound() != +MPSolver::infinity());

  // Copy given model via proto.
  MPSolver solver_copy("inequality_gap_solver", solver.ProblemType());
  operations_research::MPModelProto model_proto;
  solver.ExportModelToProto(&model_proto);
  std::string error_msg;
  const auto status = solver_copy.LoadModelFromProto(model_proto, &error_msg);
  CHECK_EQ(status, operations_research::MPSOLVER_MODEL_IS_VALID) << error_msg;

  const LinearExpr inequality_expr = inequality.linear_expr();

  // Map variables into copied variables.
  LinearExpr inequality_copy(inequality.linear_expr().offset());
  for (const auto& term : inequality.linear_expr().terms()) {
    inequality_copy +=
        term.second * LinearExpr(solver_copy.variables()[term.first->index()]);
  }

  solver_copy.MutableObjective()->OptimizeLinearExpr(inequality_copy,
                                                     less_or_equal);
  const MPSolver::ResultStatus solver_status = solver_copy.Solve();

  CHECK_EQ(solver_status, MPSolver::ResultStatus::OPTIMAL);

  if (less_or_equal) {
    return inequality.upper_bound() - solver_copy.Objective().Value();
  } else {
    return solver_copy.Objective().Value() - inequality.lower_bound();
  }
}

bool CheckValidInequality(const operations_research::MPSolver& solver,
                          const operations_research::LinearRange inequality) {
  return ComputeInequalityGap(solver, inequality) >= 0.0;
}

}  // namespace tf_opt
