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

#ifndef TF_OPT_BOUNDS_BOUNDS_TESTING_H_
#define TF_OPT_BOUNDS_BOUNDS_TESTING_H_

#include "gtest/gtest.h"
#include "tf_opt/bounds/bounds.h"

namespace tf_opt {

bool BoundsAreNear(const Bounds& left, const Bounds& right, double tolerance,
                   std::string* difference_out);

::testing::Matcher<Bounds> BoundsNear(const Bounds& rhs,
                                      double tolerance = 1e-5);

::testing::Matcher<Bounds> BoundsEquals(const Bounds& rhs);

}  // namespace tf_opt

#endif  // TF_OPT_BOUNDS_BOUNDS_TESTING_H_
