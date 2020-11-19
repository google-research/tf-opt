// Copyright 2020 The tf.opt Authors.
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

#include "tf_opt/neural_net/neuron/relu_impl_type.h"

#include <sstream>
#include <string>

#include "gtest/gtest.h"

namespace tf_opt {
namespace {

TEST(ReluImplTypeNameTest, BadName) {
  ReluImplementationType result;
  EXPECT_FALSE(ReluImplFromString("bad_name", &result));
}
TEST(ReluImplTypeNameDeathTest, BadName) {
  ASSERT_DEATH(ReluImplFromStringOrDie("bad_name"), "bad_name");
}

class ReluImplTypeTest
    : public ::testing::TestWithParam<ReluImplementationType> {};

TEST_P(ReluImplTypeTest, TestStringMethodsRoundTrip) {
  const std::string name(ToString(GetParam()));
  ReluImplementationType result;
  ASSERT_TRUE(ReluImplFromString(name, &result));
  EXPECT_EQ(result, GetParam());
  EXPECT_EQ(ReluImplFromStringOrDie(name), GetParam());

  // Check that the << operator agrees with name.
  std::ostringstream stream;
  stream << GetParam();
  EXPECT_EQ(name, stream.str());
}

INSTANTIATE_TEST_SUITE_P(
    ReluImplTypeTests, ReluImplTypeTest,
    ::testing::Values(ReluImplementationType::kBigM,
                      ReluImplementationType::kBigMRelaxation,
                      ReluImplementationType::kMultipleChoice,
                      ReluImplementationType::kIdealExponential,
                      ReluImplementationType::kMultipleChoiceSimplified));

}  // namespace
}  // namespace tf_opt
