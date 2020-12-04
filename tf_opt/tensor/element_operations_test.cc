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

#include "tf_opt/tensor/element_operations.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tf_opt {
namespace {

TEST(ElementOperationsTest, ReluElementTest) {
  ReluElement<double> relu;
  EXPECT_EQ(relu(-3.0, 0), 0.0);
  EXPECT_EQ(relu(3.0, 0), 3.0);
}

TEST(ElementOperationsTest, ClippedReluElementTest) {
  ClippedReluElement<double> clipped_relu(4.0);
  EXPECT_EQ(clipped_relu(-3.0, 0), 0.0);
  EXPECT_EQ(clipped_relu(3.0, 0), 3.0);
  EXPECT_EQ(clipped_relu(5.0, 0), 4.0);
}

TEST(ElementOperationsTest, MaxElementsTest) {
  MaxElements<double> m;
  EXPECT_EQ(m(-3.0, 7.0, 0), 7.0);
}

TEST(ElementOperationsTest, MinElementsTest) {
  MinElements<double> m;
  EXPECT_EQ(m(-3.0, 7.0, 0), -3.0);
}

TEST(ElementOperationsTest, AddAllElements) {
  const std::vector<double> vec({2.0, 3.0, 4.0});
  AddAllElements<double> add_all;
  EXPECT_NEAR(add_all(vec, 0), 9.0, 1e-10);
}

TEST(ElementOperationsTest, AddAllElementsEmpty) {
  const std::vector<double> vec;
  AddAllElements<double> add_all;
  EXPECT_NEAR(add_all(vec, 0), 0.0, 1e-10);
}

TEST(ElementOperationsTest, AverageAllElements) {
  const std::vector<double> vec({2.0, 3.0, 4.0});
  AverageAllElements<double> avg_all;
  EXPECT_NEAR(avg_all(vec, 0), 3.0, 1e-10);
}

TEST(ElementOperationsTest, AverageAllElementsEmpty) {
  const std::vector<double> vec;
  AverageAllElements<double> avg_all;
  EXPECT_NEAR(avg_all(vec, 0), 0.0, 1e-10);
}

TEST(ElementOperationsTest, BasicTfOptMaxAllElements) {
  const std::vector<double> vec({-5.0, 10.0, 20.0, 0.0, -10.0, 5.0});
  MaxAllElements<double> max_all_elements;
  EXPECT_EQ(max_all_elements(vec, 0), 20.0);
}

TEST(ElementOperationsTest, BasicTfOptMaxAllElementsEmptyInput) {
  const std::vector<double> vec;
  MaxAllElements<double> max_all_elements;
  EXPECT_EQ(max_all_elements(vec, 0), -std::numeric_limits<double>::infinity());
}

TEST(ElementOperationsTest, BasicTfOptMinAllElements) {
  const std::vector<double> vec({-5.0, 10.0, 20.0, 0.0, -10.0, 5.0});
  MinAllElements<double> min_all_elements;
  EXPECT_EQ(min_all_elements(vec, 0), -10.0);
}

TEST(ElementOperationsTest, BasicTfOptMinAllElementsEmptyInput) {
  const std::vector<double> vec;
  MinAllElements<double> min_all_elements;
  EXPECT_EQ(min_all_elements(vec, 0), std::numeric_limits<double>::infinity());
}

}  // namespace
}  // namespace tf_opt
