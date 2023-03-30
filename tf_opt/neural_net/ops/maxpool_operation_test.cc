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

#include "tf_opt/neural_net/ops/maxpool_operation.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "tf_opt/neural_net/operation.h"
#include "tf_opt/neural_net/operation_testing.h"
#include "tf_opt/neural_net/ops/constant_operation.h"
#include "tf_opt/open_source/status_matchers.h"
#include "tf_opt/tensor/shape.h"
#include "tf_opt/tensor/tensor.h"
#include "tf_opt/tensor/window.h"

namespace tf_opt {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::tf_opt::testing::IsOkAndHolds;
using ::tf_opt::testing::StatusIs;
constexpr absl::StatusCode kInvalidArgument =
    absl::StatusCode::kInvalidArgument;

Operation::Options MakeMaxpoolOptions(const Position2D ksize,
                                      const Position2D stride,
                                      PaddingType padding,
                                      MaximumImplementationType formulation) {
  Operation::Options options;
  options.integer_options[MaxpoolOperation::kOptionsWindowHeightKey] =
      static_cast<int>(ksize.row);
  options.integer_options[MaxpoolOperation::kOptionsWindowWidthKey] =
      static_cast<int>(ksize.col);
  options.integer_options[MaxpoolOperation::kOptionsStrideRowKey] =
      static_cast<int>(stride.row);
  options.integer_options[MaxpoolOperation::kOptionsStrideColKey] =
      static_cast<int>(stride.col);
  options.string_options[MaxpoolOperation::kOptionsPaddingKey] =
      ToString(padding);
  options.string_options[MaxpoolOperation::kOptionsFormulationKey] =
      ToString(formulation);
  return options;
}

class MaxpoolOperationTest : public ::testing::Test {
 protected:
  absl::StatusOr<Shape> OutputShape() const {
    return MaxpoolOperation::OutputShape(input_shape_, ksize_, stride_,
                                         padding_);
  }

  Operation::Options MakeOptions() {
    return MakeMaxpoolOptions(ksize_, stride_, padding_, formulation_);
  }

  absl::StatusOr<MaxpoolOperation> Create() const {
    return MaxpoolOperation::Create(op_name_, input_shape_, ksize_, stride_,
                                    padding_, formulation_);
  }

  absl::StatusOr<MaxpoolOperation> GenericCreate() const {
    return MaxpoolOperation::GenericCreate(op_name_, generic_inputs_,
                                           output_shape_, options_);
  }

  void ExpectOperationCorrect(const MaxpoolOperation& op) {
    EXPECT_THAT(op, OperationArgsAre(op_name_, {input_shape_}, output_shape_));
    EXPECT_EQ(op.input(), input_shape_);
    EXPECT_EQ(op.ksize(), ksize_);
    EXPECT_EQ(op.stride(), stride_);
    EXPECT_EQ(op.padding(), padding_);
    EXPECT_EQ(op.formulation(), formulation_);
  }

  std::string op_name_ = "maxpool1";
  Shape input_shape_ = Shape({1, 3, 3, 1});
  Position2D ksize_ = Position2D(2, 2);
  Position2D stride_ = Position2D(1, 1);
  PaddingType padding_ = PaddingType::VALID;
  MaximumImplementationType formulation_ =
      MaximumImplementationType::kOptimalBigM;
  Shape output_shape_ = Shape({1, 2, 2, 1});
  ConstantOperation input_operation_ =
      CreateOrDie<ConstantOperation>("c1", DoubleTensor(input_shape_));
  std::vector<Shape> generic_inputs_ = {input_shape_};
  Operation::Options options_ = MakeOptions();
};

TEST_F(MaxpoolOperationTest, OutputShapeBasic) {
  EXPECT_THAT(OutputShape(), IsOkAndHolds(output_shape_));
}

TEST_F(MaxpoolOperationTest, SimpleCreate) {
  TFOPT_ASSERT_OK_AND_ASSIGN(const auto op, Create());
  ExpectOperationCorrect(op);
}

TEST_F(MaxpoolOperationTest, CreateBadInput) {
  input_shape_ = Shape({3, 3, 1});
  EXPECT_THAT(Create(), StatusIs(kInvalidArgument,
                                 HasSubstr("Expected input to be rank four")));
}

TEST_F(MaxpoolOperationTest, GenericCreate) {
  TFOPT_ASSERT_OK_AND_ASSIGN(const auto op, GenericCreate());
  ExpectOperationCorrect(op);
}

TEST_F(MaxpoolOperationTest, GenericCreateBadNumInputs) {
  generic_inputs_ = {input_shape_, input_shape_};
  EXPECT_THAT(
      GenericCreate(),
      StatusIs(kInvalidArgument,
               HasSubstr("Expected number of inputs equals to 1, found: 2")));
}

TEST_F(MaxpoolOperationTest, GenericCreateBadOutputShape) {
  output_shape_ = Shape({7});
  EXPECT_THAT(GenericCreate(),
              StatusIs(kInvalidArgument, HasSubstr("Expected output shape:")));
}

TEST_F(MaxpoolOperationTest, GenericCreateExtraOption) {
  options_.string_options["bad_key"] = "bad_value";
  EXPECT_THAT(GenericCreate(),
              StatusIs(kInvalidArgument,
                       HasSubstr("Expected number of options at most")));
}

TEST_F(MaxpoolOperationTest, GenericCreateMissingStrideCol) {
  options_.integer_options.erase(MaxpoolOperation::kOptionsStrideColKey);
  EXPECT_THAT(GenericCreate(),
              StatusIs(kInvalidArgument,
                       HasSubstr("Required integer option not found")));
}

TEST_F(MaxpoolOperationTest, GenericCreateMissingStrideRow) {
  options_.integer_options.erase(MaxpoolOperation::kOptionsStrideRowKey);
  EXPECT_THAT(GenericCreate(),
              StatusIs(kInvalidArgument,
                       HasSubstr("Required integer option not found")));
}

TEST_F(MaxpoolOperationTest, GenericCreateMissingkSizeRow) {
  options_.integer_options.erase(MaxpoolOperation::kOptionsWindowHeightKey);
  EXPECT_THAT(GenericCreate(),
              StatusIs(kInvalidArgument,
                       HasSubstr("Required integer option not found")));
}

TEST_F(MaxpoolOperationTest, GenericCreateMissingkSizeCol) {
  options_.integer_options.erase(MaxpoolOperation::kOptionsWindowWidthKey);
  EXPECT_THAT(GenericCreate(),
              StatusIs(kInvalidArgument,
                       HasSubstr("Required integer option not found")));
}

TEST_F(MaxpoolOperationTest, GenericCreateMissingPadding) {
  options_.string_options.erase(MaxpoolOperation::kOptionsPaddingKey);
  EXPECT_THAT(GenericCreate(),
              StatusIs(kInvalidArgument,
                       HasSubstr("Required string option not found")));
}

TEST_F(MaxpoolOperationTest, GenericCreateBadPadding) {
  options_.string_options[MaxpoolOperation::kOptionsPaddingKey] = "bad_value";
  EXPECT_THAT(GenericCreate(),
              StatusIs(kInvalidArgument, HasSubstr("Invalid padding string")));
}

TEST_F(MaxpoolOperationTest, GenericCreateBadFormulation) {
  options_.string_options[MaxpoolOperation::kOptionsFormulationKey] =
      "bad_formulation";
  EXPECT_THAT(GenericCreate(),
              StatusIs(kInvalidArgument,
                       HasSubstr("Unrecognized formulation name for maximum")));
}

}  // namespace
}  // namespace tf_opt
