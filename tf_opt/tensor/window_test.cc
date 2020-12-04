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

#include "tf_opt/tensor/window.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tf_opt/open_source/status_matchers.h"

namespace tf_opt {
namespace {

using ::testing::HasSubstr;
using ::tf_opt::testing::StatusIs;

constexpr absl::StatusCode kInvalidArgument =
    absl::StatusCode::kInvalidArgument;

// NOTE: In the comments, all positions are in (row, column) form,
// zero-indexed. All sizes are in height x width form.

TEST(WindowTest, SingleStrideValidPaddingSimple) {
  WindowExtractor2D window_extractor;

  // 3x4 input, 2x3 window.
  // Expected window at output position (1,0):
  // ....
  // XWW.   <-- window is expected to start at (1,0)
  // WWW.
  const Position2D input_size(3, 4);
  const Position2D window_size(2, 3);
  const Position2D stride(1, 1);
  const absl::Status init_status = window_extractor.Initialize(
      input_size, window_size, stride, PaddingType::VALID);
  const Rectangle rectangle = window_extractor.GetWindow(Position2D(1, 0));

  TFOPT_ASSERT_OK(init_status);
  EXPECT_EQ(rectangle.start.row, 1);
  EXPECT_EQ(rectangle.start.col, 0);
  EXPECT_EQ(rectangle.size.row, 2);
  EXPECT_EQ(rectangle.size.col, 3);
}

TEST(WindowTest, TwoTwoStrideValidPaddingSimple) {
  WindowExtractor2D window_extractor;

  // 4x5 input, 2x1 window. Row strides 2, column stride 2.
  // Expected window at output position (1,2): (= (2,4) in input)
  // .....
  // .....
  // ....X   <-- window is expected to start at (2,4)
  // ....W
  //
  // Stride shape:
  // X.X.X
  // .....
  // X.X.X
  // .....
  const Position2D input_size(4, 5);
  const Position2D window_size(2, 1);
  const Position2D stride(2, 2);
  const absl::Status init_status = window_extractor.Initialize(
      input_size, window_size, stride, PaddingType::VALID);
  const Rectangle rectangle = window_extractor.GetWindow(Position2D(1, 2));

  TFOPT_ASSERT_OK(init_status);
  EXPECT_EQ(rectangle.start.row, 2);
  EXPECT_EQ(rectangle.start.col, 4);
  EXPECT_EQ(rectangle.size.row, 2);
  EXPECT_EQ(rectangle.size.col, 1);
}

TEST(WindowTest, ThreeTwoStrideValidPaddingSimple) {
  WindowExtractor2D window_extractor;

  // 5x5 input, 2x1 window. Row stride 3, column stride 2.
  // Expected window at output position (1,2): (= (4,4) in input)
  // .....
  // .....
  // .....
  // ....W   <-- window is expected to start at (4,3)
  // ....X
  //
  // Stride shape:
  // .....
  // X.X.X
  // .....
  // .....
  // X.X.X
  const Position2D input_size(5, 5);
  const Position2D window_size(2, 1);
  const Position2D stride(3, 2);
  const absl::Status init_status = window_extractor.Initialize(
      input_size, window_size, stride, PaddingType::VALID);
  const Rectangle rectangle = window_extractor.GetWindow(Position2D(1, 2));

  TFOPT_ASSERT_OK(init_status);
  EXPECT_EQ(rectangle.start.col, 4);
  EXPECT_EQ(rectangle.start.row, 3);
  EXPECT_EQ(rectangle.size.col, 1);
  EXPECT_EQ(rectangle.size.row, 2);
}

TEST(WindowTest, SingleStrideValidPaddingSimpleOutputDimensions) {
  WindowExtractor2D window_extractor;

  // 3x4 input, 2x3 window.
  const Position2D input_size(3, 4);
  const Position2D window_size(2, 3);
  const Position2D stride(1, 1);
  const absl::Status init_status = window_extractor.Initialize(
      input_size, window_size, stride, PaddingType::VALID);

  TFOPT_ASSERT_OK(init_status);
  EXPECT_EQ(window_extractor.output_size().col, 2);
  EXPECT_EQ(window_extractor.output_size().row, 2);
}

TEST(WindowTest, SingleStrideSamePaddingSimple) {
  WindowExtractor2D window_extractor;

  // 4x5 input, 4x5 window.
  // Expected window at output position (0,1):
  // *WWWWW***   <-- window is expected to start at (-1,-1)
  // *WWXWW.**
  // *WWWWW.**
  // *WWWWW.**
  // **.....**
  // *********   * = padding
  // *********
  // Note: The X is the center of the window, which, when padding is SAME,
  // corresponds to the same position in the output. Note that the center is
  // rounded down if the width or height is odd.
  const Position2D input_size(4, 5);
  const Position2D window_size(4, 5);
  const Position2D stride(1, 1);
  const absl::Status init_status = window_extractor.Initialize(
      input_size, window_size, stride, PaddingType::SAME);
  const Rectangle rectangle = window_extractor.GetWindow(Position2D(0, 1));

  TFOPT_ASSERT_OK(init_status);
  EXPECT_EQ(rectangle.start.row, -1);
  EXPECT_EQ(rectangle.start.col, -1);
  EXPECT_EQ(rectangle.size.row, 4);
  EXPECT_EQ(rectangle.size.col, 5);
}

TEST(WindowTest, SingleStrideSamePaddingEnd) {
  WindowExtractor2D window_extractor;

  // 4x5 input, 4x5 window.
  // Expected window at output position (3,4):
  // *********   * = padding
  // **.....**
  // **.....**
  // **..WWWWW   <-- window is expected to start at (2,2)
  // **..WWXWW
  // ****WWWWW
  // ****WWWWW
  const Position2D input_size(4, 5);
  const Position2D window_size(4, 5);
  const Position2D stride(1, 1);
  const absl::Status init_status = window_extractor.Initialize(
      input_size, window_size, stride, PaddingType::SAME);
  const Rectangle rectangle = window_extractor.GetWindow(Position2D(3, 4));

  TFOPT_ASSERT_OK(init_status);
  EXPECT_EQ(rectangle.start.row, 2);
  EXPECT_EQ(rectangle.start.col, 2);
  EXPECT_EQ(rectangle.size.row, 4);
  EXPECT_EQ(rectangle.size.col, 5);
}

TEST(WindowTest, TwoThreeStrideSamePaddingEnd) {
  WindowExtractor2D window_extractor;

  // 4x5 input, 4x5 window. Row stride 2, column stride 3.
  // Expected window at output position (1,1):
  // ********   * = padding
  // *.X..X**
  // *..WWWWW   <-- window is expected to start at (1,2)
  // *.XWWXWW
  // *..WWWWW
  // ***WWWWW
  //
  // Stride shape:
  // ********
  // *.X..X**
  // *.....**
  // *.X..X**
  // *.....**
  // ********
  const Position2D input_size(4, 5);
  const Position2D window_size(4, 5);
  const Position2D stride(2, 3);
  const absl::Status init_status = window_extractor.Initialize(
      input_size, window_size, stride, PaddingType::SAME);
  const Rectangle rectangle = window_extractor.GetWindow(Position2D(1, 1));

  TFOPT_ASSERT_OK(init_status);
  EXPECT_EQ(rectangle.start.row, 1);
  EXPECT_EQ(rectangle.start.col, 2);
  EXPECT_EQ(rectangle.size.row, 4);
  EXPECT_EQ(rectangle.size.col, 5);
}

TEST(WindowTest, TwoThreeStrideSamePaddingOutputDimensions) {
  WindowExtractor2D window_extractor;

  // 4x5 input, 4x5 window. Row stride 2, column stride 3.
  const Position2D input_size(4, 5);
  const Position2D window_size(4, 5);
  const Position2D stride(2, 3);
  const absl::Status init_status = window_extractor.Initialize(
      input_size, window_size, stride, PaddingType::SAME);

  TFOPT_ASSERT_OK(init_status);
  EXPECT_EQ(window_extractor.output_size().row, 2);
  EXPECT_EQ(window_extractor.output_size().col, 2);
}

TEST(WindowDeathTest, SingleStrideValidPaddingInvalidOutputPosition) {
  WindowExtractor2D window_extractor;

  // 3x4 input, 2x3 window.
  // Expected window at output position (1,2):
  // ....
  // ..XWW  <-- window is out of bounds; error expected
  // ..WWW
  const Position2D input_size(3, 4);
  const Position2D window_size(2, 3);
  const Position2D stride(1, 1);
  const absl::Status init_status = window_extractor.Initialize(
      input_size, window_size, stride, PaddingType::VALID);
  TFOPT_ASSERT_OK(init_status);
  ASSERT_DEATH(window_extractor.GetWindow(Position2D(1, 2)), "");
}

TEST(WindowTest, ValidPaddingWindowHeightLargerThanInputHeight) {
  WindowExtractor2D window_extractor;

  const Position2D input_size(3, 3);
  const Position2D window_size(4, 2);
  const Position2D stride(1, 1);
  const absl::Status init_status = window_extractor.Initialize(
      input_size, window_size, stride, PaddingType::VALID);

  EXPECT_THAT(init_status,
              StatusIs(kInvalidArgument, HasSubstr("window does not fit")));
}

TEST(WindowTest, ValidPaddingWindowWidthLargerThanInputWidth) {
  WindowExtractor2D window_extractor;

  const Position2D input_size(3, 3);
  const Position2D window_size(2, 4);
  const Position2D stride(1, 1);
  const absl::Status init_status = window_extractor.Initialize(
      input_size, window_size, stride, PaddingType::VALID);

  EXPECT_THAT(init_status,
              StatusIs(kInvalidArgument, HasSubstr("window does not fit")));
}

TEST(WindowTest, SamePaddingWindowHeightLargerThanInputHeight) {
  WindowExtractor2D window_extractor;

  const Position2D input_size(3, 3);
  const Position2D window_size(4, 2);
  const Position2D stride(1, 1);
  const absl::Status init_status = window_extractor.Initialize(
      input_size, window_size, stride, PaddingType::SAME);

  TFOPT_EXPECT_OK(init_status);
  EXPECT_EQ(window_extractor.output_size().col, 3);
  EXPECT_EQ(window_extractor.output_size().row, 3);
}

TEST(WindowTest, SamePaddingWindowWidthLargerThanInputWidth) {
  WindowExtractor2D window_extractor;

  const Position2D input_size(3, 3);
  const Position2D window_size(2, 4);
  const Position2D stride(1, 1);
  const absl::Status init_status = window_extractor.Initialize(
      input_size, window_size, stride, PaddingType::SAME);

  TFOPT_EXPECT_OK(init_status);
  EXPECT_EQ(window_extractor.output_size().col, 3);
  EXPECT_EQ(window_extractor.output_size().row, 3);
}

TEST(WindowTest, InitializeNonpositiveInputHeight) {
  WindowExtractor2D window_extractor;
  const absl::Status init_status = window_extractor.Initialize(
      Position2D(0, 3), Position2D(2, 2), Position2D(1, 1), PaddingType::VALID);
  EXPECT_THAT(init_status, StatusIs(kInvalidArgument,
                                    HasSubstr("Expected input height > 0")));
}

TEST(WindowTest, InitializeNonpositiveInputWidth) {
  WindowExtractor2D window_extractor;
  const absl::Status init_status = window_extractor.Initialize(
      Position2D(3, 0), Position2D(2, 2), Position2D(1, 1), PaddingType::VALID);
  EXPECT_THAT(init_status, StatusIs(kInvalidArgument,
                                    HasSubstr("Expected input width > 0")));
}

TEST(WindowTest, InitializeNonpositiveWindowHeight) {
  WindowExtractor2D window_extractor;
  const absl::Status init_status = window_extractor.Initialize(
      Position2D(3, 3), Position2D(0, 2), Position2D(1, 1), PaddingType::VALID);
  EXPECT_THAT(init_status, StatusIs(kInvalidArgument,
                                    HasSubstr("Expected window height > 0")));
}

TEST(WindowTest, InitializeNonpositiveWindowWidth) {
  WindowExtractor2D window_extractor;
  const absl::Status init_status = window_extractor.Initialize(
      Position2D(3, 3), Position2D(2, 0), Position2D(1, 1), PaddingType::VALID);
  EXPECT_THAT(init_status, StatusIs(kInvalidArgument,
                                    HasSubstr("Expected window width > 0")));
}

TEST(WindowTest, InitializeNonpositiveStrideRow) {
  WindowExtractor2D window_extractor;
  const absl::Status init_status = window_extractor.Initialize(
      Position2D(3, 3), Position2D(2, 2), Position2D(0, 1), PaddingType::VALID);
  EXPECT_THAT(init_status,
              StatusIs(kInvalidArgument, HasSubstr("Expected stride row > 0")));
}

TEST(WindowTest, InitializeNonpositiveStrideCol) {
  WindowExtractor2D window_extractor;
  const absl::Status init_status = window_extractor.Initialize(
      Position2D(3, 3), Position2D(2, 2), Position2D(1, 0), PaddingType::VALID);
  EXPECT_THAT(init_status,
              StatusIs(kInvalidArgument, HasSubstr("Expected stride col > 0")));
}

}  // namespace
}  // namespace tf_opt
