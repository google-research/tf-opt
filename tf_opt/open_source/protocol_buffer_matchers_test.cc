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

// Forked from third_party/exegesis/exegesis/testing/test_utils_test.cc

#include "tf_opt/open_source/protocol_buffer_matchers.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tf_opt/open_source/test.pb.h"

namespace tf_opt {
namespace {

using ::testing::Not;
using ::testing::Pointwise;
using ::testing::StringMatchResultListener;
using ::tf_opt::testing::EqualsProto;
using ::tf_opt::testing::TestProto;
using ::tf_opt::testing::proto::IgnoringFields;
using ::tf_opt::testing::proto::Partially;

TEST(EquasProtoMatcherTest, EqualsString) {
  TestProto actual_proto;
  actual_proto.set_integer_field(1);
  actual_proto.set_string_field("blabla");
  EXPECT_THAT(actual_proto,
              EqualsProto("integer_field: 1 string_field: 'blabla'"));
}

TEST(EquasProtoMatcherTest, EqualsProto) {
  TestProto actual_proto;
  actual_proto.set_integer_field(1);
  actual_proto.set_string_field("blabla");
  TestProto expected_proto;
  expected_proto.set_integer_field(1);
  expected_proto.set_string_field("blabla");
  EXPECT_THAT(actual_proto, EqualsProto(expected_proto));
}

TEST(EqualsProtoMatcherTest, DifferentProtos) {
  TestProto actual_proto;
  actual_proto.set_integer_field(1);
  auto matcher = EqualsProto<TestProto>("integer_field: 2");
  StringMatchResultListener listener;
  EXPECT_FALSE(matcher.impl().MatchAndExplain(actual_proto, &listener));
  EXPECT_EQ(listener.str(),
            "with the difference:\nmodified: integer_field: 2 -> 1");
}

TEST(IgnoringFieldsTest, IgnoresFields) {
  TestProto actual_proto;
  actual_proto.set_integer_field(1);
  EXPECT_THAT(actual_proto,
              IgnoringFields({"tf_opt.testing.TestProto.integer_field"},
                             EqualsProto("integer_field: 2")));
  EXPECT_THAT(actual_proto,
              Not(IgnoringFields(
                  {"tf_opt.testing.TestProto.integer_field"},
                  EqualsProto("integer_field: 2 string_field: 'hello'"))));

  actual_proto.mutable_message_field()->set_field_a(1);
  actual_proto.mutable_message_field()->set_field_b(2);
  EXPECT_THAT(
      actual_proto,
      IgnoringFields(
          {"tf_opt.testing.TestProto.integer_field",
           "tf_opt.testing.SubProto.field_a"},
          EqualsProto(
              "integer_field: 2 message_field { field_a: 2 field_b: 2 }")));
}

TEST(EqualsProtoPartiallyTest, Partially) {
  TestProto actual_proto;
  actual_proto.set_integer_field(1);
  actual_proto.set_string_field("blabla");
  EXPECT_THAT(actual_proto, Partially(EqualsProto("string_field: 'blabla'")));

  EXPECT_THAT(actual_proto,
              Not(Partially(EqualsProto("string_field: 'foobar'"))));
}

TEST(EqualsProtoTupleMatcherTest, Pointwise) {
  std::vector<TestProto> actual_protos(3);
  actual_protos[0].set_integer_field(1);
  actual_protos[1].set_string_field("hello");
  actual_protos[2].set_integer_field(2);
  actual_protos[2].set_string_field("world");
  const std::vector<std::string> expected_protos = {
      "integer_field: 1", "string_field: 'hello'",
      "integer_field: 2 string_field: 'world'"};
  EXPECT_THAT(actual_protos, Pointwise(EqualsProto(), expected_protos));
}


}  // namespace
}  // namespace tf_opt
