// Copyright 2025 The tf.opt Authors.
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

#ifndef TF_OPT_OPEN_SOURCE_STATUS_MATCHERS_H_
#define TF_OPT_OPEN_SOURCE_STATUS_MATCHERS_H_

#include <sstream>

#include "gmock/gmock.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tf_opt/open_source/status_macros.h"

namespace tf_opt {
namespace testing {

// Implements IsOk() as a polymorphic matcher.
MATCHER(IsOk, "") { return arg.ok(); }

#define TFOPT_ASSERT_OK(status_expr) \
  ASSERT_THAT(status_expr, ::tf_opt::testing::IsOk())

#define TFOPT_EXPECT_OK(status_expr) \
  EXPECT_THAT(status_expr, ::tf_opt::testing::IsOk())

#define TFOPT_ASSERT_OK_AND_ASSIGN(lhs, rexpr) \
  TFOPT_ASSERT_OK(rexpr);                      \
  lhs = (rexpr).value();

namespace internal {

// Monomorphic matcher for the error code of a Status.
bool StatusIsMatcher(const absl::Status& actual_status,
                     const absl::StatusCode& expected_error_code) {
  return actual_status.code() == expected_error_code;
}

// Monomorphic matcher for the error code of a StatusOr.
template <typename T>
bool StatusIsMatcher(const absl::StatusOr<T>& actual_status_or,
                     const absl::StatusCode& expected_error_code) {
  return StatusIsMatcher(actual_status_or.status(), expected_error_code);
}

// Monomorphic matcher for the error code & message of a Status.
bool StatusIsMatcher(
    const absl::Status& actual_status,
    const absl::StatusCode& expected_error_code,
    const ::testing::Matcher<const std::string&>& expected_message) {
  ::testing::StringMatchResultListener sink;
  return actual_status.code() == expected_error_code &&
         expected_message.MatchAndExplain(std::string(actual_status.message()),
                                          &sink);
}

// Monomorphic matcher for the error code & message of a StatusOr.
template <typename T>
bool StatusIsMatcher(
    const absl::StatusOr<T>& actual_status_or,
    const absl::StatusCode& expected_error_code,
    const ::testing::Matcher<const std::string&>& expected_message) {
  return StatusIsMatcher(actual_status_or.status(), expected_error_code,
                         expected_message);
}

}  // namespace internal

// Implements StatusIs() as a polymorphic matcher.
MATCHER_P(StatusIs, expected_error_code, "") {
  return ::tf_opt::testing::internal::StatusIsMatcher(arg, expected_error_code);
}

// Implements StatusIs() as a polymorphic matcher.
MATCHER_P2(StatusIs, expected_error_code, expected_message, "") {
  return ::tf_opt::testing::internal::StatusIsMatcher(arg, expected_error_code,
                                                      expected_message);
}

namespace internal {

// Monomorphic implementation of a matcher for a StatusOr.
template <typename StatusOrType>
class IsOkAndHoldsMatcherImpl
    : public ::testing::MatcherInterface<StatusOrType> {
 public:
  using ValueType = typename std::remove_reference<decltype(
      std::declval<StatusOrType>().value())>::type;

  template <typename InnerMatcher>
  explicit IsOkAndHoldsMatcherImpl(InnerMatcher&& inner_matcher)
      : inner_matcher_(::testing::SafeMatcherCast<const ValueType&>(
            std::forward<InnerMatcher>(inner_matcher))) {}

  void DescribeTo(std::ostream* os) const {
    *os << "is OK and has a value that ";
    inner_matcher_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const {
    *os << "isn't OK or has a value that ";
    inner_matcher_.DescribeNegationTo(os);
  }

  bool MatchAndExplain(StatusOrType actual_value,
                       ::testing::MatchResultListener* listener) const {
    if (!actual_value.ok()) {
      *listener << "which has status " << actual_value.status();
      return false;
    }

    ::testing::StringMatchResultListener inner_listener;
    const bool matches =
        inner_matcher_.MatchAndExplain(actual_value.value(), &inner_listener);
    const std::string inner_explanation = inner_listener.str();
    if (!inner_explanation.empty()) {
      *listener << "which contains value "
                << ::testing::PrintToString(actual_value.value()) << ", "
                << inner_explanation;
    }
    return matches;
  }

 private:
  const ::testing::Matcher<const ValueType&> inner_matcher_;
};

// Implements IsOkAndHolds() as a polymorphic matcher.
template <typename InnerMatcher>
class IsOkAndHoldsMatcher {
 public:
  explicit IsOkAndHoldsMatcher(InnerMatcher inner_matcher)
      : inner_matcher_(std::move(inner_matcher)) {}

  // Converts this polymorphic matcher to a monomorphic one of the given type.
  // StatusOrType can be either StatusOr<T> or a reference to StatusOr<T>.
  template <typename StatusOrType>
  operator ::testing::Matcher<StatusOrType>() const {
    return ::testing::MakeMatcher(
        new IsOkAndHoldsMatcherImpl<StatusOrType>(inner_matcher_));
  }

 private:
  const InnerMatcher inner_matcher_;
};

}  // namespace internal

// Returns a gMock matcher that matches a StatusOr<> whose status is
// OK and whose value matches the inner matcher.
template <typename InnerMatcher>
internal::IsOkAndHoldsMatcher<typename std::decay<InnerMatcher>::type>
IsOkAndHolds(InnerMatcher&& inner_matcher) {
  return internal::IsOkAndHoldsMatcher<typename std::decay<InnerMatcher>::type>(
      std::forward<InnerMatcher>(inner_matcher));
}

}  // namespace testing
}  // namespace tf_opt
#endif  // TF_OPT_OPEN_SOURCE_STATUS_MATCHERS_H_
