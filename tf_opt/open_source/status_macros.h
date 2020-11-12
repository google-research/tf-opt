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

#ifndef TF_OPT_OPEN_SOURCE_STATUS_MACROS_H_
#define TF_OPT_OPEN_SOURCE_STATUS_MACROS_H_

#include <sstream>

#include "glog/logging.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tf_opt/open_source/status_builder.h"

namespace tf_opt {

#define TFOPT_CHECK_OK(val) CHECK_EQ(::absl::OkStatus(), (val))
#define TFOPT_CHECK_OK(val) CHECK_EQ(::absl::OkStatus(), (val))
#define TFOPT_DCHECK_OK(val) DCHECK_EQ(::absl::OkStatus(), (val))

#define TFOPT_RETURN_IF_ERROR(expr)                                   \
  TFOPT_IMPL_ELSE_BLOCKER_                                            \
  if (::tf_opt::StatusAdaptorForMacros status_macro_internal_adaptor{ \
          (expr)}) {                                                  \
  } else /* NOLINT */                                                 \
    return status_macro_internal_adaptor.Consume()

#define TFOPT_ASSIGN_OR_RETURN(...)                                      \
  TFOPT_IMPL_GET_VARIADIC_((__VA_ARGS__, TFOPT_IMPL_ASSIGN_OR_RETURN_3_, \
                            TFOPT_IMPL_ASSIGN_OR_RETURN_2_))             \
  (__VA_ARGS__)

// =================================================================
// == Implementation details, do not rely on anything below here. ==
// =================================================================

// Some builds do not support C++14 fully yet, using C++11 constexpr technique.
constexpr bool HasPotentialConditionalOperator(const char* lhs, int index) {
  return (index == -1 ? false
                      : (lhs[index] == '?' ? true
                                           : HasPotentialConditionalOperator(
                                                 lhs, index - 1)));
}

// MSVC incorrectly expands variadic macros, splice together a macro call to
// work around the bug.
#define TFOPT_IMPL_GET_VARIADIC_HELPER_(_1, _2, _3, NAME, ...) NAME
#define TFOPT_IMPL_GET_VARIADIC_(args) TFOPT_IMPL_GET_VARIADIC_HELPER_ args

#define TFOPT_IMPL_ASSIGN_OR_RETURN_2_(lhs, rexpr) \
  TFOPT_IMPL_ASSIGN_OR_RETURN_3_(lhs, rexpr, std::move(_))
#define TFOPT_IMPL_ASSIGN_OR_RETURN_3_(lhs, rexpr, error_expression)           \
  TFOPT_IMPL_ASSIGN_OR_RETURN_(TFOPT_IMPL_CONCAT_(_status_or_value, __LINE__), \
                               lhs, rexpr, error_expression)
#define TFOPT_IMPL_ASSIGN_OR_RETURN_(statusor, lhs, rexpr, error_expression) \
  auto statusor = (rexpr);                                                   \
  if (ABSL_PREDICT_FALSE(!statusor.ok())) {                                  \
    ::tf_opt::StatusBuilder _(std::move(statusor).status());                 \
    (void)_; /* error_expression is allowed to not use this variable */      \
    return (error_expression);                                               \
  }                                                                          \
  {                                                                          \
    static_assert(                                                           \
        #lhs[0] != '(' || #lhs[sizeof(#lhs) - 2] != ')' ||                   \
            !HasPotentialConditionalOperator(#lhs, sizeof(#lhs) - 2),        \
        "Identified potential conditional operator, consider not "           \
        "using TFOPT_ASSIGN_OR_RETURN");                                     \
  }                                                                          \
  TFOPT_IMPL_UNPARENTHESIZE_IF_PARENTHESIZED(lhs) = std::move(statusor).value()

// Internal helpers for macro expansion.
#define TFOPT_IMPL_EAT(...)
#define TFOPT_IMPL_REM(...) __VA_ARGS__
#define TFOPT_IMPL_EMPTY()

// Internal helpers for emptyness arguments check.
#define TFOPT_IMPL_IS_EMPTY_INNER(...) \
  TFOPT_IMPL_IS_EMPTY_INNER_HELPER((__VA_ARGS__, 0, 1))
// MSVC expands variadic macros incorrectly, so we need this extra indirection
// to work around that (b/110959038).
#define TFOPT_IMPL_IS_EMPTY_INNER_HELPER(args) TFOPT_IMPL_IS_EMPTY_INNER_I args
#define TFOPT_IMPL_IS_EMPTY_INNER_I(e0, e1, is_empty, ...) is_empty

#define TFOPT_IMPL_IS_EMPTY(...) TFOPT_IMPL_IS_EMPTY_I(__VA_ARGS__)
#define TFOPT_IMPL_IS_EMPTY_I(...) TFOPT_IMPL_IS_EMPTY_INNER(_, ##__VA_ARGS__)

// Internal helpers for if statement.
#define TFOPT_IMPL_IF_1(_Then, _Else) _Then
#define TFOPT_IMPL_IF_0(_Then, _Else) _Else
#define TFOPT_IMPL_IF(_Cond, _Then, _Else) \
  TFOPT_IMPL_CONCAT_(TFOPT_IMPL_IF_, _Cond)(_Then, _Else)

// Expands to 1 if the input is parenthesized. Otherwise expands to 0.
#define TFOPT_IMPL_IS_PARENTHESIZED(...) \
  TFOPT_IMPL_IS_EMPTY(TFOPT_IMPL_EAT __VA_ARGS__)

// If the input is parenthesized, removes the parentheses. Otherwise expands to
// the input unchanged.
#define TFOPT_IMPL_UNPARENTHESIZE_IF_PARENTHESIZED(...)                   \
  TFOPT_IMPL_IF(TFOPT_IMPL_IS_PARENTHESIZED(__VA_ARGS__), TFOPT_IMPL_REM, \
                TFOPT_IMPL_EMPTY())                                       \
  __VA_ARGS__

// Internal helper for concatenating macro values.
#define TFOPT_IMPL_CONCAT_INNER_(x, y) x##y
#define TFOPT_IMPL_CONCAT_(x, y) TFOPT_IMPL_CONCAT_INNER_(x, y)

// The GNU compiler emits a warning for code like:
//
//   if (foo)
//     if (bar) { } else baz;
//
// because it thinks you might want the else to bind to the first if.  This
// leads to problems with code like:
//
//   if (do_expr) TFOPT_RETURN_IF_ERROR(expr) << "Some message";
//
// The "switch (0) case 0:" idiom is used to suppress this.
#define TFOPT_IMPL_ELSE_BLOCKER_ \
  switch (0)                     \
  case 0:                        \
  default:  // NOLINT

// Provides a conversion to bool so that it can be used inside an if statement
// that declares a variable.
class StatusAdaptorForMacros {
 public:
  explicit StatusAdaptorForMacros(const absl::Status& status)
      : builder_(status) {}

  explicit StatusAdaptorForMacros(absl::Status&& status)
      : builder_(std::move(status)) {}

  explicit StatusAdaptorForMacros(::tf_opt::StatusBuilder&& builder)
      : builder_(std::move(builder)) {}

  StatusAdaptorForMacros(const StatusAdaptorForMacros&) = delete;
  StatusAdaptorForMacros& operator=(const StatusAdaptorForMacros&) = delete;

  explicit operator bool() const { return ABSL_PREDICT_TRUE(builder_.ok()); }

  ::tf_opt::StatusBuilder&& Consume() { return std::move(builder_); }

 private:
  ::tf_opt::StatusBuilder builder_;
};

}  // namespace tf_opt

#endif  // TF_OPT_OPEN_SOURCE_STATUS_MACROS_H_
