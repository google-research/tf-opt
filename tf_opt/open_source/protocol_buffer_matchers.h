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

// gMock matchers used to validate protocol buffer arguments. A fork of the
// official gmock protocol-buffer-matchers.h, from October 6th, 2020. See also
// b/135192747.
//
// WHAT THIS IS
// ============
//
// This library defines the matchers in the ::tfopt::testing namespace:
//
//   EqualsProto(pb)              The argument equals pb.
//   EqualsInitializedProto(pb)   The argument is initialized and equals pb.
//   EquivToProto(pb)             The argument is equivalent to pb.
//   EquivToInitializedProto(pb)  The argument is initialized and equivalent
//                                to pb.
//   IsInitializedProto()         The argument is an initialized protobuf.
//
// where:
//
//   - pb can be either a protobuf value or a human-readable string
//     representation of it.
//   - When pb is a string, the matcher can optionally accept a
//     template argument for the type of the protobuf,
//     e.g. EqualsProto<Foo>("foo: 1").
//   - "equals" is defined as the argument's Equals(pb) method returns true.
//   - "equivalent to" is defined as the argument's Equivalent(pb) method
//     returns true.
//   - "initialized" means that the argument's IsInitialized() method returns
//     true.
//
// These matchers can match either a protobuf value or a pointer to
// it.  They make a copy of pb, and thus can out-live pb.  When the
// match fails, the matchers print a detailed message (the value of
// the actual protobuf, the value of the expected protobuf, and which
// fields are different).
//
// This library also defines the following matcher transformer
// functions in the ::tfopt::testing::proto namespace:
//
//   Approximately(m, margin, fraction)
//                     The same as m, except that it compares
//                     floating-point fields approximately (using
//                     google::protobuf::util::MessageDifferencer's APPROXIMATE
//                     comparison option).  m can be any of the
//                     Equals* and EquivTo* protobuf matchers above. If margin
//                     is specified, floats and doubles will be considered
//                     approximately equal if they are within that margin, i.e.
//                     abs(expected - actual) <= margin. If fraction is
//                     specified, floats and doubles will be considered
//                     approximately equal if they are within a fraction of
//                     their magnitude, i.e. abs(expected - actual) <=
//                     fraction * max(abs(expected), abs(actual)). Two fields
//                     will be considered equal if they're within the fraction
//                     _or_ within the margin, so omitting or setting the
//                     fraction to 0.0 will only check against the margin.
//                     Similarly, setting the margin to 0.0 will only check
//                     using the fraction. If margin and fraction are omitted,
//                     MathLimits<T>::kStdError for that type (T=float or
//                     T=double) is used for both the margin and fraction.
//   TreatingNaNsAsEqual(m)
//                     The same as m, except that treats floating-point fields
//                     that are NaN as equal. m can be any of the Equals* and
//                     EquivTo* protobuf matchers above.
//   IgnoringFields(fields, m)
//                     The same as m, except the specified fields will be
//                     ignored when matching (using
//                     google::protobuf::util::MessageDifferencer::IgnoreField). fields is
//                     represented as a container or an initializer list of
//                     strings and each element is specified by their fully
//                     qualified names, i.e., the names corresponding to
//                     FieldDescriptor.full_name().  m can be
//                     any of the Equals* and EquivTo* protobuf matchers above.
//                     It can also be any of the transformer matchers listed
//                     here (e.g. Approximately, TreatingNaNsAsEqual) as long as
//                     the intent of the each concatenated matcher is mutually
//                     exclusive (e.g. using IgnoringFields in conjunction with
//                     Partially can have different results depending on whether
//                     the fields specified in IgnoringFields is part of the
//                     fields covered by Partially).
//   IgnoringFieldPaths(field_paths, m)
//                     The same as m, except the specified fields will be
//                     ignored when matching. field_paths is represented as a
//                     container or an initializer list of strings and
//                     each element is specified by their path relative to the
//                     proto being matched by m. Paths can contain indices
//                     and/or extensions. Examples:
//                       Ignores field singular_field/repeated_field:
//                         singular_field
//                         repeated_field
//                       Ignores just the third repeated_field instance:
//                         repeated_field[2]
//                       Ignores some_field in singular_nested/repeated_nested:
//                         singular_nested.some_field
//                         repeated_nested.some_field
//                       Ignores some_field in instance 2 of repeated_nested:
//                         repeated_nested[2].some_field
//                       Ignores extension SomeExtension.msg of repeated_nested:
//                         repeated_nested.(package.SomeExtension.msg)
//                       Ignores subfield of extension:
//                         repeated_nested.(package.SomeExtension.msg).subfield
//                     If you are trying to ignore fields from a proto group,
//                     please note that the group name is converted to lower
//                     case. The same restrictions as for IgnoringFields apply.
//   IgnoringRepeatedFieldOrdering(m)
//                     The same as m, except that it ignores the relative
//                     ordering of elements within each repeated field in m.
//                     See google::protobuf::util::MessageDifferencer::TreatAsSet() for
//                     more details.
//   Partially(m)
//                     The same as m, except that only fields present in
//                     the expected protobuf are considered (using
//                     google::protobuf::util::MessageDifferencer's PARTIAL
//                     comparison option).   m can be any of the
//                     Equals* and EquivTo* protobuf matchers above.
//   WhenDeserialized(typed_pb_matcher)
//                     The string argument is a serialization of a
//                     protobuf that matches typed_pb_matcher.
//                     typed_pb_matcher can be an Equals* or EquivTo*
//                     protobuf matcher (possibly with Approximately()
//                     or Partially() modifiers) where the type of the
//                     protobuf is known at run time (e.g. it cannot
//                     be EqualsProto("...") as it's unclear what type
//                     the string represents).
//   WhenDeserializedAs<PB>(pb_matcher)
//                     Like WhenDeserialized(), except that the type
//                     of the deserialized protobuf must be PB.  Since
//                     the protobuf type is known, pb_matcher can be *any*
//                     valid protobuf matcher, including EqualsProto("...").
//   WhenParsedFromProtoText(typed_pb_matcher)
//                     The string argument is a text-format of a
//                     protobuf that matches typed_pb_matcher.
//                     typed_pb_matcher can be an Equals* or EquivTo*
//                     protobuf matcher (possibly with Approximately()
//                     or Partially() modifiers) where the type of the
//                     protobuf is known at run time (e.g. it cannot
//                     be EqualsProto("...") as it's unclear what type
//                     the string represents).
//   WhenParsedFromProtoTextAs<PB>(pb_matcher)
//                     Like WhenParsedFromProtoText(), except that the type of
//                     the parsed protobuf must be PB.  Since the protobuf type
//                     is known, pb_matcher can be *any* valid protobuf matcher,
//                     including EqualsProto("...").
//
// Approximately(), TreatingNaNsAsEqual(), Partially(), IgnoringFields(), and
// IgnoringRepeatedFieldOrdering() can be combined (nested)
// and the composition order is irrelevant:
//
//   Approximately(Partially(EquivToProto(pb)))
// and
//   Partially(Approximately(EquivToProto(pb)))
// are the same thing.
//
// EXAMPLES
// ========
//
//   using ::tf_opt::testing::EqualsProto;
//   using ::tf_opt::testing::EquivToProto;
//   using ::tf_opt::testing::proto::Approximately;
//   using ::tf_opt::testing::proto::Partially;
//   using ::tf_opt::testing::proto::WhenDeserialized;
//   using ::tf_opt::testing::proto::WhenParsedFromProtoText;
//   using ::tf_opt::testing::proto::WhenUnpacked;
//
//   // my_pb.Equals(expected_pb).
//   EXPECT_THAT(my_pb, EqualsProto(expected_pb));
//
//   // my_pb is equivalent to a protobuf whose foo field is 1 and
//   // whose bar field is "x".
//   EXPECT_THAT(my_pb, EquivToProto(R"(foo: 1  # In-string comment
//                                      bar: 'x')"));
//
//   // my_pb is equal to expected_pb, comparing all floating-point
//   // fields approximately.
//   EXPECT_THAT(my_pb, Approximately(EqualsProto(expected_pb)));
//
//   // my_pb is equivalent to expected_pb.  A field is ignored in the
//   // comparison if it's present in my_pb but not in expected_pb.
//   EXPECT_THAT(my_pb, Partially(EquivToProto(expected_pb)));
//
//   string data;
//   my_pb.SerializeToString(&data);
//   // data can be deserialized to a protobuf that equals expected_pb.
//   EXPECT_THAT(data, WhenDeserialized(EqualsProto(expected_pb)));
//   // The following line doesn't compile, as the matcher doesn't know
//   // the type of the protobuf.
//   // EXPECT_THAT(data, WhenDeserialized(EqualsProto("foo: 1")));
//
//   string text = my_pb.DebugString();
//   EXPECT_THAT(text, WhenParsedFromProtoText(EqualsProto(expected_pb)));
//   // The following line doesn't compile, as the matcher doesn't know
//   // the type of the protobuf.
//   // EXPECT_THAT(text, WhenParsedFromProtoText(EqualsProto("foo: 1")));

#ifndef TF_OPT_OPEN_SOURCE_PROTOCOL_BUFFER_MATCHERS_H_
#define TF_OPT_OPEN_SOURCE_PROTOCOL_BUFFER_MATCHERS_H_

#include <initializer_list>
#include <iostream>  // NOLINT
#include <memory>
#include <sstream>  // NOLINT
#include <string>   // NOLINT
#include <vector>   // NOLINT

#include "ortools/base/logging.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/map.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/field_comparator.h"
#include "google/protobuf/util/message_differencer.h"
#include "gmock/gmock.h"
#include "absl/memory/memory.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace google {
namespace protobuf {
using GeneratedMessageBaseType = Message;
}  // namespace protobuf
}  // namespace google

namespace tf_opt {
namespace testing {
namespace internal {

// Utilities.

// How to compare two fields (equal vs. equivalent).
typedef google::protobuf::util::MessageDifferencer::MessageFieldComparison
    ProtoFieldComparison;

// How to compare two floating-points (exact vs. approximate).
typedef google::protobuf::util::DefaultFieldComparator::FloatComparison
    ProtoFloatComparison;

// How to compare repeated fields (whether the order of elements matters).
typedef google::protobuf::util::MessageDifferencer::RepeatedFieldComparison
    RepeatedFieldComparison;

// Whether to compare all fields (full) or only fields present in the
// expected protobuf (partial).
typedef google::protobuf::util::MessageDifferencer::Scope ProtoComparisonScope;

const ProtoFieldComparison kProtoEqual =
    google::protobuf::util::MessageDifferencer::EQUAL;
const ProtoFieldComparison kProtoEquiv =
    google::protobuf::util::MessageDifferencer::EQUIVALENT;
const ProtoFloatComparison kProtoExact =
    google::protobuf::util::DefaultFieldComparator::EXACT;
const ProtoFloatComparison kProtoApproximate =
    google::protobuf::util::DefaultFieldComparator::APPROXIMATE;
const RepeatedFieldComparison kProtoCompareRepeatedFieldsRespectOrdering =
    google::protobuf::util::MessageDifferencer::AS_LIST;
const RepeatedFieldComparison kProtoCompareRepeatedFieldsIgnoringOrdering =
    google::protobuf::util::MessageDifferencer::AS_SET;
const ProtoComparisonScope kProtoFull = google::protobuf::util::MessageDifferencer::FULL;
const ProtoComparisonScope kProtoPartial =
    google::protobuf::util::MessageDifferencer::PARTIAL;

// Options for comparing two protobufs.
struct ProtoComparison {
  ProtoComparison()
      : field_comp(kProtoEqual),
        float_comp(kProtoExact),
        treating_nan_as_equal(false),
        has_custom_margin(false),
        has_custom_fraction(false),
        repeated_field_comp(kProtoCompareRepeatedFieldsRespectOrdering),
        scope(kProtoFull),
        float_margin(0.0),
        float_fraction(0.0) {}

  ProtoFieldComparison field_comp;
  ProtoFloatComparison float_comp;
  bool treating_nan_as_equal;
  bool has_custom_margin;    // only used when float_comp = APPROXIMATE
  bool has_custom_fraction;  // only used when float_comp = APPROXIMATE
  RepeatedFieldComparison repeated_field_comp;
  ProtoComparisonScope scope;
  double float_margin;       // only used when has_custom_margin is set.
  double float_fraction;     // only used when has_custom_fraction is set.
  std::vector<std::string> ignore_fields;
  std::vector<std::string> ignore_field_paths;
};

// Whether the protobuf must be initialized.
const bool kMustBeInitialized = true;
const bool kMayBeUninitialized = false;

// Parses the TextFormat representation of a protobuf, allowing required fields
// to be missing.  Returns true iff successful.
bool ParsePartialFromAscii(absl::string_view pb_ascii, google::protobuf::Message* proto,
                           std::string* error_text);

// Returns a protobuf of type Proto by parsing the given TextFormat
// representation of it.  Required fields can be missing, in which case the
// returned protobuf will not be fully initialized.
template <class Proto>
Proto MakePartialProtoFromAscii(absl::string_view str) {
  Proto proto;
  std::string error_text;
  CHECK(ParsePartialFromAscii(str, &proto, &error_text))
      << "Failed to parse \"" << str << "\" as a "
      << proto.GetDescriptor()->full_name() << ":\n" << error_text;
  return proto;
}

// Returns true iff p and q can be compared. Lite protos must share the same
// type name and full protos must share the same descriptor.
bool ProtoComparable(const google::protobuf::MessageLite& p,
                     const google::protobuf::MessageLite& q);
bool ProtoComparable(const google::protobuf::Message& p, const google::protobuf::Message& q);

// Returns true iff actual and expected are comparable and match.  The
// comp argument specifies how the two are compared.
bool ProtoCompare(const ProtoComparison& comp,
                  const google::protobuf::MessageLite& actual,
                  const google::protobuf::MessageLite& expected);
bool ProtoCompare(const ProtoComparison& comp, const google::protobuf::Message& actual,
                  const google::protobuf::Message& expected);

// Overload for ProtoCompare where the expected message is specified as a text
// proto.  If the text cannot be parsed as a message of the same type as the
// actual message, a CHECK failure will cause the test to fail and no subsequent
// tests will be run.
template <typename Proto>
inline bool ProtoCompare(const ProtoComparison& comp, const Proto& actual,
                         absl::string_view expected) {
  return ProtoCompare(comp, actual, MakePartialProtoFromAscii<Proto>(expected));
}

// Describes the types of the expected and the actual protocol buffer.
std::string DescribeTypes(const google::protobuf::MessageLite& expected,
                          const google::protobuf::MessageLite& actual);

// Prints the protocol buffer pointed to by proto.
template <class MessageType>
std::string PrintProtoPointee(const MessageType* proto) {
  if (proto == nullptr) return "";

  return "which points to " + ::testing::PrintToString(*proto);
}

// Describes the differences between the two protocol buffers.
std::string DescribeDiff(const ProtoComparison& comp,
                         const google::protobuf::MessageLite& actual,
                         const google::protobuf::MessageLite& expected);
std::string DescribeDiff(const ProtoComparison& comp,
                         const google::protobuf::Message& actual,
                         const google::protobuf::Message& expected);

// Common code for implementing EqualsProto, EquivToProto,
// EqualsInitializedProto, and EquivToInitializedProto.
template <class MessageType>
class ProtoMatcherBaseImpl {
 public:
  ProtoMatcherBaseImpl(
      bool must_be_initialized,     // Must the argument be fully initialized?
      const ProtoComparison& comp)  // How to compare the two protobufs.
      : must_be_initialized_(must_be_initialized), comp_(new auto(comp)) {}

  ProtoMatcherBaseImpl(const ProtoMatcherBaseImpl& other)
      : must_be_initialized_(other.must_be_initialized_),
        comp_(new auto(*other.comp_)) {}

  ProtoMatcherBaseImpl(ProtoMatcherBaseImpl&& other) noexcept = default;

  virtual ~ProtoMatcherBaseImpl() {}

  // Prints the expected protocol buffer.
  virtual void PrintExpectedTo(::std::ostream* os) const = 0;

  // Returns the expected value as a protobuf object; if the object
  // cannot be created (e.g. in ProtoStringMatcher), explains why to
  // 'listener' and returns NULL.  The caller must call
  // DeleteExpectedProto() on the returned value later.
  virtual const MessageType* CreateExpectedProto(
      const google::protobuf::MessageLite& arg,  // For determining the type of the
                                       // expected protobuf.
      ::testing::MatchResultListener* listener) const = 0;

  // Deletes the given expected protobuf, which must be obtained from
  // a call to CreateExpectedProto() earlier.
  virtual void DeleteExpectedProto(
      const google::protobuf::MessageLite* expected) const = 0;

  // Makes this matcher compare floating-points approximately.
  void SetCompareApproximately() { comp_->float_comp = kProtoApproximate; }

  // Makes this matcher treating NaNs as equal when comparing floating-points.
  void SetCompareTreatingNaNsAsEqual() { comp_->treating_nan_as_equal = true; }

  // Makes this matcher ignore string elements specified by their fully
  // qualified names, i.e., names corresponding to FieldDescriptor.full_name().
  template <class Iterator>
  void AddCompareIgnoringFields(Iterator first, Iterator last) {
    comp_->ignore_fields.insert(comp_->ignore_fields.end(), first, last);
  }

  // Makes this matcher ignore string elements specified by their relative
  // FieldPath.
  template <class Iterator>
  void AddCompareIgnoringFieldPaths(Iterator first, Iterator last) {
    comp_->ignore_field_paths.insert(comp_->ignore_field_paths.end(), first,
                                     last);
  }

  // Makes this matcher compare repeated fields ignoring ordering of elements.
  void SetCompareRepeatedFieldsIgnoringOrdering() {
    comp_->repeated_field_comp = kProtoCompareRepeatedFieldsIgnoringOrdering;
  }

  // Sets the margin of error for approximate floating point comparison.
  void SetMargin(double margin) {
    CHECK_GE(margin, 0.0) << "Using a negative margin for Approximately";
    comp_->has_custom_margin = true;
    comp_->float_margin = margin;
  }

  // Sets the relative fraction of error for approximate floating point
  // comparison.
  void SetFraction(double fraction) {
    CHECK(0.0 <= fraction && fraction < 1.0) <<
        "Fraction for Approximately must be >= 0.0 and < 1.0";
    comp_->has_custom_fraction = true;
    comp_->float_fraction = fraction;
  }

  // Makes this matcher compare protobufs partially.
  void SetComparePartially() { comp_->scope = kProtoPartial; }

  bool MatchAndExplain(const google::protobuf::MessageLite& arg,
                       ::testing::MatchResultListener* listener) const {
    return MatchAndExplain(arg, false, listener);
  }

  bool MatchAndExplain(const google::protobuf::MessageLite* arg,
                       ::testing::MatchResultListener* listener) const {
    return (arg != nullptr) && MatchAndExplain(*arg, true, listener);
  }

  // Describes the expected relation between the actual protobuf and
  // the expected one.
  void DescribeRelationToExpectedProto(::std::ostream* os) const {
    if (comp_->repeated_field_comp ==
        kProtoCompareRepeatedFieldsIgnoringOrdering) {
      *os << "(ignoring repeated field ordering) ";
    }
    if (!comp_->ignore_fields.empty()) {
      *os << "(ignoring fields: ";
      absl::string_view sep = "";
      for (size_t i = 0; i < comp_->ignore_fields.size(); ++i, sep = ", ")
        *os << sep << comp_->ignore_fields[i];
      *os << ") ";
    }
    if (comp_->float_comp == kProtoApproximate) {
      *os << "approximately ";
      if (comp_->has_custom_margin || comp_->has_custom_fraction) {
        *os << "(";
        if (comp_->has_custom_margin) {
          std::stringstream ss;
          ss << std::setprecision(std::numeric_limits<double>::digits10 + 2)
             << comp_->float_margin;
          *os << "absolute error of float or double fields <= " << ss.str();
        }
        if (comp_->has_custom_margin && comp_->has_custom_fraction) {
          *os << " or ";
        }
        if (comp_->has_custom_fraction) {
          std::stringstream ss;
          ss << std::setprecision(std::numeric_limits<double>::digits10 + 2)
             << comp_->float_fraction;
          *os << "relative error of float or double fields <= " << ss.str();
        }
        *os << ") ";
      }
    }

    *os << (comp_->scope == kProtoPartial ? "partially " : "")
        << (comp_->field_comp == kProtoEqual ? "equal" : "equivalent")
        << (comp_->treating_nan_as_equal ? " (treating NaNs as equal)" : "")
        << " to ";
    PrintExpectedTo(os);
  }

  void DescribeTo(::std::ostream* os) const {
    *os << "is " << (must_be_initialized_ ? "fully initialized and " : "");
    DescribeRelationToExpectedProto(os);
  }

  void DescribeNegationTo(::std::ostream* os) const {
    *os << "is " << (must_be_initialized_ ? "not fully initialized or " : "")
        << "not ";
    DescribeRelationToExpectedProto(os);
  }

  bool must_be_initialized() const { return must_be_initialized_; }

  const ProtoComparison& comp() const { return *comp_; }

 private:
  bool MatchAndExplain(const google::protobuf::MessageLite& arg,
                       bool is_matcher_for_pointer,
                       ::testing::MatchResultListener* listener) const;

  const bool must_be_initialized_;
  std::unique_ptr<ProtoComparison> comp_;
};

template <class MessageType>
bool ProtoMatcherBaseImpl<MessageType>::MatchAndExplain(
    const google::protobuf::MessageLite& arg,
    bool is_matcher_for_pointer,  // true iff this matcher is used to match
                                  // a protobuf pointer.
    ::testing::MatchResultListener* listener) const {
  const auto& casted_arg = dynamic_cast<const MessageType&>(arg);
  if (must_be_initialized_ && !casted_arg.IsInitialized()) {
    *listener << "which isn't fully initialized";
    return false;
  }

  const MessageType* const expected = CreateExpectedProto(casted_arg, listener);
  if (expected == nullptr) return false;

  // Protobufs of different types cannot be compared.
  const bool comparable = ProtoComparable(casted_arg, *expected);
  const bool match = comparable && ProtoCompare(comp(), casted_arg, *expected);

  // Explaining the match result is expensive.  We don't want to waste
  // time calculating an explanation if the listener isn't interested.
  if (listener->IsInterested()) {
    absl::string_view sep = "";
    if (is_matcher_for_pointer) {
      *listener << PrintProtoPointee(&casted_arg);
      sep = ",\n";
    }

    if (!comparable) {
      *listener << sep << DescribeTypes(*expected, casted_arg);
    } else if (!match) {
      *listener << sep << DescribeDiff(comp(), casted_arg, *expected);
    }
  }

  DeleteExpectedProto(expected);
  return match;
}

// ProtoMatcherBase must be defined here with a 'using' statement, because there
// is code in google3 which assumes that all proto matchers derive from a base
// class called ProtoMatcherBase.
//  ProtoMatcherBaseImpl<GeneratedMessageType> (aka ProtoMatcherBase)
//   |
//  ProtoMatcherImpl<GeneratedMessageType>
//   |
//  ProtoMatcher
using ProtoMatcherBase = ProtoMatcherBaseImpl<google::protobuf::GeneratedMessageBaseType>;

// Returns a copy of the given proto2 message.
template <class MessageType>
inline MessageType* CloneProto2(const MessageType& src) {
  MessageType* clone = src.New();
  clone->CheckTypeAndMergeFrom(src);
  return clone;
}

// Implements EqualsProto, EquivToProto, EqualsInitializedProto, and
// EquivToInitializedProto, where the matcher parameter is a protobuf.
template <class MessageType>
class ProtoMatcherImpl : public ProtoMatcherBaseImpl<MessageType> {
 public:
  using Proto = MessageType;

  ProtoMatcherImpl(
      const MessageType& expected,  // The expected protobuf.
      bool must_be_initialized,     // Must the argument be fully initialized?
      const ProtoComparison& comp)  // How to compare the two protobufs.
      : ProtoMatcherBaseImpl<MessageType>(must_be_initialized, comp),
        expected_(CloneProto2(expected)) {
    if (must_be_initialized) {
      CHECK(expected.IsInitialized())
          << "The protocol buffer given to *InitializedProto() "
          << "must itself be initialized, but the following required fields "
          << "are missing: " << expected.InitializationErrorString() << ".";
    }
  }

  void PrintExpectedTo(::std::ostream* os) const override {
    *os << expected_->GetTypeName() << " ";
    ::testing::internal::UniversalPrint(*expected_, os);
  }

  const MessageType* CreateExpectedProto(
      const google::protobuf::MessageLite& /* arg */,
      ::testing::MatchResultListener* /* listener */) const override {
    return expected_.get();
  }

  void DeleteExpectedProto(
      const google::protobuf::MessageLite* /* expected */) const override {}

  const std::shared_ptr<const MessageType>& expected() const {
    return expected_;
  }

 private:
  const std::shared_ptr<const MessageType> expected_;
};

// ProtoMatcher must be defined as a class (no using statement here) for
// backwards compatibility with inherited classes. When defined with using
// statement, the inherited class must use a fully qualified class name with the
// template type to call the base class ctor - it seems that C++ does not (?)
// infer the base class type in ctor from the inheritenace (weird, but true).
class ProtoMatcher : public ProtoMatcherImpl<google::protobuf::GeneratedMessageBaseType> {
 public:
  ProtoMatcher(
      const google::protobuf::GeneratedMessageBaseType&
          expected,                 // The expected protobuf.
      bool must_be_initialized,     // Must the argument be fully initialized?
      const ProtoComparison& comp)  // How to compare the two protobufs.
      : ProtoMatcherImpl(expected, must_be_initialized, comp) {}
};

class ProtoMatcherLite : public ProtoMatcherImpl<google::protobuf::MessageLite> {
 public:
  ProtoMatcherLite(
      const google::protobuf::MessageLite& expected,  // The expected protobuf.
      bool must_be_initialized,     // Must the argument be fully initialized?
      const ProtoComparison& comp)  // How to compare the two protobufs.
      : ProtoMatcherImpl(expected, must_be_initialized, comp) {}
};
// Implements EqualsProto, EquivToProto, EqualsInitializedProto, and
// EquivToInitializedProto, where the matcher parameter is a string.
class ProtoStringMatcher : public ProtoMatcherBase {
 public:
  ProtoStringMatcher(
      absl::string_view expected,  // The text for the expected protobuf.
      bool must_be_initialized,    // Must the argument be fully initialized?
      const ProtoComparison comp)  // How to compare the two protobufs.
      : ProtoMatcherBase(must_be_initialized, comp), expected_(expected) {}

  // Parses the expected string as a protobuf of the same type as arg,
  // and returns the parsed protobuf (or NULL when the parse fails).
  // The caller must call DeleteExpectedProto() on the return value
  // later.
  const google::protobuf::Message* CreateExpectedProto(
      const google::protobuf::MessageLite& arg,
      ::testing::MatchResultListener* listener) const override {
    auto* expected_proto = dynamic_cast<google::protobuf::Message*>(arg.New());
    // We don't insist that the expected string parses as an
    // *initialized* protobuf.  Otherwise EqualsProto("...") may
    // wrongfully fail when the actual protobuf is not fully
    // initialized.  If the user wants to ensure that the actual
    // protobuf is initialized, they should use
    // EqualsInitializedProto("...") instead of EqualsProto("..."),
    // and the MatchAndExplain() function in ProtoMatcherBase will
    // enforce it.
    std::string error_text;
    if (ParsePartialFromAscii(expected_, expected_proto, &error_text)) {
      return expected_proto;
    } else {
      delete expected_proto;
      if (listener->IsInterested()) {
        *listener << "where ";
        PrintExpectedTo(listener->stream());
        *listener << " doesn't parse as a " << arg.GetTypeName() << ":\n"
                  << error_text;
      }
      return nullptr;
    }
  }

  void DeleteExpectedProto(const google::protobuf::MessageLite* expected) const override {
    delete expected;
  }

  void PrintExpectedTo(::std::ostream* os) const override {
    *os << "<" << expected_ << ">";
  }

 private:
  const std::string expected_;
};

using PolymorphicProtoMatcher = ::testing::PolymorphicMatcher<ProtoMatcher>;
using PolymorphicProtoMatcherLite =
    ::testing::PolymorphicMatcher<ProtoMatcherLite>;

// Enum representing the possible representations of protos as strings. Used
// to configure the adapter's behavior when converting the input string.
enum class ProtoStringFormat { kBinaryFormat, kTextFormat };

// Common code for implementing adapters that allow proto matchers to match
// strings (WhenDeserialized/WhenParsedFromProtoText).
template <class Proto>
class ProtoMatcherStringAdapter {
 protected:
  using InnerMatcher = ::testing::Matcher<const Proto&>;

  ProtoMatcherStringAdapter(const InnerMatcher& proto_matcher,
                            const ProtoStringFormat input_format)
      : proto_matcher_(proto_matcher), input_format_(input_format) {}

  ~ProtoMatcherStringAdapter() {}

 private:
  // Creates an empty protobuf with the expected type.
  virtual Proto* MakeEmptyProto() const = 0;

  // Type name of the expected protobuf.
  virtual std::string ExpectedTypeName() const = 0;

  // Name of the type argument given to
  // WhenDeserializedAs<>()/WhenParsedFromProtoTextAs<>(), or "protobuf" for
  // WhenDeserialized()/WhenParsedFromProtoText().
  virtual std::string TypeArgName() const = 0;

  // Deserializes or parses the string as a protobuf of the same type as the
  // expected protobuf.
  ::std::unique_ptr<Proto> ToProto(
      google::protobuf::io::ZeroCopyInputStream* input) const {
    auto proto = absl::WrapUnique(this->MakeEmptyProto());
    bool converted = false;
    switch (input_format_) {
      case ProtoStringFormat::kBinaryFormat:
        // ParsePartialFromString() parses a serialized representation of a
        // protobuf, allowing required fields to be missing.  This means
        // that we don't insist on the parsed protobuf being fully
        // initialized.  This allows the user to choose whether it should
        // be initialized using EqualsProto vs EqualsInitializedProto, for
        // example.
        converted = proto->ParsePartialFromZeroCopyStream(input);
        break;
      case ProtoStringFormat::kTextFormat: {
        google::protobuf::TextFormat::Parser parser;
        parser.AllowPartialMessage(true);
        converted = parser.Parse(input, proto.get());
        break;
      }
    }
    return converted ? std::move(proto) : nullptr;
  }

  // Gets the past tense of the verb describing the conversion. Used for
  // describing the matcher and explaining matches.
  std::string ConversionVerbPast() const {
    switch (input_format_) {
      case ProtoStringFormat::kBinaryFormat:
        return "deserialized";
      case ProtoStringFormat::kTextFormat:
        return "parsed";
    }
  }

  // Gets the present tense of the verb describing the conversion. Used for
  // describing the matcher and explaining matches.
  std::string ConversionVerbPresent() const {
    switch (input_format_) {
      case ProtoStringFormat::kBinaryFormat:
        return "deserializes";
      case ProtoStringFormat::kTextFormat:
        return "parses";
    }
  }

 public:
  void DescribeTo(::std::ostream* os) const {
    *os << "can be " << ConversionVerbPast() << " as a " << TypeArgName()
        << " that ";
    proto_matcher_.DescribeTo(os);
  }

  void DescribeNegationTo(::std::ostream* os) const {
    *os << "cannot be " << ConversionVerbPast() << " as a " << TypeArgName()
        << " that ";
    proto_matcher_.DescribeTo(os);
  }

  bool MatchAndExplain(google::protobuf::io::ZeroCopyInputStream* arg,
                       ::testing::MatchResultListener* listener) const {
    // Deserializes the string arg as a protobuf of the same type as the
    // expected protobuf.
    ::std::unique_ptr<const Proto> proto_arg = ToProto(arg);
    if (!listener->IsInterested()) {
      // No need to explain the match result.
      return (proto_arg != nullptr) && proto_matcher_.Matches(*proto_arg);
    }

    ::std::ostream* const os = listener->stream();
    if (proto_arg == nullptr) {
      *os << "which cannot be " << ConversionVerbPast() << " as a "
          << ExpectedTypeName();
      return false;
    }

    *os << "which " << ConversionVerbPresent() << " to ";
    ::testing::internal::UniversalPrint(*proto_arg, os);

    ::testing::StringMatchResultListener inner_listener;
    const bool match =
        proto_matcher_.MatchAndExplain(*proto_arg, &inner_listener);
    const std::string explain = inner_listener.str();
    if (!explain.empty()) {
      *os << ",\n" << explain;
    }

    return match;
  }

  bool MatchAndExplain(const absl::Cord& cord,
                       ::testing::MatchResultListener* listener) const {
    // TODO: it is better to use google::protobuf::io::CordInputStream, this is
    // not available in open source proto yet.
    return MatchAndExplain(std::string(cord), listener);
  }

  bool MatchAndExplain(absl::string_view sp,
                       ::testing::MatchResultListener* listener) const {
    google::protobuf::io::ArrayInputStream input(sp.data(), sp.size());
    return MatchAndExplain(&input, listener);
  }

 private:
  const InnerMatcher proto_matcher_;
  const ProtoStringFormat input_format_;
};

// Implements WhenDeserialized(proto_matcher) and
// WhenParsedFromProtoText(proto_matcher).
class UntypedProtoMatcherStringAdapter final
    : public ProtoMatcherStringAdapter<google::protobuf::GeneratedMessageBaseType> {
 public:
  UntypedProtoMatcherStringAdapter(const PolymorphicProtoMatcher& proto_matcher,
                                   const ProtoStringFormat input_format)
      : ProtoMatcherStringAdapter<google::protobuf::GeneratedMessageBaseType>(
            proto_matcher, input_format),
        expected_proto_(proto_matcher.impl().expected()) {}

 private:
  google::protobuf::GeneratedMessageBaseType* MakeEmptyProto() const override {
    return expected_proto_->New();
  }

  std::string ExpectedTypeName() const override {
    return expected_proto_->GetTypeName();
  }

  std::string TypeArgName() const override { return "protobuf"; }

  // The expected protobuf specified in the inner matcher
  // (proto_matcher_).  We only need a std::shared_ptr to it instead of
  // making a copy, as the expected protobuf will never be changed
  // once created.
  const std::shared_ptr<const google::protobuf::GeneratedMessageBaseType> expected_proto_;
};

// Implements WhenDeserializedAs<Proto>(proto_matcher) and
// WhenParsedFromProtoTextAs<Proto>(proto_matcher).
template <class Proto>
class TypedProtoMatcherStringAdapter final
    : public ProtoMatcherStringAdapter<Proto> {
 private:
  using InnerMatcher = ::testing::Matcher<const Proto&>;

 public:
  TypedProtoMatcherStringAdapter(const InnerMatcher& inner_matcher,
                                 const ProtoStringFormat input_format)
      : ProtoMatcherStringAdapter<Proto>(inner_matcher, input_format) {}

 private:
  Proto* MakeEmptyProto() const override { return new Proto; }

  std::string ExpectedTypeName() const override {
    return Proto().GetTypeName();
  }

  std::string TypeArgName() const override { return ExpectedTypeName(); }
};

// Implements the IsInitializedProto matcher, which is used to verify that a
// protocol buffer is valid using the IsInitialized method.
class IsInitializedProtoMatcher {
 public:
  void DescribeTo(::std::ostream* os) const {
    *os << "is a fully initialized protocol buffer";
  }

  void DescribeNegationTo(::std::ostream* os) const {
    *os << "is not a fully initialized protocol buffer";
  }

  template <typename T>
  bool MatchAndExplain(T& arg,  // NOLINT
                       ::testing::MatchResultListener* listener) const {
    if (!arg.IsInitialized()) {
      *listener << "which is missing the following required fields: "
                << arg.InitializationErrorString();
      return false;
    }
    return true;
  }

  // It's critical for this overload to take a T* instead of a const
  // T*.  Otherwise the other version would be a better match when arg
  // is a pointer to a non-const value.
  template <typename T>
  bool MatchAndExplain(T* arg, ::testing::MatchResultListener* listener) const {
    if (listener->IsInterested() && arg != nullptr) {
      *listener << PrintProtoPointee(arg);
    }
    if (arg == nullptr) {
      *listener << "which is null";
      return false;
    } else if (!arg->IsInitialized()) {
      *listener << ", which is missing the following required fields: "
                << arg->InitializationErrorString();
      return false;
    } else {
      return true;
    }
  }
};

// Implements EqualsProto and EquivToProto for 2-tuple matchers.
class TupleProtoMatcher {
 public:
  explicit TupleProtoMatcher(const ProtoComparison& comp)
      : comp_(new auto(comp)) {}

  TupleProtoMatcher(const TupleProtoMatcher& other)
      : comp_(new auto(*other.comp_)) {}
  TupleProtoMatcher(TupleProtoMatcher&& other) = default;

  template <typename T1, typename T2>
  operator ::testing::Matcher<::testing::tuple<T1, T2>>() const {
    return MakeMatcher(new Impl< ::testing::tuple<T1, T2> >(*comp_));
  }
  template <typename T1, typename T2>
  operator ::testing::Matcher<const ::testing::tuple<T1, T2>&>() const {
    return MakeMatcher(new Impl<const ::testing::tuple<T1, T2>&>(*comp_));
  }

  // Allows matcher transformers, e.g., Approximately(), Partially(), etc. to
  // change the behavior of this 2-tuple matcher.
  TupleProtoMatcher& mutable_impl() { return *this; }

  // Makes this matcher compare floating-points approximately.
  void SetCompareApproximately() { comp_->float_comp = kProtoApproximate; }

  // Makes this matcher treating NaNs as equal when comparing floating-points.
  void SetCompareTreatingNaNsAsEqual() { comp_->treating_nan_as_equal = true; }

  // Makes this matcher ignore string elements specified by their fully
  // qualified names, i.e., names corresponding to FieldDescriptor.full_name().
  template <class Iterator>
  void AddCompareIgnoringFields(Iterator first, Iterator last) {
    comp_->ignore_fields.insert(comp_->ignore_fields.end(), first, last);
  }

  // Makes this matcher ignore string elements specified by their relative
  // FieldPath.
  template <class Iterator>
  void AddCompareIgnoringFieldPaths(Iterator first, Iterator last) {
    comp_->ignore_field_paths.insert(comp_->ignore_field_paths.end(), first,
                                     last);
  }

  // Makes this matcher compare repeated fields ignoring ordering of elements.
  void SetCompareRepeatedFieldsIgnoringOrdering() {
    comp_->repeated_field_comp = kProtoCompareRepeatedFieldsIgnoringOrdering;
  }

  // Sets the margin of error for approximate floating point comparison.
  void SetMargin(double margin) {
    CHECK_GE(margin, 0.0) << "Using a negative margin for Approximately";
    comp_->has_custom_margin = true;
    comp_->float_margin = margin;
  }

  // Sets the relative fraction of error for approximate floating point
  // comparison.
  void SetFraction(double fraction) {
    CHECK(0.0 <= fraction && fraction <= 1.0) <<
        "Fraction for Relatively must be >= 0.0 and < 1.0";
    comp_->has_custom_fraction = true;
    comp_->float_fraction = fraction;
  }

  // Makes this matcher compares protobufs partially.
  void SetComparePartially() { comp_->scope = kProtoPartial; }

 private:
  template <typename Tuple>
  class Impl : public ::testing::MatcherInterface<Tuple> {
   public:
    explicit Impl(const ProtoComparison& comp) : comp_(comp) {}
    bool MatchAndExplain(
        Tuple args,
        ::testing::MatchResultListener* /* listener */) const override {
      using ::testing::get;
      return ProtoCompare(comp_, get<0>(args), get<1>(args));
    }
    void DescribeTo(::std::ostream* os) const override {
      *os << (comp_.field_comp == kProtoEqual ? "are equal"
                                               : "are equivalent");
    }
    void DescribeNegationTo(::std::ostream* os) const override {
      *os << (comp_.field_comp == kProtoEqual ? "are not equal"
                                               : "are not equivalent");
    }

   private:
    const ProtoComparison comp_;
  };

  std::unique_ptr<ProtoComparison> comp_;
};

}  // namespace internal

// Creates a polymorphic matcher that matches a 2-tuple where
// first.Equals(second) is true.
inline internal::TupleProtoMatcher EqualsProto() {
  internal::ProtoComparison comp;
  comp.field_comp = internal::kProtoEqual;
  return internal::TupleProtoMatcher(comp);
}

// Creates a polymorphic matcher that matches a 2-tuple where
// first.Equivalent(second) is true.
inline internal::TupleProtoMatcher EquivToProto() {
  internal::ProtoComparison comp;
  comp.field_comp = internal::kProtoEquiv;
  return internal::TupleProtoMatcher(comp);
}

// Constructs a matcher that matches the argument if
// argument.Equals(x) or argument->Equals(x) returns true.
inline internal::PolymorphicProtoMatcher EqualsProto(const google::protobuf::Message& x) {
  internal::ProtoComparison comp;
  comp.field_comp = internal::kProtoEqual;
  return ::testing::MakePolymorphicMatcher(
      internal::ProtoMatcher(x, internal::kMayBeUninitialized, comp));
}
inline internal::PolymorphicProtoMatcherLite EqualsProto(
    const google::protobuf::MessageLite& x) {
  internal::ProtoComparison comp;
  comp.field_comp = internal::kProtoEqual;
  return ::testing::MakePolymorphicMatcher(
      internal::ProtoMatcherLite(x, internal::kMayBeUninitialized, comp));
}
inline ::testing::PolymorphicMatcher<internal::ProtoStringMatcher> EqualsProto(
    absl::string_view x) {
  internal::ProtoComparison comp;
  comp.field_comp = internal::kProtoEqual;
  return ::testing::MakePolymorphicMatcher(
      internal::ProtoStringMatcher(x, internal::kMayBeUninitialized, comp));
}
template <class Proto>
inline internal::PolymorphicProtoMatcher EqualsProto(absl::string_view str) {
  return EqualsProto(internal::MakePartialProtoFromAscii<Proto>(str));
}

// Constructs a matcher that matches the argument if
// argument.Equivalent(x) or argument->Equivalent(x) returns true.
inline internal::PolymorphicProtoMatcher EquivToProto(
    const google::protobuf::Message& x) {
  internal::ProtoComparison comp;
  comp.field_comp = internal::kProtoEquiv;
  return ::testing::MakePolymorphicMatcher(
      internal::ProtoMatcher(x, internal::kMayBeUninitialized, comp));
}
inline ::testing::PolymorphicMatcher<internal::ProtoStringMatcher> EquivToProto(
    absl::string_view x) {
  internal::ProtoComparison comp;
  comp.field_comp = internal::kProtoEquiv;
  return ::testing::MakePolymorphicMatcher(
      internal::ProtoStringMatcher(x, internal::kMayBeUninitialized, comp));
}
template <class Proto>
inline internal::PolymorphicProtoMatcher EquivToProto(absl::string_view str) {
  return EquivToProto(internal::MakePartialProtoFromAscii<Proto>(str));
}

// Constructs a matcher that matches the argument if
// argument.IsInitialized() or argument->IsInitialized() returns true.
inline ::testing::PolymorphicMatcher<internal::IsInitializedProtoMatcher>
IsInitializedProto() {
  return ::testing::MakePolymorphicMatcher(
      internal::IsInitializedProtoMatcher());
}

// Constructs a matcher that matches an argument whose IsInitialized()
// and Equals(x) methods both return true.  The argument can be either
// a protocol buffer or a pointer to it.
inline internal::PolymorphicProtoMatcher EqualsInitializedProto(
    const google::protobuf::Message& x) {
  internal::ProtoComparison comp;
  comp.field_comp = internal::kProtoEqual;
  return ::testing::MakePolymorphicMatcher(
      internal::ProtoMatcher(x, internal::kMustBeInitialized, comp));
}
inline internal::PolymorphicProtoMatcherLite EqualsInitializedProto(
    const google::protobuf::MessageLite& x) {
  internal::ProtoComparison comp;
  comp.field_comp = internal::kProtoEqual;
  return ::testing::MakePolymorphicMatcher(
      internal::ProtoMatcherLite(x, internal::kMustBeInitialized, comp));
}
inline ::testing::PolymorphicMatcher<internal::ProtoStringMatcher>
EqualsInitializedProto(absl::string_view x) {
  internal::ProtoComparison comp;
  comp.field_comp = internal::kProtoEqual;
  return ::testing::MakePolymorphicMatcher(
      internal::ProtoStringMatcher(x, internal::kMustBeInitialized, comp));
}
template <class Proto>
inline internal::PolymorphicProtoMatcher EqualsInitializedProto(
    absl::string_view str) {
  return EqualsInitializedProto(
      internal::MakePartialProtoFromAscii<Proto>(str));
}

// Constructs a matcher that matches an argument whose IsInitialized()
// and Equivalent(x) methods both return true.  The argument can be
// either a protocol buffer or a pointer to it.
inline internal::PolymorphicProtoMatcher
EquivToInitializedProto(const google::protobuf::Message& x) {
  internal::ProtoComparison comp;
  comp.field_comp = internal::kProtoEquiv;
  return ::testing::MakePolymorphicMatcher(
      internal::ProtoMatcher(x, internal::kMustBeInitialized, comp));
}
inline ::testing::PolymorphicMatcher<internal::ProtoStringMatcher>
EquivToInitializedProto(absl::string_view x) {
  internal::ProtoComparison comp;
  comp.field_comp = internal::kProtoEquiv;
  return ::testing::MakePolymorphicMatcher(
      internal::ProtoStringMatcher(x, internal::kMustBeInitialized, comp));
}
template <class Proto>
inline internal::PolymorphicProtoMatcher EquivToInitializedProto(
    absl::string_view str) {
  return EquivToInitializedProto(
      internal::MakePartialProtoFromAscii<Proto>(str));
}

namespace proto {

// Approximately(m) returns a matcher that is the same as m, except
// that it compares floating-point fields approximately (using
// google::protobuf::util::MessageDifferencer's APPROXIMATE comparison option).
// The inner matcher m can be any of the Equals* and EquivTo* protobuf
// matchers above.
template <class InnerProtoMatcher>
inline InnerProtoMatcher Approximately(InnerProtoMatcher inner_proto_matcher) {
  static_assert(sizeof(InnerProtoMatcher) != 0 &&
                    std::is_same<google::protobuf::GeneratedMessageBaseType,
                                 google::protobuf::Message>::value,
                "The Approximately() matcher requires full (non-lite) protos.");
  inner_proto_matcher.mutable_impl().SetCompareApproximately();
  return inner_proto_matcher;
}

// Alternative version of Approximately which takes an explicit margin of error.
template <class InnerProtoMatcher>
inline InnerProtoMatcher Approximately(InnerProtoMatcher inner_proto_matcher,
                                       double margin) {
  static_assert(sizeof(InnerProtoMatcher) != 0 &&
                    std::is_same<google::protobuf::GeneratedMessageBaseType,
                                 google::protobuf::Message>::value,
                "The Approximately() matcher requires full (non-lite) protos.");
  inner_proto_matcher.mutable_impl().SetCompareApproximately();
  inner_proto_matcher.mutable_impl().SetMargin(margin);
  return inner_proto_matcher;
}

// Alternative version of Approximately which takes an explicit margin of error
// and a relative fraction of error and will match if either is satisfied.
template <class InnerProtoMatcher>
inline InnerProtoMatcher Approximately(InnerProtoMatcher inner_proto_matcher,
                                       double margin, double fraction) {
  static_assert(sizeof(InnerProtoMatcher) != 0 &&
                    std::is_same<google::protobuf::GeneratedMessageBaseType,
                                 google::protobuf::Message>::value,
                "The Approximately() matcher requires full (non-lite) protos.");
  inner_proto_matcher.mutable_impl().SetCompareApproximately();
  inner_proto_matcher.mutable_impl().SetMargin(margin);
  inner_proto_matcher.mutable_impl().SetFraction(fraction);
  return inner_proto_matcher;
}

// TreatingNaNsAsEqual(m) returns a matcher that is the same as m, except that
// it compares floating-point fields such that NaNs are equal.
// The inner matcher m can be any of the Equals* and EquivTo* protobuf matchers
// above.
template <class InnerProtoMatcher>
inline InnerProtoMatcher TreatingNaNsAsEqual(
    InnerProtoMatcher inner_proto_matcher) {
  static_assert(
      sizeof(InnerProtoMatcher) != 0 &&
          std::is_same<google::protobuf::GeneratedMessageBaseType,
                       google::protobuf::Message>::value,
      "The TreatingNaNsAsEqual() matcher requires full (non-lite) protos.");
  inner_proto_matcher.mutable_impl().SetCompareTreatingNaNsAsEqual();
  return inner_proto_matcher;
}

// IgnoringFields(fields, m) returns a matcher that is the same as m, except the
// specified fields will be ignored when matching
// (using google::protobuf::util::MessageDifferencer::IgnoreField). Each element in fields
// are specified by their fully qualified names, i.e., the names corresponding
// to FieldDescriptor.full_name(). (e.g. testing.internal.FooProto2.member).
// m can be any of the Equals* and EquivTo* protobuf matchers above.
// It can also be any of the transformer matchers listed here (e.g.
// Approximately, TreatingNaNsAsEqual) as long as the intent of the each
// concatenated matcher is mutually exclusive (e.g. using IgnoringFields in
// conjunction with Partially can have different results depending on whether
// the fields specified in IgnoringFields is part of the fields covered by
// Partially).
template <class InnerProtoMatcher, class Container>
inline InnerProtoMatcher IgnoringFields(const Container& ignore_fields,
                                        InnerProtoMatcher inner_proto_matcher) {
  static_assert(
      sizeof(InnerProtoMatcher) != 0 &&
          std::is_same<google::protobuf::GeneratedMessageBaseType,
                       google::protobuf::Message>::value,
      "The IgnoringFields() matcher requires full (non-lite) protos.");
  inner_proto_matcher.mutable_impl().AddCompareIgnoringFields(
      ignore_fields.begin(), ignore_fields.end());
  return inner_proto_matcher;
}

// See top comment.
template <class InnerProtoMatcher, class Container>
inline InnerProtoMatcher IgnoringFieldPaths(
    const Container& ignore_field_paths,
    InnerProtoMatcher inner_proto_matcher) {
  static_assert(
      sizeof(InnerProtoMatcher) != 0 &&
          std::is_same<google::protobuf::GeneratedMessageBaseType,
                       google::protobuf::Message>::value,
      "The IgnoringFieldPaths() matcher requires full (non-lite) protos.");
  inner_proto_matcher.mutable_impl().AddCompareIgnoringFieldPaths(
      ignore_field_paths.begin(), ignore_field_paths.end());
  return inner_proto_matcher;
}

#ifdef LANG_CXX11
template <class InnerProtoMatcher, class T>
inline InnerProtoMatcher IgnoringFields(std::initializer_list<T> il,
                                        InnerProtoMatcher inner_proto_matcher) {
  static_assert(
      sizeof(InnerProtoMatcher) != 0 &&
          std::is_same<google::protobuf::GeneratedMessageBaseType,
                       google::protobuf::Message>::value,
      "The IgnoringFields() matcher requires full (non-lite) protos.");
  inner_proto_matcher.mutable_impl().AddCompareIgnoringFields(
      il.begin(), il.end());
  return inner_proto_matcher;
}

template <class InnerProtoMatcher, class T>
inline InnerProtoMatcher IgnoringFieldPaths(
    std::initializer_list<T> il, InnerProtoMatcher inner_proto_matcher) {
  static_assert(
      sizeof(InnerProtoMatcher) != 0 &&
          std::is_same<google::protobuf::GeneratedMessageBaseType,
                       google::protobuf::Message>::value,
      "The IgnoringFieldPaths() matcher requires full (non-lite) protos.");
  inner_proto_matcher.mutable_impl().AddCompareIgnoringFieldPaths(il.begin(),
                                                                  il.end());
  return inner_proto_matcher;
}
#endif  // LANG_CXX11

// IgnoringRepeatedFieldOrdering(m) returns a matcher that is the same as m,
// except that it ignores the relative ordering of elements within each repeated
// field in m. See google::protobuf::MessageDifferencer::TreatAsSet() for more details.
template <class InnerProtoMatcher>
inline InnerProtoMatcher IgnoringRepeatedFieldOrdering(
    InnerProtoMatcher inner_proto_matcher) {
  static_assert(sizeof(InnerProtoMatcher) != 0 &&
                    std::is_same<google::protobuf::GeneratedMessageBaseType,
                                 google::protobuf::Message>::value,
                "The IgnoringRepeatedFieldOrdering() matcher requires full "
                "(non-lite) protos.");
  inner_proto_matcher.mutable_impl().SetCompareRepeatedFieldsIgnoringOrdering();
  return inner_proto_matcher;
}

// Partially(m) returns a matcher that is the same as m, except that
// only fields present in the expected protobuf are considered (using
// google::protobuf::util::MessageDifferencer's PARTIAL comparison option).  For
// example, Partially(EqualsProto(p)) will ignore any field that's
// not set in p when comparing the protobufs. Repeated fields are not treated
// specially; extra or missing values will cause the test to fail. The inner
// matcher m can be any of the Equals* and EquivTo* protobuf matchers above.
template <class InnerProtoMatcher>
inline InnerProtoMatcher Partially(InnerProtoMatcher inner_proto_matcher) {
  static_assert(sizeof(InnerProtoMatcher) != 0 &&
                    std::is_same<google::protobuf::GeneratedMessageBaseType,
                                 google::protobuf::Message>::value,
                "The Partially() matcher requires full (non-lite) protos.");
  inner_proto_matcher.mutable_impl().SetComparePartially();
  return inner_proto_matcher;
}

// WhenDeserialized(m) is a matcher that matches a string that can be
// deserialized as a protobuf that matches m.  m must be a protobuf
// matcher where the expected protobuf type is known at run time.
inline ::testing::PolymorphicMatcher<internal::UntypedProtoMatcherStringAdapter>
WhenDeserialized(const internal::PolymorphicProtoMatcher& proto_matcher) {
  return ::testing::MakePolymorphicMatcher(
      internal::UntypedProtoMatcherStringAdapter(
          proto_matcher, internal::ProtoStringFormat::kBinaryFormat));
}

// WhenDeserializedAs<Proto>(m) is a matcher that matches a string
// that can be deserialized as a protobuf of type Proto that matches
// m, which can be any valid protobuf matcher.
template <class Proto, class InnerMatcher>
::testing::PolymorphicMatcher<internal::TypedProtoMatcherStringAdapter<Proto>>
WhenDeserializedAs(const InnerMatcher& inner_matcher) {
  return ::testing::MakePolymorphicMatcher(
      internal::TypedProtoMatcherStringAdapter<Proto>(
          ::testing::SafeMatcherCast<const Proto&>(inner_matcher),
          internal::ProtoStringFormat::kBinaryFormat));
}

// WhenParsedFromProtoText(m) is a matcher that matches a string that can be
// parsed as a text-format protobuf that matches m. m must be a protobuf matcher
// where the expected protobuf type is known at run time.
inline ::testing::PolymorphicMatcher<internal::UntypedProtoMatcherStringAdapter>
WhenParsedFromProtoText(
    const internal::PolymorphicProtoMatcher& proto_matcher) {
  return ::testing::MakePolymorphicMatcher(
      internal::UntypedProtoMatcherStringAdapter(
          proto_matcher, internal::ProtoStringFormat::kTextFormat));
}

// WhenParsedFromProtoTextAs<Proto>(m) is a matcher that matches a string that
// can be parsed as a text-format protobuf of type Proto that matches m, which
// can be any valid protobuf matcher.
template <class Proto, class InnerMatcher>
::testing::PolymorphicMatcher<internal::TypedProtoMatcherStringAdapter<Proto>>
WhenParsedFromProtoTextAs(const InnerMatcher& inner_matcher) {
  return ::testing::MakePolymorphicMatcher(
      internal::TypedProtoMatcherStringAdapter<Proto>(
          ::testing::SafeMatcherCast<const Proto&>(inner_matcher),
          internal::ProtoStringFormat::kTextFormat));
}

}  // namespace proto
}  // namespace testing
}  // namespace tf_opt

#endif  // TF_OPT_OPEN_SOURCE_PROTOCOL_BUFFER_MATCHERS_H_
