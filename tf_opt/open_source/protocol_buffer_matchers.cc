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

#include "tf_opt/open_source/protocol_buffer_matchers.h"

#include <algorithm>
#include <string>

#include "glog/logging.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/tokenizer.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/strings/substitute.h"
#include "re2/re2.h"

namespace tf_opt {
namespace testing {
namespace internal {

////////////////////////////////////////////////////////////////////////////////
// Changes to support open source build, where:
//  * re2 doesn't use absl::string_view
//  * StringErrorCollector isn't defined by proto
// All forked from nucleus/testing/protocol-buffer-matchers.cc
////////////////////////////////////////////////////////////////////////////////
using RegExpStringPiece = re2::StringPiece;

namespace {
bool Consume(RegExpStringPiece* s, RegExpStringPiece x) {
  // We use the implementation of ABSL's StartsWith here until we can pick up a
  // dependency on Abseil.
  if (x.empty() ||
      (s->size() >= x.size() && memcmp(s->data(), x.data(), x.size()) == 0)) {
    s->remove_prefix(x.size());
    return true;
  }
  return false;
}

// Forked from third_party/nucleus/testing/protocol-buffer-matchers.cc
class StringErrorCollector : public google::protobuf::io::ErrorCollector {
 public:
  explicit StringErrorCollector(std::string* error_text)
      : error_text_(error_text) {}

  void AddError(int line, int column, const std::string& message) override {
    absl::SubstituteAndAppend(error_text_, "$0($1): $2\n", line, column,
                              message.c_str());
  }

  void AddWarning(int line, int column, const std::string& message) override {
    absl::SubstituteAndAppend(error_text_, "$0($1): $2\n", line, column,
                              message.c_str());
  }

 private:
  std::string* error_text_;
  StringErrorCollector(const StringErrorCollector&) = delete;
  StringErrorCollector& operator=(const StringErrorCollector&) = delete;
};

}  // namespace

////////////////////////////////////////////////////////////////////////////////
// End of opens source modifications.
////////////////////////////////////////////////////////////////////////////////

// Utilities.
bool ParsePartialFromAscii(absl::string_view pb_ascii, google::protobuf::Message* proto,
                           std::string* error_text) {
  google::protobuf::TextFormat::Parser parser;
  StringErrorCollector collector(error_text);
  parser.RecordErrorsTo(&collector);
  parser.AllowPartialMessage(true);
  // TODO: we should avoid the conversion below, it is needed to
  // support the open source version.
  return parser.ParseFromString(std::string(pb_ascii), proto);
}

// Returns true iff p and q can be compared.
bool ProtoComparable(const google::protobuf::MessageLite& p,
                     const google::protobuf::MessageLite& q) {
  return p.GetTypeName() == q.GetTypeName();
}

bool ProtoComparable(const google::protobuf::Message& p, const google::protobuf::Message& q) {
  return p.GetDescriptor() == q.GetDescriptor();
}

template <typename Container>
std::string JoinStringPieces(const Container& strings,
                             absl::string_view separator) {
  std::stringstream stream;
  absl::string_view sep = "";
  for (const absl::string_view& str : strings) {
    stream << sep << str;
    sep = separator;
  }
  return stream.str();
}

// Find all the descriptors for the ingore_fields.
std::vector<const google::protobuf::FieldDescriptor*> GetFieldDescriptors(
    const google::protobuf::Descriptor* proto_descriptor,
    const std::vector<std::string>& ignore_fields) {
  std::vector<const google::protobuf::FieldDescriptor*> ignore_descriptors;
  std::vector<absl::string_view> remaining_descriptors;

  const google::protobuf::DescriptorPool* pool = proto_descriptor->file()->pool();
  for (const std::string& name : ignore_fields) {
    if (const google::protobuf::FieldDescriptor* field = pool->FindFieldByName(name)) {
      ignore_descriptors.push_back(field);
    } else {
      remaining_descriptors.push_back(name);
    }
  }

  CHECK(remaining_descriptors.empty())
      << "Could not find fields for proto " << proto_descriptor->full_name()
      << " with fully qualified names: "
      << JoinStringPieces(remaining_descriptors, ",");
  return ignore_descriptors;
}

// Sets the ignored fields corresponding to ignore_fields in differencer. Dies
// if any is invalid.
void SetIgnoredFieldsOrDie(const google::protobuf::Descriptor& root_descriptor,
                           const std::vector<std::string>& ignore_fields,
                           google::protobuf::util::MessageDifferencer* differencer) {
  if (!ignore_fields.empty()) {
    std::vector<const google::protobuf::FieldDescriptor*> ignore_descriptors =
        GetFieldDescriptors(&root_descriptor, ignore_fields);
    for (const auto* descr : ignore_descriptors) {
      differencer->IgnoreField(descr);
    }
  }
}

// A criterion that ignores a field path.
class IgnoreFieldPathCriteria
    : public google::protobuf::util::MessageDifferencer::IgnoreCriteria {
 public:
  explicit IgnoreFieldPathCriteria(
      const std::vector<google::protobuf::util::MessageDifferencer::SpecificField>&
          field_path)
      : ignored_field_path_(field_path) {}

  bool IsIgnored(
      const google::protobuf::Message& message1, const google::protobuf::Message& message2,
      const google::protobuf::FieldDescriptor* field,
      const std::vector<google::protobuf::util::MessageDifferencer::SpecificField>&
          parent_fields) override {
    // The off by one is for the current field.
    if (parent_fields.size() + 1 != ignored_field_path_.size()) {
      return false;
    }
    for (int i = 0; i < parent_fields.size(); ++i) {
      const auto& cur_field = parent_fields[i];
      const auto& ignored_field = ignored_field_path_[i];
      // We could compare pointers but it's not guaranteed that descriptors come
      // from the same pool.
      if (cur_field.field->full_name() != ignored_field.field->full_name()) {
        return false;
      }

      // repeated_field[i] is ignored if repeated_field is ignored. To put it
      // differently: if ignored_field specifies an index, we ignore only a
      // field with the same index.
      if (ignored_field.index != -1 && ignored_field.index != cur_field.index) {
        return false;
      }
    }
    return field->full_name() == ignored_field_path_.back().field->full_name();
  }

 private:
  const std::vector<google::protobuf::util::MessageDifferencer::SpecificField>
      ignored_field_path_;
};

// Parses a field path and returns individual components.
std::vector<google::protobuf::util::MessageDifferencer::SpecificField>
ParseFieldPathOrDie(RegExpStringPiece relative_field_path,
                    const google::protobuf::Descriptor& root_descriptor) {
  std::vector<google::protobuf::util::MessageDifferencer::SpecificField> field_path;
  // We're parsing a dot-separated list of elements that can be either:
  //   - field names
  //   - extension names
  //   - indexed field names
  // The parser is very permissive as to what is a field name, then we check
  // the field name against the descriptor.

  // Regular parsers. Consume() does not handle optional captures so we split it
  // in two regexps.
  const RE2 field_regex(R"(([^.()[\]]+))");
  const RE2 field_subscript_regex(R"(([^.()[\]]+)\[(\d+)\])");
  const RE2 extension_regex(R"(\(([^)]+)\))");

  RegExpStringPiece input(relative_field_path);
  while (!input.empty()) {
    // Consume a dot, except on the first iteration.
    if (input.size() < relative_field_path.size() && !Consume(&input, ".")) {
      LOG(FATAL) << "Cannot parse field path '" << relative_field_path
                 << "' at offset " << relative_field_path.size() - input.size()
                 << ": expected '.'";
    }
    // Try to consume a field name. If that fails, consume an extension name.
    google::protobuf::util::MessageDifferencer::SpecificField field;
    std::string name;
    if (RE2::Consume(&input, field_subscript_regex, &name, &field.index) ||
        RE2::Consume(&input, field_regex, &name)) {
      if (field_path.empty()) {
        field.field = root_descriptor.FindFieldByName(name);
        CHECK(field.field) << "No such field '" << name << "' in message '"
                           << root_descriptor.full_name() << "'";
      } else {
        const google::protobuf::util::MessageDifferencer::SpecificField& parent =
            field_path.back();
        field.field = parent.field->message_type()->FindFieldByName(name);
        CHECK(field.field) << "No such field '" << name << "' in '"
                           << parent.field->full_name() << "'";
      }
    } else if (RE2::Consume(&input, extension_regex, &name)) {
      field.field =
          google::protobuf::DescriptorPool::generated_pool()->FindExtensionByName(name);
      CHECK(field.field) << "No such extension '" << name << "'";
      if (field_path.empty()) {
        CHECK(root_descriptor.IsExtensionNumber(field.field->number()))
            << "Extension '" << name << "' does not extend message '"
            << root_descriptor.full_name() << "'";
      } else {
        const google::protobuf::util::MessageDifferencer::SpecificField& parent =
            field_path.back();
        CHECK(parent.field->message_type()->IsExtensionNumber(
            field.field->number()))
            << "Extension '" << name << "' does not extend '"
            << parent.field->full_name() << "'";
      }
    } else {
      LOG(FATAL) << "Cannot parse field path '" << relative_field_path
                 << "' at offset " << relative_field_path.size() - input.size()
                 << ": expected field or extension";
    }
    field_path.push_back(field);
  }

  CHECK(!field_path.empty());
  CHECK(field_path.back().index == -1)
      << "Terminally ignoring fields by index is currently not supported ('"
      << relative_field_path << "')";
  return field_path;
}

// Sets the ignored field paths corresponding to field_paths in differencer.
// Dies if any path is invalid.
void SetIgnoredFieldPathsOrDie(const google::protobuf::Descriptor& root_descriptor,
                               const std::vector<std::string>& field_paths,
                               google::protobuf::util::MessageDifferencer* differencer) {
  for (absl::string_view field_path : field_paths) {
    differencer->AddIgnoreCriteria(new IgnoreFieldPathCriteria(
        ParseFieldPathOrDie(field_path, root_descriptor)));
  }
}

// Configures a MessageDifferencer and DefaultFieldComparator to use the logic
// described in comp. The configured differencer is the output of this function,
// but a FieldComparator must be provided to keep ownership clear.
void ConfigureDifferencer(const ProtoComparison& comp,
                          google::protobuf::util::DefaultFieldComparator* comparator,
                          google::protobuf::util::MessageDifferencer* differencer,
                          const google::protobuf::Descriptor* descriptor) {
  differencer->set_message_field_comparison(comp.field_comp);
  differencer->set_scope(comp.scope);
  comparator->set_float_comparison(comp.float_comp);
  comparator->set_treat_nan_as_equal(comp.treating_nan_as_equal);
  differencer->set_repeated_field_comparison(comp.repeated_field_comp);
  SetIgnoredFieldsOrDie(*descriptor, comp.ignore_fields, differencer);
  SetIgnoredFieldPathsOrDie(*descriptor, comp.ignore_field_paths, differencer);
  if (comp.float_comp == kProtoApproximate &&
      (comp.has_custom_margin || comp.has_custom_fraction)) {
    // Two fields will be considered equal if they're within the fraction _or_
    // within the margin. So setting the fraction to 0.0 makes this effectively
    // a "SetMargin". Similarly, setting the margin to 0.0 makes this
    // effectively a "SetFraction".
    comparator->SetDefaultFractionAndMargin(comp.float_fraction,
                                            comp.float_margin);
  }
  differencer->set_field_comparator(comparator);
}

namespace {

// Returns the given message serialized deterministically. The deterministic
// output is important because we rely on it for comparing lite protos for
// equality.
std::string SerializeDeterministically(const google::protobuf::MessageLite& m) {
  std::string serialized;
  {
    google::protobuf::io::StringOutputStream output_stream(&serialized);
    google::protobuf::io::CodedOutputStream coded_stream(&output_stream);
    coded_stream.SetSerializationDeterministic(true);
    m.SerializePartialToCodedStream(&coded_stream);
  }
  return serialized;
}

}  // namespace

// Returns true iff actual and expected are comparable and match. The comp
// argument specifies how the two are compared. We have separate overloads of
// this function for lite and full protos.
bool ProtoCompare(const ProtoComparison& comp,
                  const google::protobuf::MessageLite& actual,
                  const google::protobuf::MessageLite& expected) {
  // With lite protos we can only do a simple byte-for-byte comparison. It
  // should be impossible to get here with a non-default comparison, but let's
  // CHECK that just in case.
  CHECK_EQ(comp.field_comp, kProtoEqual);
  CHECK_EQ(comp.float_comp, kProtoExact);
  CHECK_EQ(comp.treating_nan_as_equal, false);
  CHECK_EQ(comp.has_custom_margin, false);
  CHECK_EQ(comp.has_custom_fraction, false);
  CHECK_EQ(comp.repeated_field_comp,
           kProtoCompareRepeatedFieldsRespectOrdering);
  CHECK_EQ(comp.scope, kProtoFull);
  CHECK_EQ(comp.float_margin, 0.0);
  CHECK_EQ(comp.float_fraction, 0.0);
  CHECK(comp.ignore_fields.empty());
  CHECK(comp.ignore_field_paths.empty());

  return ProtoComparable(actual, expected) &&
         SerializeDeterministically(actual) ==
             SerializeDeterministically(expected);
}

bool ProtoCompare(const ProtoComparison& comp, const google::protobuf::Message& actual,
                  const google::protobuf::Message& expected) {
  if (!ProtoComparable(actual, expected)) return false;

  google::protobuf::util::MessageDifferencer differencer;
  google::protobuf::util::DefaultFieldComparator field_comparator;
  ConfigureDifferencer(comp, &field_comparator, &differencer,
                       actual.GetDescriptor());

  // It's important for 'expected' to be the first argument here, as
  // Compare() is not symmetric.  When we do a partial comparison,
  // only fields present in the first argument of Compare() are
  // considered.
  return differencer.Compare(expected, actual);
}

// Describes the types of the expected and the actual protocol buffer.
std::string DescribeTypes(const google::protobuf::MessageLite& expected,
                          const google::protobuf::MessageLite& actual) {
  return "whose type should be " + expected.GetTypeName() +
         " but actually is " + actual.GetTypeName();
}

// Describes the differences between the two protocol buffers.
std::string DescribeDiff(const ProtoComparison& /* unused */,
                         const google::protobuf::MessageLite& actual,
                         const google::protobuf::MessageLite& expected) {
  // Since this overload handles lite protos, we cannot use MessageDifferencer,
  // which relies on generated descriptors. As a workaround, we parse the
  // messages into UnknownFieldSets so that we can at least print out the
  // messages in a raw form with field numbers instead of names.
  google::protobuf::UnknownFieldSet actual_unknowns, expected_unknowns;
  CHECK(actual_unknowns.ParseFromString(actual.SerializePartialAsString()));
  CHECK(expected_unknowns.ParseFromString(expected.SerializePartialAsString()));
  std::string actual_string, expected_string;
  CHECK(google::protobuf::TextFormat::PrintUnknownFieldsToString(actual_unknowns,
                                                       &actual_string));
  CHECK(google::protobuf::TextFormat::PrintUnknownFieldsToString(expected_unknowns,
                                                       &expected_string));

  // If either string is empty, make it an empty pair of quotes so that the
  // output is more clear.
  if (actual_string.empty()) {
    actual_string = "\"\"\n";
  }
  if (expected_string.empty()) {
    expected_string = "\"\"\n";
  }

  return absl::StrCat("expected:\n", expected_string, "but found:\n",
                      actual_string);
}

std::string DescribeDiff(const ProtoComparison& comp,
                         const google::protobuf::Message& actual,
                         const google::protobuf::Message& expected) {
  google::protobuf::util::MessageDifferencer differencer;
  google::protobuf::util::DefaultFieldComparator field_comparator;
  ConfigureDifferencer(comp, &field_comparator, &differencer,
                       actual.GetDescriptor());

  std::string diff;
  differencer.ReportDifferencesToString(&diff);

  // We must put 'expected' as the first argument here, as Compare()
  // reports the diff in terms of how the protobuf changes from the
  // first argument to the second argument.
  differencer.Compare(expected, actual);

  // Removes the trailing '\n' in the diff to make the output look nicer.
  if (diff.length() > 0 && *(diff.end() - 1) == '\n') {
    diff.erase(diff.end() - 1);
  }

  return "with the difference:\n" + diff;
}

}  // namespace internal
}  // namespace testing
}  // namespace tf_opt
