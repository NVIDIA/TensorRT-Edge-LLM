/*
  ___        _          Version 3.5.0
 |_ _|_ __  (_) __ _    https://github.com/pantor/inja
  | || '_ \ | |/ _` |   Licensed under the MIT License <http://opensource.org/licenses/MIT>.
  | || | | || | (_| |
 |___|_| |_|/ |\__,_|   Copyright (c) 2018-2025 Lars Berscheid
          |__/
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef INCLUDE_INJA_INJA_HPP_
#define INCLUDE_INJA_INJA_HPP_

// #include "json.hpp"
#ifndef INCLUDE_INJA_JSON_HPP_
#define INCLUDE_INJA_JSON_HPP_

#include <nlohmann/json.hpp>

namespace inja {
#ifndef INJA_DATA_TYPE
using json = nlohmann::json;
#else
using json = INJA_DATA_TYPE;
#endif
} // namespace inja

#endif // INCLUDE_INJA_JSON_HPP_

// #include "throw.hpp"
#ifndef INCLUDE_INJA_THROW_HPP_
#define INCLUDE_INJA_THROW_HPP_

#if (defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)) && !defined(INJA_NOEXCEPTION)
#ifndef INJA_THROW
#define INJA_THROW(exception) throw exception
#endif
#else
#include <cstdlib>
#ifndef INJA_THROW
#define INJA_THROW(exception) \
std::abort();                 \
    std::ignore = exception
#endif
#ifndef INJA_NOEXCEPTION
#define INJA_NOEXCEPTION
#endif
#endif

#endif // INCLUDE_INJA_THROW_HPP_

// #include "environment.hpp"
#ifndef INCLUDE_INJA_ENVIRONMENT_HPP_
#define INCLUDE_INJA_ENVIRONMENT_HPP_

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

// #include "json.hpp"

// #include "config.hpp"
#ifndef INCLUDE_INJA_CONFIG_HPP_
#define INCLUDE_INJA_CONFIG_HPP_

#include <filesystem>
#include <functional>
#include <string>

// #include "template.hpp"
#ifndef INCLUDE_INJA_TEMPLATE_HPP_
#define INCLUDE_INJA_TEMPLATE_HPP_

#include <map>
#include <memory>
#include <string>

// #include "node.hpp"
#ifndef INCLUDE_INJA_NODE_HPP_
#define INCLUDE_INJA_NODE_HPP_

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

// #include "function_storage.hpp"
#ifndef INCLUDE_INJA_FUNCTION_STORAGE_HPP_
#define INCLUDE_INJA_FUNCTION_STORAGE_HPP_

#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// #include "json.hpp"


namespace inja {

using Arguments = std::vector<const json*>;
using CallbackFunction = std::function<json(Arguments& args)>;
using VoidCallbackFunction = std::function<void(Arguments& args)>;

/*!
 * \brief Class for builtin functions and user-defined callbacks.
 */
class FunctionStorage {
public:
  enum class Operation {
    Not,
    And,
    Or,
    In,
    NotIn,
    Equal,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Add,
    Concat,
    Subtract,
    Multiplication,
    Division,
    Power,
    Modulo,
    AtId,
    At,
    Slice,
    Conditional,
    Array,
    Object,
    Capitalize,
    Default,
    DivisibleBy,
    Even,
    Exists,
    ExistsInObject,
    First,
    Float,
    Int,
    IsArray,
    IsBoolean,
    IsFloat,
    IsInteger,
    IsDefined,
    IsUndefined,
    IsNone,
    IsIterable,
    IsMapping,
    IsNumber,
    IsObject,
    IsSequence,
    IsString,
    IsTrue,
    IsFalse,
    Last,
    Length,
    Lower,
    Max,
    Min,
    Odd,
    Range,
    Replace,
    Round,
    Sort,
    Upper,
    Super,
    Join,
    MacroCall,
    Callback,
    None,
  };

  struct FunctionData {
    explicit FunctionData(const Operation& op, const CallbackFunction& cb = CallbackFunction {}): operation(op), callback(cb) {}
    const Operation operation;
    const CallbackFunction callback;
  };

private:
  const int VARIADIC {-1};

  std::map<std::pair<std::string, int>, FunctionData> function_storage = {
      {std::make_pair("at", 2), FunctionData {Operation::At}},
      {std::make_pair("capitalize", 1), FunctionData {Operation::Capitalize}},
      {std::make_pair("default", 2), FunctionData {Operation::Default}},
      {std::make_pair("default", 3), FunctionData {Operation::Default}},
      {std::make_pair("divisibleBy", 2), FunctionData {Operation::DivisibleBy}},
      {std::make_pair("even", 1), FunctionData {Operation::Even}},
      {std::make_pair("exists", 1), FunctionData {Operation::Exists}},
      {std::make_pair("existsIn", 2), FunctionData {Operation::ExistsInObject}},
      {std::make_pair("first", 1), FunctionData {Operation::First}},
      {std::make_pair("float", 1), FunctionData {Operation::Float}},
      {std::make_pair("int", 1), FunctionData {Operation::Int}},
      {std::make_pair("isArray", 1), FunctionData {Operation::IsArray}},
      {std::make_pair("isBoolean", 1), FunctionData {Operation::IsBoolean}},
      {std::make_pair("isFloat", 1), FunctionData {Operation::IsFloat}},
      {std::make_pair("isInteger", 1), FunctionData {Operation::IsInteger}},
      {std::make_pair("isNumber", 1), FunctionData {Operation::IsNumber}},
      {std::make_pair("isObject", 1), FunctionData {Operation::IsObject}},
      {std::make_pair("isString", 1), FunctionData {Operation::IsString}},
      {std::make_pair("last", 1), FunctionData {Operation::Last}},
      {std::make_pair("length", 1), FunctionData {Operation::Length}},
      {std::make_pair("lower", 1), FunctionData {Operation::Lower}},
      {std::make_pair("max", 1), FunctionData {Operation::Max}},
      {std::make_pair("min", 1), FunctionData {Operation::Min}},
      {std::make_pair("odd", 1), FunctionData {Operation::Odd}},
      {std::make_pair("range", 1), FunctionData {Operation::Range}},
      {std::make_pair("replace", 3), FunctionData {Operation::Replace}},
      {std::make_pair("round", 2), FunctionData {Operation::Round}},
      {std::make_pair("sort", 1), FunctionData {Operation::Sort}},
      {std::make_pair("upper", 1), FunctionData {Operation::Upper}},
      {std::make_pair("super", 0), FunctionData {Operation::Super}},
      {std::make_pair("super", 1), FunctionData {Operation::Super}},
      {std::make_pair("join", 2), FunctionData {Operation::Join}},
  };

public:
  void add_builtin(std::string_view name, int num_args, Operation op) {
    function_storage.emplace(std::make_pair(static_cast<std::string>(name), num_args), FunctionData {op});
  }

  void add_callback(std::string_view name, int num_args, const CallbackFunction& callback) {
    function_storage.emplace(std::make_pair(static_cast<std::string>(name), num_args), FunctionData {Operation::Callback, callback});
  }

  FunctionData find_function(std::string_view name, int num_args) const {
    auto it = function_storage.find(std::make_pair(static_cast<std::string>(name), num_args));
    if (it != function_storage.end()) {
      return it->second;

      // Find variadic function
    } else if (num_args > 0) {
      it = function_storage.find(std::make_pair(static_cast<std::string>(name), VARIADIC));
      if (it != function_storage.end()) {
        return it->second;
      }
    }

    return FunctionData {Operation::None};
  }
};

} // namespace inja

#endif // INCLUDE_INJA_FUNCTION_STORAGE_HPP_

// #include "utils.hpp"
#ifndef INCLUDE_INJA_UTILS_HPP_
#define INCLUDE_INJA_UTILS_HPP_

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>

// #include "exceptions.hpp"
#ifndef INCLUDE_INJA_EXCEPTIONS_HPP_
#define INCLUDE_INJA_EXCEPTIONS_HPP_

#include <cstddef>
#include <stdexcept>
#include <string>

namespace inja {

struct SourceLocation {
  size_t line;
  size_t column;
};

struct InjaError : public std::runtime_error {
  const std::string type;
  const std::string message;

  const SourceLocation location;

  explicit InjaError(const std::string& type, const std::string& message)
      : std::runtime_error("[inja.exception." + type + "] " + message), type(type), message(message), location({0, 0}) {}

  explicit InjaError(const std::string& type, const std::string& message, SourceLocation location)
      : std::runtime_error("[inja.exception." + type + "] (at " + std::to_string(location.line) + ":" + std::to_string(location.column) + ") " + message),
        type(type), message(message), location(location) {}
};

struct ParserError : public InjaError {
  explicit ParserError(const std::string& message, SourceLocation location): InjaError("parser_error", message, location) {}
};

struct RenderError : public InjaError {
  explicit RenderError(const std::string& message, SourceLocation location): InjaError("render_error", message, location) {}
};

struct FileError : public InjaError {
  explicit FileError(const std::string& message): InjaError("file_error", message) {}
  explicit FileError(const std::string& message, SourceLocation location): InjaError("file_error", message, location) {}
};

struct DataError : public InjaError {
  explicit DataError(const std::string& message, SourceLocation location): InjaError("data_error", message, location) {}
};

} // namespace inja

#endif // INCLUDE_INJA_EXCEPTIONS_HPP_


namespace inja {

namespace string_view {
inline std::string_view slice(std::string_view view, size_t start, size_t end) {
  start = std::min(start, view.size());
  end = std::min(std::max(start, end), view.size());
  return view.substr(start, end - start);
}

inline std::pair<std::string_view, std::string_view> split(std::string_view view, char Separator) {
  const size_t idx = view.find(Separator);
  if (idx == std::string_view::npos) {
    return std::make_pair(view, std::string_view());
  }
  return std::make_pair(slice(view, 0, idx), slice(view, idx + 1, std::string_view::npos));
}

inline bool starts_with(std::string_view view, std::string_view prefix) {
  return (view.size() >= prefix.size() && view.compare(0, prefix.size(), prefix) == 0);
}
} // namespace string_view

inline SourceLocation get_source_location(std::string_view content, size_t pos) {
  // Get line and offset position (starts at 1:1)
  auto sliced = string_view::slice(content, 0, pos);
  const std::size_t last_newline = sliced.rfind('\n');

  if (last_newline == std::string_view::npos) {
    return {1, sliced.length() + 1};
  }

  // Count newlines
  size_t count_lines = 0;
  size_t search_start = 0;
  while (search_start <= sliced.size()) {
    search_start = sliced.find('\n', search_start) + 1;
    if (search_start == 0) {
      break;
    }
    count_lines += 1;
  }

  return {count_lines + 1, sliced.length() - last_newline};
}

inline void replace_substring(std::string& s, const std::string& f, const std::string& t) {
  if (f.empty()) {
    return;
  }
  for (auto pos = s.find(f);            // find first occurrence of f
       pos != std::string::npos;        // make sure f was found
       s.replace(pos, f.size(), t),     // replace with t, and
       pos = s.find(f, pos + t.size())) // find next occurrence of f
  {}
}

} // namespace inja

#endif // INCLUDE_INJA_UTILS_HPP_

// #include "json.hpp"


namespace inja {

class NodeVisitor;
class BlockNode;
class TextNode;
class ExpressionNode;
class LiteralNode;
class DataNode;
class FunctionNode;
class ExpressionListNode;
class StatementNode;
class ForStatementNode;
class ForArrayStatementNode;
class ForObjectStatementNode;
class IfStatementNode;
class IncludeStatementNode;
class ExtendsStatementNode;
class BlockStatementNode;
class SetStatementNode;
class SetBlockStatementNode;
class MacroStatementNode;
class LoopControlStatementNode;

class NodeVisitor {
public:
  virtual ~NodeVisitor() = default;

  virtual void visit(const BlockNode& node) = 0;
  virtual void visit(const TextNode& node) = 0;
  virtual void visit(const ExpressionNode& node) = 0;
  virtual void visit(const LiteralNode& node) = 0;
  virtual void visit(const DataNode& node) = 0;
  virtual void visit(const FunctionNode& node) = 0;
  virtual void visit(const ExpressionListNode& node) = 0;
  virtual void visit(const StatementNode& node) = 0;
  virtual void visit(const ForStatementNode& node) = 0;
  virtual void visit(const ForArrayStatementNode& node) = 0;
  virtual void visit(const ForObjectStatementNode& node) = 0;
  virtual void visit(const IfStatementNode& node) = 0;
  virtual void visit(const IncludeStatementNode& node) = 0;
  virtual void visit(const ExtendsStatementNode& node) = 0;
  virtual void visit(const BlockStatementNode& node) = 0;
  virtual void visit(const SetStatementNode& node) = 0;
  virtual void visit(const SetBlockStatementNode& node) = 0;
  virtual void visit(const MacroStatementNode& node) = 0;
  virtual void visit(const LoopControlStatementNode& node) = 0;
};

/*!
 * \brief Base node class for the abstract syntax tree (AST).
 */
class AstNode {
public:
  virtual void accept(NodeVisitor& v) const = 0;

  size_t pos;

  explicit AstNode(size_t pos): pos(pos) {}
  virtual ~AstNode() {}
};

class BlockNode : public AstNode {
public:
  std::vector<std::shared_ptr<AstNode>> nodes;

  explicit BlockNode(): AstNode(0) {}

  void accept(NodeVisitor& v) const override {
    v.visit(*this);
  }
};

class TextNode : public AstNode {
public:
  const size_t length;

  explicit TextNode(size_t pos, size_t length): AstNode(pos), length(length) {}

  void accept(NodeVisitor& v) const override {
    v.visit(*this);
  }
};

class ExpressionNode : public AstNode {
public:
  explicit ExpressionNode(size_t pos): AstNode(pos) {}

  void accept(NodeVisitor& v) const override {
    v.visit(*this);
  }
};

class LiteralNode : public ExpressionNode {
public:
  const json value;

  static json parse_literal(std::string_view data_text) {
    std::string normalized;
    normalized.reserve(data_text.size());
    char quote = 0;
    bool escaped = false;
    for (size_t index = 0; index < data_text.size(); ++index) {
      const char ch = data_text[index];
      if (quote != 0) {
        if (escaped) {
          if (quote == '\'' && ch == '\'') {
            normalized.push_back('\'');
          } else if (ch == '"' || ch == '\\' || ch == '/' || ch == 'b' || ch == 'f' || ch == 'n' || ch == 'r' || ch == 't' || ch == 'u') {
            normalized.push_back('\\');
            normalized.push_back(ch);
          } else {
            // Python/Jinja preserve unknown escapes such as "\\s". Escape
            // the backslash once more for the normalized JSON literal.
            normalized.push_back('\\');
            normalized.push_back('\\');
            normalized.push_back(ch);
          }
          escaped = false;
        } else if (ch == '\\') {
          escaped = true;
        } else if (ch == quote) {
          normalized.push_back('"');
          quote = 0;
        } else {
          if (quote == '\'' && ch == '"') {
            normalized.push_back('\\');
          }
          switch (ch) {
          case '\n': normalized.append("\\n"); break;
          case '\r': normalized.append("\\r"); break;
          case '\t': normalized.append("\\t"); break;
          default:
            if (static_cast<unsigned char>(ch) < 0x20) {
              constexpr char hex[] = "0123456789abcdef";
              normalized.append("\\u00");
              normalized.push_back(hex[(static_cast<unsigned char>(ch) >> 4) & 0xf]);
              normalized.push_back(hex[static_cast<unsigned char>(ch) & 0xf]);
            } else {
              normalized.push_back(ch);
            }
          }
        }
        continue;
      }
      if (ch == '\'' || ch == '"') {
        quote = ch;
        normalized.push_back('"');
        continue;
      }
      const auto keyword = [&](std::string_view from, std::string_view to) {
        if (data_text.substr(index, from.size()) == from) {
          normalized.append(to);
          index += from.size() - 1;
          return true;
        }
        return false;
      };
      if (!keyword("True", "true") && !keyword("False", "false") && !keyword("None", "null") && !keyword("none", "null")) {
        normalized.push_back(ch);
      }
    }
    if (quote != 0 || escaped) {
      INJA_THROW(std::runtime_error("unterminated string literal"));
    }
    return json::parse(normalized);
  }

  explicit LiteralNode(std::string_view data_text, size_t pos): ExpressionNode(pos), value(parse_literal(data_text)) {}
  explicit LiteralNode(json data, size_t pos): ExpressionNode(pos), value(std::move(data)) {}

  void accept(NodeVisitor& v) const override {
    v.visit(*this);
  }
};

class DataNode : public ExpressionNode {
public:
  const std::string name;
  const json::json_pointer ptr;

  static std::string convert_dot_to_ptr(std::string_view ptr_name) {
    std::string result;
    do {
      std::string_view part;
      std::tie(part, ptr_name) = string_view::split(ptr_name, '.');
      result.push_back('/');
      result.append(part.begin(), part.end());
    } while (!ptr_name.empty());
    return result;
  }

  explicit DataNode(std::string_view ptr_name, size_t pos): ExpressionNode(pos), name(ptr_name), ptr(json::json_pointer(convert_dot_to_ptr(ptr_name))) {}

  void accept(NodeVisitor& v) const override {
    v.visit(*this);
  }
};

class FunctionNode : public ExpressionNode {
  using Op = FunctionStorage::Operation;

public:
  enum class Associativity {
    Left,
    Right,
  };

  unsigned int precedence;
  Associativity associativity;

  Op operation;

  std::string name;
  int number_args; // Can also be negative -> -1 for unknown number
  std::vector<std::shared_ptr<ExpressionNode>> arguments;
  std::vector<std::string> argument_names;
  CallbackFunction callback;

  explicit FunctionNode(std::string_view name, size_t pos)
      : ExpressionNode(pos), precedence(8), associativity(Associativity::Left), operation(Op::Callback), name(name), number_args(0) {}
  explicit FunctionNode(Op operation, size_t pos): ExpressionNode(pos), operation(operation), number_args(1) {
    switch (operation) {
    case Op::Not: {
      number_args = 1;
      precedence = 4;
      associativity = Associativity::Left;
    } break;
    case Op::And: {
      number_args = 2;
      precedence = 1;
      associativity = Associativity::Left;
    } break;
    case Op::Or: {
      number_args = 2;
      precedence = 1;
      associativity = Associativity::Left;
    } break;
    case Op::In: {
      number_args = 2;
      precedence = 2;
      associativity = Associativity::Left;
    } break;
    case Op::NotIn: {
      number_args = 2;
      precedence = 2;
      associativity = Associativity::Left;
    } break;
    case Op::Equal: {
      number_args = 2;
      precedence = 2;
      associativity = Associativity::Left;
    } break;
    case Op::NotEqual: {
      number_args = 2;
      precedence = 2;
      associativity = Associativity::Left;
    } break;
    case Op::Greater: {
      number_args = 2;
      precedence = 2;
      associativity = Associativity::Left;
    } break;
    case Op::GreaterEqual: {
      number_args = 2;
      precedence = 2;
      associativity = Associativity::Left;
    } break;
    case Op::Less: {
      number_args = 2;
      precedence = 2;
      associativity = Associativity::Left;
    } break;
    case Op::LessEqual: {
      number_args = 2;
      precedence = 2;
      associativity = Associativity::Left;
    } break;
    case Op::Add: {
      number_args = 2;
      precedence = 3;
      associativity = Associativity::Left;
    } break;
    case Op::Concat: {
      number_args = 2;
      precedence = 3;
      associativity = Associativity::Left;
    } break;
    case Op::Subtract: {
      number_args = 2;
      precedence = 3;
      associativity = Associativity::Left;
    } break;
    case Op::Multiplication: {
      number_args = 2;
      precedence = 4;
      associativity = Associativity::Left;
    } break;
    case Op::Division: {
      number_args = 2;
      precedence = 4;
      associativity = Associativity::Left;
    } break;
    case Op::Power: {
      number_args = 2;
      precedence = 5;
      associativity = Associativity::Right;
    } break;
    case Op::Modulo: {
      number_args = 2;
      precedence = 4;
      associativity = Associativity::Left;
    } break;
    case Op::AtId: {
      number_args = 2;
      precedence = 8;
      associativity = Associativity::Left;
    } break;
    default: {
      precedence = 1;
      associativity = Associativity::Left;
    }
    }
  }

  void accept(NodeVisitor& v) const override {
    v.visit(*this);
  }
};

class ExpressionListNode : public AstNode {
public:
  std::shared_ptr<ExpressionNode> root;

  explicit ExpressionListNode(): AstNode(0) {}
  explicit ExpressionListNode(size_t pos): AstNode(pos) {}

  void accept(NodeVisitor& v) const override {
    v.visit(*this);
  }
};

class StatementNode : public AstNode {
public:
  explicit StatementNode(size_t pos): AstNode(pos) {}

  virtual void accept(NodeVisitor& v) const = 0;
};

class ForStatementNode : public StatementNode {
public:
  ExpressionListNode condition;
  ExpressionListNode filter;
  bool has_filter {false};
  BlockNode body;
  BlockNode* const parent;

  explicit ForStatementNode(BlockNode* const parent, size_t pos): StatementNode(pos), parent(parent) {}

  virtual void accept(NodeVisitor& v) const = 0;
};

class ForArrayStatementNode : public ForStatementNode {
public:
  const std::string value;

  explicit ForArrayStatementNode(const std::string& value, BlockNode* const parent, size_t pos): ForStatementNode(parent, pos), value(value) {}

  void accept(NodeVisitor& v) const override {
    v.visit(*this);
  }
};

class ForObjectStatementNode : public ForStatementNode {
public:
  const std::string key;
  const std::string value;

  explicit ForObjectStatementNode(const std::string& key, const std::string& value, BlockNode* const parent, size_t pos)
      : ForStatementNode(parent, pos), key(key), value(value) {}

  void accept(NodeVisitor& v) const override {
    v.visit(*this);
  }
};

class IfStatementNode : public StatementNode {
public:
  ExpressionListNode condition;
  BlockNode true_statement;
  BlockNode false_statement;
  BlockNode* const parent;

  const bool is_nested;
  bool has_false_statement {false};

  explicit IfStatementNode(BlockNode* const parent, size_t pos): StatementNode(pos), parent(parent), is_nested(false) {}
  explicit IfStatementNode(bool is_nested, BlockNode* const parent, size_t pos): StatementNode(pos), parent(parent), is_nested(is_nested) {}

  void accept(NodeVisitor& v) const override {
    v.visit(*this);
  }
};

class IncludeStatementNode : public StatementNode {
public:
  const std::string file;

  explicit IncludeStatementNode(const std::string& file, size_t pos): StatementNode(pos), file(file) {}

  void accept(NodeVisitor& v) const override {
    v.visit(*this);
  }
};

class ExtendsStatementNode : public StatementNode {
public:
  const std::string file;

  explicit ExtendsStatementNode(const std::string& file, size_t pos): StatementNode(pos), file(file) {}

  void accept(NodeVisitor& v) const override {
    v.visit(*this);
  }
};

class BlockStatementNode : public StatementNode {
public:
  const std::string name;
  BlockNode block;
  BlockNode* const parent;

  explicit BlockStatementNode(BlockNode* const parent, const std::string& name, size_t pos): StatementNode(pos), name(name), parent(parent) {}

  void accept(NodeVisitor& v) const override {
    v.visit(*this);
  }
};

class SetStatementNode : public StatementNode {
public:
  const std::string key;
  ExpressionListNode expression;

  explicit SetStatementNode(const std::string& key, size_t pos): StatementNode(pos), key(key) {}

  void accept(NodeVisitor& v) const override {
    v.visit(*this);
  }
};

class SetBlockStatementNode : public StatementNode {
public:
  const std::string key;
  BlockNode body;
  BlockNode* const parent;

  explicit SetBlockStatementNode(std::string key, BlockNode* const parent, size_t pos)
      : StatementNode(pos), key(std::move(key)), parent(parent) {}

  void accept(NodeVisitor& v) const override {
    v.visit(*this);
  }
};

class MacroStatementNode : public StatementNode {
public:
  struct Parameter {
    std::string name;
    std::shared_ptr<ExpressionNode> default_value;
  };

  const std::string name;
  std::vector<Parameter> parameters;
  BlockNode body;
  BlockNode* const parent;

  explicit MacroStatementNode(std::string name, BlockNode* const parent, size_t pos)
      : StatementNode(pos), name(std::move(name)), parent(parent) {}

  void accept(NodeVisitor& v) const override {
    v.visit(*this);
  }
};

class LoopControlStatementNode : public StatementNode {
public:
  enum class Control { Break, Continue };

  const Control control;

  explicit LoopControlStatementNode(Control control, size_t pos): StatementNode(pos), control(control) {}

  void accept(NodeVisitor& v) const override {
    v.visit(*this);
  }
};

} // namespace inja

#endif // INCLUDE_INJA_NODE_HPP_

// #include "statistics.hpp"
#ifndef INCLUDE_INJA_STATISTICS_HPP_
#define INCLUDE_INJA_STATISTICS_HPP_

// #include "node.hpp"


namespace inja {

/*!
 * \brief A class for counting statistics on a Template.
 */
class StatisticsVisitor : public NodeVisitor {
  void visit(const BlockNode& node) override {
    for (const auto& n : node.nodes) {
      n->accept(*this);
    }
  }

  void visit(const TextNode&) override {}
  void visit(const ExpressionNode&) override {}
  void visit(const LiteralNode&) override {}

  void visit(const DataNode&) override {
    variable_counter += 1;
  }

  void visit(const FunctionNode& node) override {
    for (const auto& n : node.arguments) {
      n->accept(*this);
    }
  }

  void visit(const ExpressionListNode& node) override {
    node.root->accept(*this);
  }

  void visit(const StatementNode&) override {}
  void visit(const ForStatementNode&) override {}

  void visit(const ForArrayStatementNode& node) override {
    node.condition.accept(*this);
    node.body.accept(*this);
  }

  void visit(const ForObjectStatementNode& node) override {
    node.condition.accept(*this);
    node.body.accept(*this);
  }

  void visit(const IfStatementNode& node) override {
    node.condition.accept(*this);
    node.true_statement.accept(*this);
    node.false_statement.accept(*this);
  }

  void visit(const IncludeStatementNode&) override {}

  void visit(const ExtendsStatementNode&) override {}

  void visit(const BlockStatementNode& node) override {
    node.block.accept(*this);
  }

  void visit(const SetStatementNode&) override {}
  void visit(const SetBlockStatementNode& node) override {
    node.body.accept(*this);
  }
  void visit(const MacroStatementNode& node) override {
    node.body.accept(*this);
  }
  void visit(const LoopControlStatementNode&) override {}

public:
  size_t variable_counter {0};

  explicit StatisticsVisitor() {}
};

} // namespace inja

#endif // INCLUDE_INJA_STATISTICS_HPP_


namespace inja {

/*!
 * \brief The main inja Template.
 */
struct Template {
  BlockNode root;
  std::string content;
  std::map<std::string, std::shared_ptr<BlockStatementNode>> block_storage;
  std::map<std::string, std::shared_ptr<MacroStatementNode>> macro_storage;

  explicit Template() {}
  explicit Template(std::string content): content(std::move(content)) {}

  /// Return number of variables (total number, not distinct ones) in the template
  size_t count_variables() const {
    auto statistic_visitor = StatisticsVisitor();
    root.accept(statistic_visitor);
    return statistic_visitor.variable_counter;
  }
};

using TemplateStorage = std::map<std::string, Template>;

} // namespace inja

#endif // INCLUDE_INJA_TEMPLATE_HPP_


namespace inja {

/*!
 * \brief Class for lexer configuration.
 */
struct LexerConfig {
  std::string statement_open {"{%"};
  std::string statement_open_no_lstrip {"{%+"};
  std::string statement_open_force_lstrip {"{%-"};
  std::string statement_close {"%}"};
  std::string statement_close_force_rstrip {"-%}"};
  std::string line_statement {"##"};
  std::string expression_open {"{{"};
  std::string expression_open_force_lstrip {"{{-"};
  std::string expression_close {"}}"};
  std::string expression_close_force_rstrip {"-}}"};
  std::string comment_open {"{#"};
  std::string comment_open_force_lstrip {"{#-"};
  std::string comment_close {"#}"};
  std::string comment_close_force_rstrip {"-#}"};
  std::string open_chars {"#{"};

  bool trim_blocks {false};
  bool lstrip_blocks {false};

  void update_open_chars() {
    open_chars = "";
    if (!line_statement.empty() && open_chars.find(line_statement[0]) == std::string::npos) {
      open_chars += line_statement[0];
    }
    if (open_chars.find(statement_open[0]) == std::string::npos) {
      open_chars += statement_open[0];
    }
    if (open_chars.find(statement_open_no_lstrip[0]) == std::string::npos) {
      open_chars += statement_open_no_lstrip[0];
    }
    if (open_chars.find(statement_open_force_lstrip[0]) == std::string::npos) {
      open_chars += statement_open_force_lstrip[0];
    }
    if (open_chars.find(expression_open[0]) == std::string::npos) {
      open_chars += expression_open[0];
    }
    if (open_chars.find(expression_open_force_lstrip[0]) == std::string::npos) {
      open_chars += expression_open_force_lstrip[0];
    }
    if (open_chars.find(comment_open[0]) == std::string::npos) {
      open_chars += comment_open[0];
    }
    if (open_chars.find(comment_open_force_lstrip[0]) == std::string::npos) {
      open_chars += comment_open_force_lstrip[0];
    }
  }
};

/*!
 * \brief Class for parser configuration.
 */
struct ParserConfig {
  bool search_included_templates_in_files {true};

  std::function<Template(const std::filesystem::path&, const std::string&)> include_callback;
};

/*!
 * \brief Class for render configuration.
 */
struct RenderConfig {
  bool throw_at_missing_includes {true};
  bool html_autoescape {false};
};

} // namespace inja

#endif // INCLUDE_INJA_CONFIG_HPP_

// #include "function_storage.hpp"

// #include "parser.hpp"
#ifndef INCLUDE_INJA_PARSER_HPP_
#define INCLUDE_INJA_PARSER_HPP_

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stack>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// #include "config.hpp"

// #include "exceptions.hpp"

// #include "function_storage.hpp"

// #include "lexer.hpp"
#ifndef INCLUDE_INJA_LEXER_HPP_
#define INCLUDE_INJA_LEXER_HPP_

#include <cctype>
#include <cstddef>
#include <string_view>

// #include "config.hpp"

// #include "exceptions.hpp"

// #include "token.hpp"
#ifndef INCLUDE_INJA_TOKEN_HPP_
#define INCLUDE_INJA_TOKEN_HPP_

#include <string>
#include <string_view>

namespace inja {

/*!
 * \brief Helper-class for the inja Lexer.
 */
struct Token {
  enum class Kind {
    Text,
    ExpressionOpen,     // {{
    ExpressionClose,    // }}
    LineStatementOpen,  // ##
    LineStatementClose, // \n
    StatementOpen,      // {%
    StatementClose,     // %}
    CommentOpen,        // {#
    CommentClose,       // #}
    Id,                 // this, this.foo
    Number,             // 1, 2, -1, 5.2, -5.3
    String,             // "this"
    Plus,               // +
    Minus,              // -
    Times,              // *
    Slash,              // /
    Percent,            // %
    Power,              // ^
    Comma,              // ,
    Dot,                // .
    Colon,              // :
    LeftParen,          // (
    RightParen,         // )
    LeftBracket,        // [
    RightBracket,       // ]
    LeftBrace,          // {
    RightBrace,         // }
    Equal,              // ==
    NotEqual,           // !=
    GreaterThan,        // >
    GreaterEqual,       // >=
    LessThan,           // <
    LessEqual,          // <=
    Pipe,               // |
    Tilde,              // ~
    Unknown,
    Eof,
  };

  Kind kind {Kind::Unknown};
  std::string_view text;

  explicit constexpr Token() = default;
  explicit constexpr Token(Kind kind, std::string_view text): kind(kind), text(text) {}

  std::string describe() const {
    switch (kind) {
    case Kind::Text:
      return "<text>";
    case Kind::LineStatementClose:
      return "<eol>";
    case Kind::Eof:
      return "<eof>";
    default:
      return static_cast<std::string>(text);
    }
  }
};

} // namespace inja

#endif // INCLUDE_INJA_TOKEN_HPP_

// #include "utils.hpp"


namespace inja {

/*!
 * \brief Class for lexing an inja Template.
 */
class Lexer {
  enum class State {
    Text,
    ExpressionStart,
    ExpressionStartForceLstrip,
    ExpressionBody,
    LineStart,
    LineBody,
    StatementStart,
    StatementStartNoLstrip,
    StatementStartForceLstrip,
    StatementBody,
    CommentStart,
    CommentStartForceLstrip,
    CommentBody,
  };

  enum class MinusState {
    Operator,
    Number,
  };

  const LexerConfig& config;

  State state;
  MinusState minus_state;
  std::string_view m_in;
  size_t tok_start;
  size_t pos;

  Token scan_body(std::string_view close, Token::Kind closeKind, std::string_view close_trim = std::string_view(), bool trim = false) {
  again:
    // skip whitespace (except for \n as it might be a close)
    if (tok_start >= m_in.size()) {
      return make_token(Token::Kind::Eof);
    }
    const char ch = m_in[tok_start];
    if (ch == ' ' || ch == '\t' || ch == '\r') {
      tok_start += 1;
      goto again;
    }

    // check for close
    if (!close_trim.empty() && inja::string_view::starts_with(m_in.substr(tok_start), close_trim)) {
      state = State::Text;
      pos = tok_start + close_trim.size();
      const Token tok = make_token(closeKind);
      skip_whitespaces_and_newlines();
      return tok;
    }

    if (inja::string_view::starts_with(m_in.substr(tok_start), close)) {
      state = State::Text;
      pos = tok_start + close.size();
      const Token tok = make_token(closeKind);
      if (trim) {
        skip_whitespaces_and_first_newline();
      }
      return tok;
    }

    // skip \n
    if (ch == '\n') {
      tok_start += 1;
      goto again;
    }

    pos = tok_start + 1;
    if (std::isalpha(ch)) {
      minus_state = MinusState::Operator;
      return scan_id();
    }

    const MinusState current_minus_state = minus_state;
    if (minus_state == MinusState::Operator) {
      minus_state = MinusState::Number;
    }

    switch (ch) {
    case '+':
      return make_token(Token::Kind::Plus);
    case '-':
      if (current_minus_state == MinusState::Operator) {
        return make_token(Token::Kind::Minus);
      }
      return scan_number();
    case '*':
      return make_token(Token::Kind::Times);
    case '/':
      return make_token(Token::Kind::Slash);
    case '^':
      return make_token(Token::Kind::Power);
    case '%':
      return make_token(Token::Kind::Percent);
    case '.':
      return make_token(Token::Kind::Dot);
    case ',':
      return make_token(Token::Kind::Comma);
    case ':':
      return make_token(Token::Kind::Colon);
    case '|':
      return make_token(Token::Kind::Pipe);
    case '~':
      return make_token(Token::Kind::Tilde);
    case '(':
      return make_token(Token::Kind::LeftParen);
    case ')':
      minus_state = MinusState::Operator;
      return make_token(Token::Kind::RightParen);
    case '[':
      return make_token(Token::Kind::LeftBracket);
    case ']':
      minus_state = MinusState::Operator;
      return make_token(Token::Kind::RightBracket);
    case '{':
      return make_token(Token::Kind::LeftBrace);
    case '}':
      minus_state = MinusState::Operator;
      return make_token(Token::Kind::RightBrace);
    case '>':
      if (pos < m_in.size() && m_in[pos] == '=') {
        pos += 1;
        return make_token(Token::Kind::GreaterEqual);
      }
      return make_token(Token::Kind::GreaterThan);
    case '<':
      if (pos < m_in.size() && m_in[pos] == '=') {
        pos += 1;
        return make_token(Token::Kind::LessEqual);
      }
      return make_token(Token::Kind::LessThan);
    case '=':
      if (pos < m_in.size() && m_in[pos] == '=') {
        pos += 1;
        return make_token(Token::Kind::Equal);
      }
      return make_token(Token::Kind::Unknown);
    case '!':
      if (pos < m_in.size() && m_in[pos] == '=') {
        pos += 1;
        return make_token(Token::Kind::NotEqual);
      }
      return make_token(Token::Kind::Unknown);
    case '\"':
    case '\'':
      return scan_string();
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      minus_state = MinusState::Operator;
      return scan_number();
    case '_':
    case '@':
    case '$':
      minus_state = MinusState::Operator;
      return scan_id();
    default:
      return make_token(Token::Kind::Unknown);
    }
  }

  Token scan_id() {
    for (;;) {
      if (pos >= m_in.size()) {
        break;
      }
      const char ch = m_in[pos];
      if (!std::isalnum(ch) && ch != '.' && ch != '_') {
        break;
      }
      pos += 1;
    }
    return make_token(Token::Kind::Id);
  }

  Token scan_number() {
    for (;;) {
      if (pos >= m_in.size()) {
        break;
      }
      const char ch = m_in[pos];
      // be very permissive in lexer (we'll catch errors when conversion happens)
      if (!(std::isdigit(ch) || ch == '.' || ch == 'e' || ch == 'E' || (ch == '+' && (pos == 0 || m_in[pos-1] == 'e' || m_in[pos-1] == 'E')) || (ch == '-' && (pos == 0 || m_in[pos-1] == 'e' || m_in[pos-1] == 'E')))) {
        break;
      }
      pos += 1;
    }
    return make_token(Token::Kind::Number);
  }

  Token scan_string() {
    bool escape {false};
    for (;;) {
      if (pos >= m_in.size()) {
        break;
      }
      const char ch = m_in[pos++];
      if (ch == '\\') {
        escape = !escape;
      } else if (!escape && ch == m_in[tok_start]) {
        break;
      } else {
        escape = false;
      }
    }
    return make_token(Token::Kind::String);
  }

  Token make_token(Token::Kind kind) const {
    return Token(kind, string_view::slice(m_in, tok_start, pos));
  }

  void skip_whitespaces_and_newlines() {
    if (pos < m_in.size()) {
      while (pos < m_in.size() && (m_in[pos] == ' ' || m_in[pos] == '\t' || m_in[pos] == '\n' || m_in[pos] == '\r')) {
        pos += 1;
      }
    }
  }

  void skip_whitespaces_and_first_newline() {
    if (pos < m_in.size()) {
      const char ch = m_in[pos];
      if (ch == '\n') {
        pos += 1;
      } else if (ch == '\r') {
        pos += 1;
        if (pos < m_in.size() && m_in[pos] == '\n') {
          pos += 1;
        }
      }
    }
  }

  static std::string_view clear_final_line_if_whitespace(std::string_view text) {
    std::string_view result = text;
    while (!result.empty()) {
      const char ch = result.back();
      if (ch == ' ' || ch == '\t') {
        result.remove_suffix(1);
      } else if (ch == '\n' || ch == '\r') {
        break;
      } else {
        return text;
      }
    }
    return result;
  }

  static std::string_view clear_whitespace_suffix(std::string_view text) {
    while (!text.empty()) {
      const char ch = text.back();
      if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') {
        break;
      }
      text.remove_suffix(1);
    }
    return text;
  }

public:
  explicit Lexer(const LexerConfig& config): config(config), state(State::Text), minus_state(MinusState::Number), tok_start(0), pos(0) {}

  SourceLocation current_position() const {
    return get_source_location(m_in, tok_start);
  }

  void start(std::string_view input) {
    m_in = input;
    tok_start = 0;
    pos = 0;
    state = State::Text;
    minus_state = MinusState::Number;

    // Consume byte order mark (BOM) for UTF-8
    if (inja::string_view::starts_with(m_in, "\xEF\xBB\xBF")) {
      m_in = m_in.substr(3);
    }
  }

  Token scan() {
    tok_start = pos;

  again:
    if (tok_start >= m_in.size()) {
      return make_token(Token::Kind::Eof);
    }

    switch (state) {
    default:
    case State::Text: {
      // fast-scan to first open character
      const size_t open_start = m_in.substr(pos).find_first_of(config.open_chars);
      if (open_start == std::string_view::npos) {
        // didn't find open, return remaining text as text token
        pos = m_in.size();
        return make_token(Token::Kind::Text);
      }
      pos += open_start;

      // try to match one of the opening sequences, and get the close
      const std::string_view open_str = m_in.substr(pos);
      bool must_lstrip = false;
      bool force_lstrip = false;
      if (inja::string_view::starts_with(open_str, config.expression_open)) {
        if (inja::string_view::starts_with(open_str, config.expression_open_force_lstrip)) {
          state = State::ExpressionStartForceLstrip;
          must_lstrip = true;
          force_lstrip = true;
        } else {
          state = State::ExpressionStart;
        }
      } else if (inja::string_view::starts_with(open_str, config.statement_open)) {
        if (inja::string_view::starts_with(open_str, config.statement_open_no_lstrip)) {
          state = State::StatementStartNoLstrip;
        } else if (inja::string_view::starts_with(open_str, config.statement_open_force_lstrip)) {
          state = State::StatementStartForceLstrip;
          must_lstrip = true;
          force_lstrip = true;
        } else {
          state = State::StatementStart;
          must_lstrip = config.lstrip_blocks;
        }
      } else if (inja::string_view::starts_with(open_str, config.comment_open)) {
        if (inja::string_view::starts_with(open_str, config.comment_open_force_lstrip)) {
          state = State::CommentStartForceLstrip;
          must_lstrip = true;
          force_lstrip = true;
        } else {
          state = State::CommentStart;
          must_lstrip = config.lstrip_blocks;
        }
      } else if (!config.line_statement.empty() && (pos == 0 || m_in[pos - 1] == '\n') && inja::string_view::starts_with(open_str, config.line_statement)) {
        state = State::LineStart;
      } else {
        pos += 1; // wasn't actually an opening sequence
        goto again;
      }

      std::string_view text = string_view::slice(m_in, tok_start, pos);
      if (must_lstrip) {
        text = force_lstrip ? clear_whitespace_suffix(text) : clear_final_line_if_whitespace(text);
      }

      if (text.empty()) {
        goto again; // don't generate empty token
      }
      return Token(Token::Kind::Text, text);
    }
    case State::ExpressionStart: {
      state = State::ExpressionBody;
      pos += config.expression_open.size();
      return make_token(Token::Kind::ExpressionOpen);
    }
    case State::ExpressionStartForceLstrip: {
      state = State::ExpressionBody;
      pos += config.expression_open_force_lstrip.size();
      return make_token(Token::Kind::ExpressionOpen);
    }
    case State::LineStart: {
      state = State::LineBody;
      pos += config.line_statement.size();
      return make_token(Token::Kind::LineStatementOpen);
    }
    case State::StatementStart: {
      state = State::StatementBody;
      pos += config.statement_open.size();
      return make_token(Token::Kind::StatementOpen);
    }
    case State::StatementStartNoLstrip: {
      state = State::StatementBody;
      pos += config.statement_open_no_lstrip.size();
      return make_token(Token::Kind::StatementOpen);
    }
    case State::StatementStartForceLstrip: {
      state = State::StatementBody;
      pos += config.statement_open_force_lstrip.size();
      return make_token(Token::Kind::StatementOpen);
    }
    case State::CommentStart: {
      state = State::CommentBody;
      pos += config.comment_open.size();
      return make_token(Token::Kind::CommentOpen);
    }
    case State::CommentStartForceLstrip: {
      state = State::CommentBody;
      pos += config.comment_open_force_lstrip.size();
      return make_token(Token::Kind::CommentOpen);
    }
    case State::ExpressionBody:
      return scan_body(config.expression_close, Token::Kind::ExpressionClose, config.expression_close_force_rstrip);
    case State::LineBody:
      return scan_body("\n", Token::Kind::LineStatementClose);
    case State::StatementBody:
      return scan_body(config.statement_close, Token::Kind::StatementClose, config.statement_close_force_rstrip, config.trim_blocks);
    case State::CommentBody: {
      // fast-scan to comment close
      const size_t end = m_in.substr(pos).find(config.comment_close);
      if (end == std::string_view::npos) {
        pos = m_in.size();
        return make_token(Token::Kind::Eof);
      }

      // Check for trim pattern
      const bool must_rstrip = inja::string_view::starts_with(m_in.substr(pos + end - 1), config.comment_close_force_rstrip);

      // return the entire comment in the close token
      state = State::Text;
      pos += end + config.comment_close.size();
      Token tok = make_token(Token::Kind::CommentClose);

      if (must_rstrip || config.trim_blocks) {
        skip_whitespaces_and_first_newline();
      }
      return tok;
    }
    }
  }

  const LexerConfig& get_config() const {
    return config;
  }
};

} // namespace inja

#endif // INCLUDE_INJA_LEXER_HPP_

// #include "node.hpp"

// #include "template.hpp"

// #include "throw.hpp"

// #include "token.hpp"


namespace inja {

/*!
 * \brief Class for parsing an inja Template.
 */
class Parser {
  using Arguments = std::vector<std::shared_ptr<ExpressionNode>>;
  using OperatorStack = std::stack<std::shared_ptr<FunctionNode>>;

  const ParserConfig& config;

  Lexer lexer;
  TemplateStorage& template_storage;
  const FunctionStorage& function_storage;

  Token tok, peek_tok;
  bool have_peek_tok {false};

  std::string_view literal_start;

  BlockNode* current_block {nullptr};
  ExpressionListNode* current_expression_list {nullptr};

  std::stack<IfStatementNode*> if_statement_stack;
  std::stack<ForStatementNode*> for_statement_stack;
  std::stack<BlockStatementNode*> block_statement_stack;
  std::stack<SetBlockStatementNode*> set_block_statement_stack;
  std::stack<MacroStatementNode*> macro_statement_stack;
  std::vector<std::pair<std::string, size_t>> unresolved_functions;
  size_t generation_depth {0};

  void throw_parser_error(const std::string& message) const {
    INJA_THROW(ParserError(message, lexer.current_position()));
  }

  void get_next_token() {
    if (have_peek_tok) {
      tok = peek_tok;
      have_peek_tok = false;
    } else {
      tok = lexer.scan();
    }
  }

  void get_peek_token() {
    if (!have_peek_tok) {
      peek_tok = lexer.scan();
      have_peek_tok = true;
    }
  }

  void add_literal(Arguments &arguments, const char* content_ptr) {
    const std::string_view data_text(literal_start.data(), tok.text.data() - literal_start.data() + tok.text.size());
    arguments.emplace_back(std::make_shared<LiteralNode>(data_text, data_text.data() - content_ptr));
  }

  void add_operator(Arguments &arguments, OperatorStack &operator_stack) {
    auto function = operator_stack.top();
    operator_stack.pop();

    if (static_cast<int>(arguments.size()) < function->number_args) {
      throw_parser_error("too few arguments");
    }

    for (int i = 0; i < function->number_args; ++i) {
      function->arguments.insert(function->arguments.begin(), arguments.back());
      arguments.pop_back();
    }
    arguments.emplace_back(function);
  }

  void add_to_template_storage(const std::filesystem::path& path, std::string& template_name) {
    if (template_storage.find(template_name) != template_storage.end()) {
      return;
    }

    const std::string original_name = template_name;

    if (config.search_included_templates_in_files) {
      // Build the relative path
      template_name = (path / original_name).string();
      if (template_name.compare(0, 2, "./") == 0) {
        template_name.erase(0, 2);
      }

      if (template_storage.find(template_name) == template_storage.end()) {
        // Load file
        std::ifstream file;
        file.open(template_name);
        if (!file.fail()) {
          const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

          auto include_template = Template(text);
          template_storage.emplace(template_name, include_template);
          parse_into_template(template_storage[template_name], template_name);
          return;
        } else if (!config.include_callback) {
          INJA_THROW(FileError("failed accessing file at '" + template_name + "'"));
        }
      }
    }

    // Try include callback
    if (config.include_callback) {
      auto include_template = config.include_callback(path, original_name);
      template_storage.emplace(template_name, include_template);
    }
  }

  std::string parse_filename() const {
    if (tok.kind != Token::Kind::String) {
      throw_parser_error("expected string, got '" + tok.describe() + "'");
    }

    if (tok.text.length() < 2) {
      throw_parser_error("expected filename, got '" + static_cast<std::string>(tok.text) + "'");
    }

    // Remove first and last character ""
    return std::string {tok.text.substr(1, tok.text.length() - 2)};
  }

  bool parse_expression(Template& tmpl, Token::Kind closing) {
    current_expression_list->root = parse_expression(tmpl);
    return tok.kind == closing;
  }

  void resolve_function(const std::shared_ptr<FunctionNode>& func) {
    const auto function_data = function_storage.find_function(func->name, func->number_args);
    if (function_data.operation == FunctionStorage::Operation::None) {
      func->operation = FunctionStorage::Operation::MacroCall;
      unresolved_functions.emplace_back(func->name, func->pos);
      return;
    }
    func->operation = function_data.operation;
    if (function_data.operation == FunctionStorage::Operation::Callback) {
      func->callback = function_data.callback;
    }
  }

  void parse_call_arguments(Template& tmpl, const std::shared_ptr<FunctionNode>& func) {
    do {
      get_next_token();
      if (tok.kind == Token::Kind::RightParen) {
        break;
      }

      std::string argument_name;
      size_t argument_pos = tok.text.data() - tmpl.content.c_str();
      if (tok.kind == Token::Kind::Id) {
        get_peek_token();
        if (peek_tok.text == "=") {
          argument_name = static_cast<std::string>(tok.text);
          get_next_token();
          get_next_token();
        }
      }

      auto expression = parse_expression(tmpl);
      if (!expression) {
        throw_parser_error("expected function argument");
      }
      if (func->name == "namespace" && !argument_name.empty()) {
        func->number_args += 2;
        func->arguments.emplace_back(std::make_shared<LiteralNode>(json(argument_name), argument_pos));
        func->argument_names.emplace_back();
        func->arguments.emplace_back(std::move(expression));
        func->argument_names.emplace_back();
      } else {
        ++func->number_args;
        func->arguments.emplace_back(std::move(expression));
        func->argument_names.emplace_back(std::move(argument_name));
      }
    } while (tok.kind == Token::Kind::Comma);

    if (tok.kind != Token::Kind::RightParen) {
      throw_parser_error("expected right parenthesis, got '" + tok.describe() + "'");
    }
  }

  std::shared_ptr<ExpressionNode> parse_array(Template& tmpl, size_t position) {
    auto result = std::make_shared<FunctionNode>(FunctionStorage::Operation::Array, position);
    result->number_args = 0;
    get_next_token();
    while (tok.kind != Token::Kind::RightBracket) {
      auto element = parse_expression(tmpl);
      if (!element) {
        throw_parser_error("expected array element");
      }
      result->arguments.emplace_back(std::move(element));
      ++result->number_args;
      if (tok.kind == Token::Kind::Comma) {
        get_next_token();
        if (tok.kind == Token::Kind::RightBracket) {
          break;
        }
      } else if (tok.kind != Token::Kind::RightBracket) {
        throw_parser_error("expected ',' or ']' in array literal");
      }
    }
    return result;
  }

  std::shared_ptr<ExpressionNode> parse_object(Template& tmpl, size_t position) {
    auto result = std::make_shared<FunctionNode>(FunctionStorage::Operation::Object, position);
    result->number_args = 0;
    get_next_token();
    while (tok.kind != Token::Kind::RightBrace) {
      auto key = parse_expression(tmpl);
      if (!key || tok.kind != Token::Kind::Colon) {
        throw_parser_error("expected object key followed by ':'");
      }
      get_next_token();
      auto value = parse_expression(tmpl);
      if (!value) {
        throw_parser_error("expected object value");
      }
      result->arguments.emplace_back(std::move(key));
      result->arguments.emplace_back(std::move(value));
      result->number_args += 2;
      if (tok.kind == Token::Kind::Comma) {
        get_next_token();
        if (tok.kind == Token::Kind::RightBrace) {
          break;
        }
      } else if (tok.kind != Token::Kind::RightBrace) {
        throw_parser_error("expected ',' or '}' in object literal");
      }
    }
    return result;
  }

  std::shared_ptr<ExpressionNode> parse_expression(Template& tmpl, std::string_view stop_id = {}) {
    Arguments arguments;
    OperatorStack operator_stack;

    while (tok.kind != Token::Kind::Eof) {
      if (!stop_id.empty() && tok.kind == Token::Kind::Id && tok.text == stop_id) {
        goto break_loop;
      }
      // Literals
      switch (tok.kind) {
      case Token::Kind::String: {
        literal_start = tok.text;
        auto literal = std::make_shared<LiteralNode>(tok.text, tok.text.data() - tmpl.content.c_str());
        const auto previous_literal = arguments.empty() ? nullptr : std::dynamic_pointer_cast<LiteralNode>(arguments.back());
        const auto previous_concat = arguments.empty() ? nullptr : std::dynamic_pointer_cast<FunctionNode>(arguments.back());
        const bool adjacent_string = operator_stack.empty() &&
            ((previous_literal && previous_literal->value.is_string()) ||
             (previous_concat && previous_concat->operation == FunctionStorage::Operation::Concat));
        if (adjacent_string) {
          auto concat = std::make_shared<FunctionNode>(FunctionStorage::Operation::Concat, literal->pos);
          concat->arguments.emplace_back(arguments.back());
          concat->arguments.emplace_back(std::move(literal));
          arguments.back() = std::move(concat);
        } else {
          arguments.emplace_back(std::move(literal));
        }
      } break;
      case Token::Kind::Number: {
        literal_start = tok.text;
        add_literal(arguments, tmpl.content.c_str());
      } break;
      case Token::Kind::LeftBracket: {
        const auto bracket_pos = static_cast<size_t>(tok.text.data() - tmpl.content.c_str());
        const auto data = arguments.empty() ? nullptr : std::dynamic_pointer_cast<DataNode>(arguments.back());
        const auto function = arguments.empty() ? nullptr : std::dynamic_pointer_cast<FunctionNode>(arguments.back());
        const bool adjacent = bracket_pos > 0 && std::isspace(static_cast<unsigned char>(tmpl.content[bracket_pos - 1])) == 0;
        const bool follows_value = (data && data->pos + data->name.size() == bracket_pos) || (function && adjacent);
        if (follows_value) {
          auto container = arguments.back();
          arguments.pop_back();
          get_next_token();
          auto const null_value = [&] { return std::make_shared<LiteralNode>(json(nullptr), bracket_pos); };
          std::shared_ptr<ExpressionNode> first;
          if (tok.kind != Token::Kind::Colon && tok.kind != Token::Kind::RightBracket) {
            first = parse_expression(tmpl);
          }
          if (tok.kind == Token::Kind::Colon) {
            auto slice = std::make_shared<FunctionNode>(FunctionStorage::Operation::Slice, bracket_pos);
            slice->arguments.emplace_back(std::move(container));
            slice->arguments.emplace_back(first ? std::move(first) : null_value());
            get_next_token();
            std::shared_ptr<ExpressionNode> stop;
            if (tok.kind != Token::Kind::Colon && tok.kind != Token::Kind::RightBracket) {
              stop = parse_expression(tmpl);
            }
            slice->arguments.emplace_back(stop ? std::move(stop) : null_value());
            if (tok.kind == Token::Kind::Colon) {
              get_next_token();
              std::shared_ptr<ExpressionNode> step;
              if (tok.kind != Token::Kind::RightBracket) {
                step = parse_expression(tmpl);
              }
              slice->arguments.emplace_back(step ? std::move(step) : null_value());
            } else {
              slice->arguments.emplace_back(null_value());
            }
            if (tok.kind != Token::Kind::RightBracket) {
              throw_parser_error("expected closing bracket after slice");
            }
            arguments.emplace_back(std::move(slice));
            break;
          }
          if (!first || tok.kind != Token::Kind::RightBracket) {
            throw_parser_error("expected index and closing bracket");
          }
          auto access = std::make_shared<FunctionNode>(FunctionStorage::Operation::At, bracket_pos);
          access->arguments.emplace_back(std::move(container));
          access->arguments.emplace_back(std::move(first));
          arguments.emplace_back(std::move(access));
          break;
        }
        arguments.emplace_back(parse_array(tmpl, bracket_pos));
      } break;
      case Token::Kind::LeftBrace: {
        arguments.emplace_back(parse_object(tmpl, tok.text.data() - tmpl.content.c_str()));
      } break;
      case Token::Kind::RightBracket: {
        goto break_loop;
      } break;
      case Token::Kind::RightBrace: {
        goto break_loop;
      } break;
      case Token::Kind::Id: {
        get_peek_token();

        // Data Literal
        if (tok.text == static_cast<decltype(tok.text)>("true") || tok.text == static_cast<decltype(tok.text)>("false") ||
            tok.text == static_cast<decltype(tok.text)>("null") || tok.text == static_cast<decltype(tok.text)>("none") ||
            tok.text == static_cast<decltype(tok.text)>("True") || tok.text == static_cast<decltype(tok.text)>("False") ||
            tok.text == static_cast<decltype(tok.text)>("None")) {
          literal_start = tok.text;
          add_literal(arguments, tmpl.content.c_str());

          // Operator
        } else if (tok.text == "is") {
          while (!operator_stack.empty() && operator_stack.top()->operation != FunctionStorage::Operation::Not &&
                 operator_stack.top()->precedence > 2) {
            add_operator(arguments, operator_stack);
          }
          if (arguments.empty()) {
            throw_parser_error("missing value before 'is'");
          }
          get_next_token();
          bool negate = false;
          if (tok.kind == Token::Kind::Id && tok.text == "not") {
            negate = true;
            get_next_token();
          }
          if (tok.kind != Token::Kind::Id) {
            throw_parser_error("expected test name after 'is'");
          }
          const std::map<std::string_view, FunctionStorage::Operation> tests {
              {"defined", FunctionStorage::Operation::IsDefined},
              {"undefined", FunctionStorage::Operation::IsUndefined},
              {"none", FunctionStorage::Operation::IsNone},
              {"iterable", FunctionStorage::Operation::IsIterable},
              {"mapping", FunctionStorage::Operation::IsMapping},
              {"sequence", FunctionStorage::Operation::IsSequence},
              {"string", FunctionStorage::Operation::IsString},
              {"boolean", FunctionStorage::Operation::IsBoolean},
              {"number", FunctionStorage::Operation::IsNumber},
              {"integer", FunctionStorage::Operation::IsInteger},
              {"true", FunctionStorage::Operation::IsTrue},
              {"false", FunctionStorage::Operation::IsFalse},
          };
          const auto test = tests.find(tok.text);
          if (test == tests.end()) {
            throw_parser_error("unknown test '" + static_cast<std::string>(tok.text) + "'");
          }
          auto test_node = std::make_shared<FunctionNode>(test->second, tok.text.data() - tmpl.content.c_str());
          test_node->arguments.emplace_back(arguments.back());
          arguments.pop_back();
          if (negate) {
            auto not_node = std::make_shared<FunctionNode>(FunctionStorage::Operation::Not, tok.text.data() - tmpl.content.c_str());
            not_node->arguments.emplace_back(std::move(test_node));
            arguments.emplace_back(std::move(not_node));
          } else {
            arguments.emplace_back(std::move(test_node));
          }
        } else if (tok.text == "if" && !arguments.empty()) {
          while (!operator_stack.empty()) {
            add_operator(arguments, operator_stack);
          }
          if (arguments.size() != 1) {
            throw_parser_error("malformed conditional expression");
          }
          auto true_value = arguments.back();
          arguments.pop_back();
          get_next_token();
          auto condition = parse_expression(tmpl, "else");
          if (!condition) {
            throw_parser_error("conditional expression requires a condition");
          }
          std::shared_ptr<ExpressionNode> false_value;
          if (tok.kind == Token::Kind::Id && tok.text == "else") {
            get_next_token();
            false_value = parse_expression(tmpl);
            if (!false_value) {
              throw_parser_error("conditional expression requires a false value");
            }
          } else {
            false_value = std::make_shared<LiteralNode>(json(nullptr), true_value->pos);
          }
          auto conditional = std::make_shared<FunctionNode>(FunctionStorage::Operation::Conditional, true_value->pos);
          conditional->arguments.emplace_back(std::move(condition));
          conditional->arguments.emplace_back(std::move(true_value));
          conditional->arguments.emplace_back(std::move(false_value));
          arguments.emplace_back(std::move(conditional));
          goto break_loop;
        } else if (tok.text == "not" && peek_tok.kind == Token::Kind::Id && peek_tok.text == "in") {
          auto function_node = std::make_shared<FunctionNode>(FunctionStorage::Operation::NotIn, tok.text.data() - tmpl.content.c_str());
          while (!operator_stack.empty() && operator_stack.top()->precedence >= function_node->precedence) {
            add_operator(arguments, operator_stack);
          }
          operator_stack.emplace(std::move(function_node));
          get_next_token();
        } else if (tok.text == "and" || tok.text == "or" || tok.text == "in" || tok.text == "not") {
          goto parse_operator;

          // Functions
        } else if (peek_tok.kind == Token::Kind::LeftParen) {
          auto const call_pos = tok.text.data() - tmpl.content.c_str();
          std::string callable(tok.text);
          auto const dot = callable.rfind('.');
          auto func = std::make_shared<FunctionNode>(
              dot == std::string::npos ? callable : callable.substr(dot + 1), call_pos);
          if (dot != std::string::npos) {
            func->arguments.emplace_back(std::make_shared<DataNode>(callable.substr(0, dot), call_pos));
            func->argument_names.emplace_back();
            ++func->number_args;
          }
          get_next_token();
          parse_call_arguments(tmpl, func);

          resolve_function(func);
          arguments.emplace_back(func);

          // Variables
        } else {
          arguments.emplace_back(std::make_shared<DataNode>(static_cast<std::string>(tok.text), tok.text.data() - tmpl.content.c_str()));
        }

        // Operators
      } break;
      case Token::Kind::Equal:
      case Token::Kind::NotEqual:
      case Token::Kind::GreaterThan:
      case Token::Kind::GreaterEqual:
      case Token::Kind::LessThan:
      case Token::Kind::LessEqual:
      case Token::Kind::Plus:
      case Token::Kind::Minus:
      case Token::Kind::Times:
      case Token::Kind::Slash:
      case Token::Kind::Power:
      case Token::Kind::Percent:
      case Token::Kind::Tilde:
      case Token::Kind::Dot: {

        if (tok.kind == Token::Kind::Dot && !arguments.empty()) {
          auto container = arguments.back();
          arguments.pop_back();
          get_next_token();
          if (tok.kind != Token::Kind::Id) {
            throw_parser_error("expected member name after '.'");
          }
          const auto member_pos = tok.text.data() - tmpl.content.c_str();
          const std::string member(tok.text);
          get_peek_token();
          if (peek_tok.kind == Token::Kind::LeftParen) {
            auto func = std::make_shared<FunctionNode>(member, member_pos);
            func->arguments.emplace_back(std::move(container));
            func->argument_names.emplace_back();
            ++func->number_args;
            get_next_token();
            parse_call_arguments(tmpl, func);
            resolve_function(func);
            arguments.emplace_back(std::move(func));
          } else {
            auto access = std::make_shared<FunctionNode>(FunctionStorage::Operation::At, member_pos);
            access->arguments.emplace_back(std::move(container));
            access->arguments.emplace_back(std::make_shared<LiteralNode>(json(member), member_pos));
            arguments.emplace_back(std::move(access));
          }
          break;
        }

      parse_operator:
        FunctionStorage::Operation operation;
        switch (tok.kind) {
        case Token::Kind::Id: {
          if (tok.text == "and") {
            operation = FunctionStorage::Operation::And;
          } else if (tok.text == "or") {
            operation = FunctionStorage::Operation::Or;
          } else if (tok.text == "in") {
            operation = FunctionStorage::Operation::In;
          } else if (tok.text == "not") {
            operation = FunctionStorage::Operation::Not;
          } else {
            throw_parser_error("unknown operator in parser.");
          }
        } break;
        case Token::Kind::Equal: {
          operation = FunctionStorage::Operation::Equal;
        } break;
        case Token::Kind::NotEqual: {
          operation = FunctionStorage::Operation::NotEqual;
        } break;
        case Token::Kind::GreaterThan: {
          operation = FunctionStorage::Operation::Greater;
        } break;
        case Token::Kind::GreaterEqual: {
          operation = FunctionStorage::Operation::GreaterEqual;
        } break;
        case Token::Kind::LessThan: {
          operation = FunctionStorage::Operation::Less;
        } break;
        case Token::Kind::LessEqual: {
          operation = FunctionStorage::Operation::LessEqual;
        } break;
        case Token::Kind::Plus: {
          operation = FunctionStorage::Operation::Add;
        } break;
        case Token::Kind::Tilde: {
          operation = FunctionStorage::Operation::Concat;
        } break;
        case Token::Kind::Minus: {
          operation = FunctionStorage::Operation::Subtract;
        } break;
        case Token::Kind::Times: {
          operation = FunctionStorage::Operation::Multiplication;
        } break;
        case Token::Kind::Slash: {
          operation = FunctionStorage::Operation::Division;
        } break;
        case Token::Kind::Power: {
          operation = FunctionStorage::Operation::Power;
        } break;
        case Token::Kind::Percent: {
          operation = FunctionStorage::Operation::Modulo;
        } break;
        case Token::Kind::Dot: {
          operation = FunctionStorage::Operation::AtId;
        } break;
        default: {
          throw_parser_error("unknown operator in parser.");
        }
        }
        auto function_node = std::make_shared<FunctionNode>(operation, tok.text.data() - tmpl.content.c_str());

        while (!operator_stack.empty() &&
               ((operator_stack.top()->precedence > function_node->precedence) ||
                (operator_stack.top()->precedence == function_node->precedence && function_node->associativity == FunctionNode::Associativity::Left))) {
          add_operator(arguments, operator_stack);
        }

        operator_stack.emplace(function_node);
      } break;
      case Token::Kind::Comma: {
        goto break_loop;
      } break;
      case Token::Kind::Colon: {
        goto break_loop;
      } break;
      case Token::Kind::LeftParen: {
        const auto position = tok.text.data() - tmpl.content.c_str();
        get_next_token();
        auto expr = parse_expression(tmpl);
        if (!expr) {
          throw_parser_error("empty expression in parentheses");
        }
        if (tok.kind == Token::Kind::Comma) {
          auto tuple = std::make_shared<FunctionNode>(FunctionStorage::Operation::Array, position);
          tuple->number_args = 1;
          tuple->arguments.emplace_back(std::move(expr));
          while (tok.kind == Token::Kind::Comma) {
            get_next_token();
            if (tok.kind == Token::Kind::RightParen) {
              break;
            }
            auto element = parse_expression(tmpl);
            if (!element) {
              throw_parser_error("expected tuple element");
            }
            tuple->arguments.emplace_back(std::move(element));
            ++tuple->number_args;
          }
          arguments.emplace_back(std::move(tuple));
        } else {
          arguments.emplace_back(std::move(expr));
        }
        if (tok.kind != Token::Kind::RightParen) {
          throw_parser_error("expected right parenthesis, got '" + tok.describe() + "'");
        }
      } break;

      // parse function call pipe syntax
      case Token::Kind::Pipe: {
        // get function name
        get_next_token();
        if (tok.kind != Token::Kind::Id) {
          throw_parser_error("expected function name, got '" + tok.describe() + "'");
        }
        auto func = std::make_shared<FunctionNode>(tok.text, tok.text.data() - tmpl.content.c_str());
        // add first parameter as last value from arguments
        func->number_args += 1;
        func->arguments.emplace_back(arguments.back());
        func->argument_names.emplace_back();
        arguments.pop_back();
        get_peek_token();
        if (peek_tok.kind == Token::Kind::LeftParen) {
          get_next_token();
          parse_call_arguments(tmpl, func);
        }
        resolve_function(func);
        arguments.emplace_back(func);
      } break;
      default:
        goto break_loop;
      }

      get_next_token();
    }

  break_loop:
    while (!operator_stack.empty()) {
      add_operator(arguments, operator_stack);
    }

    std::shared_ptr<ExpressionNode> expr;
    if (arguments.size() == 1) {
      expr = arguments[0];
      arguments = {};
    } else if (arguments.size() > 1) {
      throw_parser_error("malformed expression");
    }
    return expr;
  }

  bool parse_statement(Template& tmpl, Token::Kind closing, const std::filesystem::path& path) {
    if (tok.kind != Token::Kind::Id) {
      return false;
    }

    if (tok.text == static_cast<decltype(tok.text)>("macro")) {
      get_next_token();
      if (tok.kind != Token::Kind::Id) {
        throw_parser_error("expected macro name, got '" + tok.describe() + "'");
      }
      const std::string name(tok.text);
      const auto position = tok.text.data() - tmpl.content.c_str();
      auto macro = std::make_shared<MacroStatementNode>(name, current_block, position);
      if (!tmpl.macro_storage.emplace(name, macro).second) {
        throw_parser_error("macro with the name '" + name + "' already exists");
      }
      current_block->nodes.emplace_back(macro);
      get_next_token();
      if (tok.kind != Token::Kind::LeftParen) {
        throw_parser_error("expected '(' after macro name");
      }
      get_next_token();
      bool saw_default = false;
      while (tok.kind != Token::Kind::RightParen) {
        if (tok.kind != Token::Kind::Id) {
          throw_parser_error("expected macro parameter, got '" + tok.describe() + "'");
        }
        MacroStatementNode::Parameter parameter{static_cast<std::string>(tok.text), nullptr};
        get_next_token();
        if (tok.text == "=") {
          saw_default = true;
          get_next_token();
          parameter.default_value = parse_expression(tmpl);
          if (!parameter.default_value) {
            throw_parser_error("expected default value for macro parameter '" + parameter.name + "'");
          }
        } else if (saw_default) {
          throw_parser_error("non-default macro parameter follows default parameter");
        }
        macro->parameters.emplace_back(std::move(parameter));
        if (tok.kind == Token::Kind::Comma) {
          get_next_token();
        } else if (tok.kind != Token::Kind::RightParen) {
          throw_parser_error("expected ',' or ')' in macro declaration");
        }
      }
      get_next_token();
      macro_statement_stack.emplace(macro.get());
      current_block = &macro->body;
    } else if (tok.text == static_cast<decltype(tok.text)>("endmacro")) {
      if (macro_statement_stack.empty()) {
        throw_parser_error("endmacro without matching macro");
      }
      auto& macro = macro_statement_stack.top();
      get_next_token();
      current_block = macro->parent;
      macro_statement_stack.pop();
    } else if (tok.text == static_cast<decltype(tok.text)>("if")) {
      get_next_token();

      auto if_statement_node = std::make_shared<IfStatementNode>(current_block, tok.text.data() - tmpl.content.c_str());
      current_block->nodes.emplace_back(if_statement_node);
      if_statement_stack.emplace(if_statement_node.get());
      current_block = &if_statement_node->true_statement;
      current_expression_list = &if_statement_node->condition;

      if (!parse_expression(tmpl, closing)) {
        return false;
      }
    } else if (tok.text == static_cast<decltype(tok.text)>("else") || tok.text == static_cast<decltype(tok.text)>("elif")) {
      if (if_statement_stack.empty()) {
        throw_parser_error("else/elif without matching if");
      }
      const bool is_elif = tok.text == static_cast<decltype(tok.text)>("elif");
      auto& if_statement_data = if_statement_stack.top();
      get_next_token();

      if_statement_data->has_false_statement = true;
      current_block = &if_statement_data->false_statement;

      // Chained {% else if ... %} or Jinja's {% elif ... %}.
      if (is_elif || (tok.kind == Token::Kind::Id && tok.text == static_cast<decltype(tok.text)>("if"))) {
        if (!is_elif) {
          get_next_token();
        }

        auto if_statement_node = std::make_shared<IfStatementNode>(true, current_block, tok.text.data() - tmpl.content.c_str());
        current_block->nodes.emplace_back(if_statement_node);
        if_statement_stack.emplace(if_statement_node.get());
        current_block = &if_statement_node->true_statement;
        current_expression_list = &if_statement_node->condition;

        if (!parse_expression(tmpl, closing)) {
          return false;
        }
      }
    } else if (tok.text == static_cast<decltype(tok.text)>("endif")) {
      if (if_statement_stack.empty()) {
        throw_parser_error("endif without matching if");
      }

      // Nested if statements
      while (if_statement_stack.top()->is_nested) {
        if_statement_stack.pop();
      }

      auto& if_statement_data = if_statement_stack.top();
      get_next_token();

      current_block = if_statement_data->parent;
      if_statement_stack.pop();
    } else if (tok.text == static_cast<decltype(tok.text)>("block")) {
      get_next_token();

      if (tok.kind != Token::Kind::Id) {
        throw_parser_error("expected block name, got '" + tok.describe() + "'");
      }

      const std::string block_name = static_cast<std::string>(tok.text);

      auto block_statement_node = std::make_shared<BlockStatementNode>(current_block, block_name, tok.text.data() - tmpl.content.c_str());
      current_block->nodes.emplace_back(block_statement_node);
      block_statement_stack.emplace(block_statement_node.get());
      current_block = &block_statement_node->block;
      auto success = tmpl.block_storage.emplace(block_name, block_statement_node);
      if (!success.second) {
        throw_parser_error("block with the name '" + block_name + "' does already exist");
      }

      get_next_token();
    } else if (tok.text == static_cast<decltype(tok.text)>("endblock")) {
      if (block_statement_stack.empty()) {
        throw_parser_error("endblock without matching block");
      }

      auto& block_statement_data = block_statement_stack.top();
      get_next_token();

      current_block = block_statement_data->parent;
      block_statement_stack.pop();
    } else if (tok.text == static_cast<decltype(tok.text)>("for")) {
      get_next_token();

      // options: for a in arr; for a, b in obj
      if (tok.kind != Token::Kind::Id) {
        throw_parser_error("expected id, got '" + tok.describe() + "'");
      }

      Token value_token = tok;
      get_next_token();

      // Object type
      std::shared_ptr<ForStatementNode> for_statement_node;
      if (tok.kind == Token::Kind::Comma) {
        get_next_token();
        if (tok.kind != Token::Kind::Id) {
          throw_parser_error("expected id, got '" + tok.describe() + "'");
        }

        const Token key_token = value_token;
        value_token = tok;
        get_next_token();

        for_statement_node = std::make_shared<ForObjectStatementNode>(static_cast<std::string>(key_token.text), static_cast<std::string>(value_token.text),
                                                                      current_block, tok.text.data() - tmpl.content.c_str());

        // Array type
      } else {
        for_statement_node =
            std::make_shared<ForArrayStatementNode>(static_cast<std::string>(value_token.text), current_block, tok.text.data() - tmpl.content.c_str());
      }

      current_block->nodes.emplace_back(for_statement_node);
      for_statement_stack.emplace(for_statement_node.get());
      current_block = &for_statement_node->body;
      if (tok.kind != Token::Kind::Id || tok.text != static_cast<decltype(tok.text)>("in")) {
        throw_parser_error("expected 'in', got '" + tok.describe() + "'");
      }
      get_next_token();

      for_statement_node->condition.root = parse_expression(tmpl, "if");
      if (!for_statement_node->condition.root) {
        throw_parser_error("for loop requires an iterable expression");
      }
      if (tok.kind == Token::Kind::Id && tok.text == "if") {
        for_statement_node->has_filter = true;
        get_next_token();
        for_statement_node->filter.root = parse_expression(tmpl);
        if (!for_statement_node->filter.root) {
          throw_parser_error("for-loop filter requires an expression");
        }
      }
      if (tok.kind != closing) {
        return false;
      }
    } else if (tok.text == static_cast<decltype(tok.text)>("endfor")) {
      if (for_statement_stack.empty()) {
        throw_parser_error("endfor without matching for");
      }

      auto& for_statement_data = for_statement_stack.top();
      get_next_token();

      current_block = for_statement_data->parent;
      for_statement_stack.pop();
    } else if (tok.text == static_cast<decltype(tok.text)>("include")) {
      get_next_token();

      std::string template_name = parse_filename();
      add_to_template_storage(path, template_name);

      current_block->nodes.emplace_back(std::make_shared<IncludeStatementNode>(template_name, tok.text.data() - tmpl.content.c_str()));

      get_next_token();
    } else if (tok.text == static_cast<decltype(tok.text)>("extends")) {
      get_next_token();

      std::string template_name = parse_filename();
      add_to_template_storage(path, template_name);

      current_block->nodes.emplace_back(std::make_shared<ExtendsStatementNode>(template_name, tok.text.data() - tmpl.content.c_str()));

      get_next_token();
    } else if (tok.text == static_cast<decltype(tok.text)>("set")) {
      get_next_token();

      if (tok.kind != Token::Kind::Id) {
        throw_parser_error("expected variable name, got '" + tok.describe() + "'");
      }

      const std::string key = static_cast<std::string>(tok.text);
      get_next_token();

      if (tok.kind == closing) {
        auto set_block = std::make_shared<SetBlockStatementNode>(key, current_block, tok.text.data() - tmpl.content.c_str());
        current_block->nodes.emplace_back(set_block);
        set_block_statement_stack.emplace(set_block.get());
        current_block = &set_block->body;
        return true;
      }
      auto set_statement_node = std::make_shared<SetStatementNode>(key, tok.text.data() - tmpl.content.c_str());
      current_block->nodes.emplace_back(set_statement_node);
      current_expression_list = &set_statement_node->expression;
      if (tok.text != static_cast<decltype(tok.text)>("=")) {
        throw_parser_error("expected '=' or statement close, got '" + tok.describe() + "'");
      }
      get_next_token();

      if (!parse_expression(tmpl, closing)) {
        return false;
      }
    } else if (tok.text == static_cast<decltype(tok.text)>("endset")) {
      if (set_block_statement_stack.empty()) {
        throw_parser_error("endset without matching set");
      }
      auto& set_block = set_block_statement_stack.top();
      get_next_token();
      current_block = set_block->parent;
      set_block_statement_stack.pop();
    } else if (tok.text == static_cast<decltype(tok.text)>("break") ||
               tok.text == static_cast<decltype(tok.text)>("continue")) {
      if (for_statement_stack.empty()) {
        throw_parser_error("loop control outside a for loop");
      }
      auto const control = tok.text == static_cast<decltype(tok.text)>("break")
          ? LoopControlStatementNode::Control::Break
          : LoopControlStatementNode::Control::Continue;
      current_block->nodes.emplace_back(
          std::make_shared<LoopControlStatementNode>(control, tok.text.data() - tmpl.content.c_str()));
      get_next_token();
    } else if (tok.text == static_cast<decltype(tok.text)>("generation")) {
      ++generation_depth;
      get_next_token();
    } else if (tok.text == static_cast<decltype(tok.text)>("endgeneration")) {
      if (generation_depth == 0) {
        throw_parser_error("endgeneration without matching generation");
      }
      --generation_depth;
      get_next_token();
    } else {
      return false;
    }
    return true;
  }

  void parse_into(Template& tmpl, const std::filesystem::path& path) {
    lexer.start(tmpl.content);
    current_block = &tmpl.root;

    for (;;) {
      get_next_token();
      switch (tok.kind) {
      case Token::Kind::Eof: {
        if (!if_statement_stack.empty()) {
          throw_parser_error("unmatched if");
        }
        if (!for_statement_stack.empty()) {
          throw_parser_error("unmatched for");
        }
        if (!macro_statement_stack.empty()) {
          throw_parser_error("unmatched macro");
        }
        if (!set_block_statement_stack.empty()) {
          throw_parser_error("unmatched set block");
        }
        if (generation_depth != 0) {
          throw_parser_error("unmatched generation block");
        }
        for (const auto& [name, position] : unresolved_functions) {
          if (tmpl.macro_storage.find(name) == tmpl.macro_storage.end()) {
            tok.text = std::string_view(tmpl.content).substr(position, name.size());
            throw_parser_error("unknown function " + name);
          }
        }
      }
        current_block = nullptr;
        return;
      case Token::Kind::Text: {
        current_block->nodes.emplace_back(std::make_shared<TextNode>(tok.text.data() - tmpl.content.c_str(), tok.text.size()));
      } break;
      case Token::Kind::StatementOpen: {
        get_next_token();
        if (!parse_statement(tmpl, Token::Kind::StatementClose, path)) {
          throw_parser_error("expected statement, got '" + tok.describe() + "'");
        }
        if (tok.kind != Token::Kind::StatementClose) {
          throw_parser_error("expected statement close, got '" + tok.describe() + "'");
        }
      } break;
      case Token::Kind::LineStatementOpen: {
        get_next_token();
        if (!parse_statement(tmpl, Token::Kind::LineStatementClose, path)) {
          throw_parser_error("expected statement, got '" + tok.describe() + "'");
        }
        if (tok.kind != Token::Kind::LineStatementClose && tok.kind != Token::Kind::Eof) {
          throw_parser_error("expected line statement close, got '" + tok.describe() + "'");
        }
      } break;
      case Token::Kind::ExpressionOpen: {
        get_next_token();

        auto expression_list_node = std::make_shared<ExpressionListNode>(tok.text.data() - tmpl.content.c_str());
        current_block->nodes.emplace_back(expression_list_node);
        current_expression_list = expression_list_node.get();

        if (!parse_expression(tmpl, Token::Kind::ExpressionClose)) {
          throw_parser_error("expected expression close, got '" + tok.describe() + "'");
        }
      } break;
      case Token::Kind::CommentOpen: {
        get_next_token();
        if (tok.kind != Token::Kind::CommentClose) {
          throw_parser_error("expected comment close, got '" + tok.describe() + "'");
        }
      } break;
      default: {
        throw_parser_error("unexpected token '" + tok.describe() + "'");
      } break;
      }
    }
    current_block = nullptr;
  }

public:
  explicit Parser(const ParserConfig& parser_config, const LexerConfig& lexer_config, TemplateStorage& template_storage,
                  const FunctionStorage& function_storage)
      : config(parser_config), lexer(lexer_config), template_storage(template_storage), function_storage(function_storage) {}

  Template parse(std::string_view input, const std::filesystem::path& path) {
    auto result = Template(std::string(input));
    parse_into(result, path);
    return result;
  }

  void parse_into_template(Template& tmpl, const std::filesystem::path& filename) {
    auto sub_parser = Parser(config, lexer.get_config(), template_storage, function_storage);
    sub_parser.parse_into(tmpl, filename.parent_path());
  }

  static std::string load_file(const std::filesystem::path& filename) {
    std::ifstream file;
    file.open(filename);
    if (file.fail()) {
      INJA_THROW(FileError("failed accessing file at '" + filename.string() + "'"));
    }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return text;
  }
};

} // namespace inja

#endif // INCLUDE_INJA_PARSER_HPP_

// #include "renderer.hpp"
#ifndef INCLUDE_INJA_RENDERER_HPP_
#define INCLUDE_INJA_RENDERER_HPP_

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stack>
#include <string>
#include <utility>
#include <vector>

// #include "config.hpp"

// #include "exceptions.hpp"

// #include "function_storage.hpp"

// #include "node.hpp"

// #include "template.hpp"

// #include "throw.hpp"

// #include "utils.hpp"


namespace inja {

/*!
@brief Escapes HTML
*/
inline std::string htmlescape(const std::string& data) {
  std::string buffer;
  buffer.reserve(static_cast<size_t>(1.1 * data.size()));
  for (size_t pos = 0; pos != data.size(); ++pos) {
    switch (data[pos]) {
      case '&':  buffer.append("&amp;");       break;
      case '\"': buffer.append("&quot;");      break;
      case '\'': buffer.append("&apos;");      break;
      case '<':  buffer.append("&lt;");        break;
      case '>':  buffer.append("&gt;");        break;
      default:   buffer.append(&data[pos], 1); break;
    }
  }
  return buffer;
}

/*!
 * \brief Class for rendering a Template with data.
 */
class Renderer : public NodeVisitor {
  using Op = FunctionStorage::Operation;

  const RenderConfig config;
  const TemplateStorage& template_storage;
  const FunctionStorage& function_storage;

  const Template* current_template;
  size_t current_level {0};
  std::vector<const Template*> template_stack;
  std::vector<const BlockStatementNode*> block_statement_stack;

  const json* data_input;
  std::ostream* output_stream;

  json additional_data;
  json* current_loop_data = &additional_data["loop"];

  std::vector<std::shared_ptr<json>> data_tmp_stack;
  std::stack<const json*> data_eval_stack;
  std::stack<const AstNode*> not_found_stack;

  struct SavedValue {
    bool existed;
    json value;
  };
  std::vector<std::map<std::string, SavedValue>> local_scopes;

  bool break_rendering {false};
  enum class LoopControl { None, Break, Continue };
  LoopControl loop_control {LoopControl::None};

public:
  static constexpr std::string_view object_order_key {"__edgellm_internal_key_order__"};

  static std::vector<std::string> object_keys(const json& value) {
    std::vector<std::string> result;
    const auto order = value.find(std::string(object_order_key));
    if (order != value.end() && order->is_array()) {
      for (const auto& key : *order) {
        if (key.is_string() && value.contains(key.get_ref<const std::string&>())) {
          result.push_back(key.get<std::string>());
        }
      }
    }
    for (auto it = value.begin(); it != value.end(); ++it) {
      if (it.key() != object_order_key && std::find(result.begin(), result.end(), it.key()) == result.end()) {
        result.push_back(it.key());
      }
    }
    return result;
  }

  static size_t visible_size(const json& value) {
    return value.is_object() ? object_keys(value).size() : value.size();
  }

  static std::string python_string_literal(const std::string& value) {
    const char quote = value.find('\'') != std::string::npos && value.find('"') == std::string::npos ? '"' : '\'';
    std::string result(1, quote);
    constexpr char hex[] = "0123456789abcdef";
    for (const unsigned char ch : value) {
      if (ch == '\\') {
        result += "\\\\";
      } else if (ch == static_cast<unsigned char>(quote)) {
        result += '\\';
        result += static_cast<char>(ch);
      } else {
        switch (ch) {
          case '\b': result += "\\b"; break;
          case '\t': result += "\\t"; break;
          case '\n': result += "\\n"; break;
          case '\f': result += "\\f"; break;
          case '\r': result += "\\r"; break;
          default:
            if (ch < 0x20 || ch == 0x7f) {
              result += "\\x";
              result += hex[ch >> 4];
              result += hex[ch & 0x0f];
            } else {
              result += static_cast<char>(ch);
            }
            break;
        }
      }
    }
    result += quote;
    return result;
  }

  static std::string python_repr(const json& value) {
    if (value.is_string()) {
      return python_string_literal(value.get_ref<const json::string_t&>());
    }
    if (value.is_boolean()) {
      return value.get<bool>() ? "True" : "False";
    }
    if (value.is_null()) {
      return "None";
    }
    if (value.is_array()) {
      std::string result{"["};
      for (size_t index = 0; index < value.size(); ++index) {
        result += (index == 0 ? "" : ", ") + python_repr(value[index]);
      }
      return result + "]";
    }
    if (value.is_object()) {
      std::string result{"{"};
      bool first = true;
      for (const auto& key : object_keys(value)) {
        result += (first ? "" : ", ") + python_string_literal(key) + ": " + python_repr(value.at(key));
        first = false;
      }
      return result + "}";
    }
    return value.dump();
  }

private:
  static bool truthy(const json* data) {
    if (data == nullptr) {
      return false;
    }
    if (data->is_boolean()) {
      return data->get<bool>();
    } else if (data->is_number()) {
      return (*data != 0);
    } else if (data->is_null()) {
      return false;
    } else if (data->is_string()) {
      return !data->get_ref<const json::string_t&>().empty();
    } else if (data->is_object()) {
      return visible_size(*data) != 0;
    }
    return !data->empty();
  }

  static std::string string_value(const json& value) {
    if (value.is_string()) {
      return value.get<std::string>();
    }
    if (value.is_boolean()) {
      return value.get<bool>() ? "True" : "False";
    }
    if (value.is_null()) {
      return "None";
    }
    return python_repr(value);
  }

  void begin_scope() {
    local_scopes.emplace_back();
  }

  void remember_value(const std::string& key) {
    if (local_scopes.empty() || local_scopes.back().find(key) != local_scopes.back().end()) {
      return;
    }
    const auto value = additional_data.find(key);
    local_scopes.back().emplace(key, SavedValue{value != additional_data.end(), value != additional_data.end() ? *value : json()});
  }

  void set_value(const std::string& key, json value) {
    const auto dot = key.find('.');
    const auto root = key.substr(0, dot);
    if (dot == std::string::npos) {
      remember_value(root);
    }
    std::string ptr = key;
    replace_substring(ptr, ".", "/");
    additional_data[json::json_pointer("/" + ptr)] = std::move(value);
  }

  void unset_value(const std::string& key) {
    remember_value(key);
    additional_data.erase(key);
  }

  void end_scope() {
    for (const auto& [key, saved] : local_scopes.back()) {
      if (saved.existed) {
        additional_data[key] = saved.value;
      } else {
        additional_data.erase(key);
      }
    }
    local_scopes.pop_back();
  }

  std::shared_ptr<json> eval_expression(const std::shared_ptr<ExpressionNode>& expression) {
    expression->accept(*this);
    if (data_eval_stack.empty()) {
      throw_renderer_error("empty expression", *expression);
    }
    const auto result = data_eval_stack.top();
    data_eval_stack.pop();
    if (result == nullptr) {
      not_found_stack.pop();
      return std::make_shared<json>(nullptr);
    }
    return std::make_shared<json>(*result);
  }

  void print_data(const std::shared_ptr<json>& value) {
    if (value->is_string()) {
      if (config.html_autoescape) {
        *output_stream << htmlescape(value->get_ref<const json::string_t&>());
      } else {
        *output_stream << value->get_ref<const json::string_t&>();
      }
    } else if (value->is_number_unsigned()) {
      *output_stream << value->get<const json::number_unsigned_t>();
    } else if (value->is_number_integer()) {
      *output_stream << value->get<const json::number_integer_t>();
    } else if (value->is_null()) {
    } else {
      *output_stream << python_repr(*value);
    }
  }

  const std::shared_ptr<json> eval_expression_list(const ExpressionListNode& expression_list, bool* is_undefined = nullptr) {
    if (!expression_list.root) {
      throw_renderer_error("empty expression", expression_list);
    }

    expression_list.root->accept(*this);

    if (data_eval_stack.empty()) {
      throw_renderer_error("empty expression", expression_list);
    } else if (data_eval_stack.size() != 1) {
      throw_renderer_error("malformed expression", expression_list);
    }

    const auto result = data_eval_stack.top();
    data_eval_stack.pop();

    if (result == nullptr) {
      not_found_stack.pop();
      if (is_undefined != nullptr) {
        *is_undefined = true;
      }
      return std::make_shared<json>(nullptr);
    }
    if (is_undefined != nullptr) {
      *is_undefined = false;
    }
    return std::make_shared<json>(*result);
  }

  void throw_renderer_error(const std::string& message, const AstNode& node) {
    const SourceLocation loc = get_source_location(current_template->content, node.pos);
    INJA_THROW(RenderError(message, loc));
  }

  void make_result(const json&& result) {
    auto result_ptr = std::make_shared<json>(result);
    data_tmp_stack.push_back(result_ptr);
    data_eval_stack.push(result_ptr.get());
  }

  void push_result(const json* result, const AstNode& missing_node) {
    data_eval_stack.push(result);
    if (result == nullptr) {
      not_found_stack.push(&missing_node);
    }
  }

  template <size_t N, size_t N_start = 0, bool throw_not_found = true> std::array<const json*, N> get_arguments(const FunctionNode& node) {
    if (node.arguments.size() < N_start + N) {
      throw_renderer_error("function needs " + std::to_string(N_start + N) + " variables, but has only found " + std::to_string(node.arguments.size()), node);
    }

    std::array<const json*, N> result;
    for (size_t i = 0; i < N; i += 1) {
      const auto stack_size = data_eval_stack.size();
      node.arguments[N_start + i]->accept(*this);
      if (data_eval_stack.size() != stack_size + 1) {
        throw_renderer_error("function argument produced a malformed expression", node);
      }
      result[i] = data_eval_stack.top();
      data_eval_stack.pop();

      if (!result[i]) {
        const auto missing_node = not_found_stack.top();
        not_found_stack.pop();

        if (throw_not_found) {
          const auto data_node = dynamic_cast<const DataNode*>(missing_node);
          throw_renderer_error(data_node ? "variable '" + data_node->name + "' not found" : "expression could not be evaluated", *missing_node);
        }
      }
    }
    return result;
  }

  template <bool throw_not_found = true> Arguments get_argument_vector(const FunctionNode& node) {
    const size_t N = node.arguments.size();
    Arguments result {N};
    for (size_t i = 0; i < N; i += 1) {
      const auto stack_size = data_eval_stack.size();
      node.arguments[i]->accept(*this);
      if (data_eval_stack.size() != stack_size + 1) {
        throw_renderer_error("function argument produced a malformed expression", node);
      }
      result[i] = data_eval_stack.top();
      data_eval_stack.pop();

      if (!result[i]) {
        const auto missing_node = not_found_stack.top();
        not_found_stack.pop();

        if (throw_not_found) {
          const auto data_node = dynamic_cast<const DataNode*>(missing_node);
          throw_renderer_error(data_node ? "variable '" + data_node->name + "' not found" : "expression could not be evaluated", *missing_node);
        }
      }
    }
    return result;
  }

  void visit(const BlockNode& node) override {
    for (const auto& n : node.nodes) {
      n->accept(*this);

      if (break_rendering || loop_control != LoopControl::None) {
        break;
      }
    }
  }

  void visit(const TextNode& node) override {
    output_stream->write(current_template->content.c_str() + node.pos, node.length);
  }

  void visit(const ExpressionNode&) override {}

  void visit(const LiteralNode& node) override {
    data_eval_stack.push(&node.value);
  }

  void visit(const DataNode& node) override {
    if (additional_data.contains(node.ptr)) {
      data_eval_stack.push(&(additional_data[node.ptr]));
    } else if (data_input->contains(node.ptr)) {
      data_eval_stack.push(&(*data_input)[node.ptr]);
    } else {
      // Try to evaluate as a no-argument callback
      const auto function_data = function_storage.find_function(node.name, 0);
      if (function_data.operation == FunctionStorage::Operation::Callback) {
        Arguments empty_args {};
        const auto value = std::make_shared<json>(function_data.callback(empty_args));
        data_tmp_stack.push_back(value);
        data_eval_stack.push(value.get());
      } else {
        data_eval_stack.push(nullptr);
        not_found_stack.emplace(&node);
      }
    }
  }

  void visit(const FunctionNode& node) override {
    switch (node.operation) {
    case Op::Not: {
      const auto args = get_arguments<1, 0, false>(node);
      make_result(!truthy(args[0]));
    } break;
    case Op::And: {
      const auto first = get_arguments<1, 0, false>(node)[0];
      if (truthy(first)) {
        node.arguments[1]->accept(*this);
      } else {
        push_result(first, node);
      }
    } break;
    case Op::Or: {
      const auto first = get_arguments<1, 0, false>(node)[0];
      if (truthy(first)) {
        push_result(first, node);
      } else {
        node.arguments[1]->accept(*this);
      }
    } break;
    case Op::In: {
      const auto args = get_arguments<2>(node);
      if (args[1]->is_object() && args[0]->is_string()) {
        make_result(args[1]->contains(args[0]->get<std::string>()));
      } else if (args[1]->is_string() && args[0]->is_string()) {
        make_result(args[1]->get_ref<const json::string_t&>().find(args[0]->get_ref<const json::string_t&>()) != std::string::npos);
      } else {
        make_result(std::find(args[1]->begin(), args[1]->end(), *args[0]) != args[1]->end());
      }
    } break;
    case Op::NotIn: {
      const auto args = get_arguments<2>(node);
      if (args[1]->is_object() && args[0]->is_string()) {
        make_result(!args[1]->contains(args[0]->get<std::string>()));
      } else if (args[1]->is_string() && args[0]->is_string()) {
        make_result(args[1]->get_ref<const json::string_t&>().find(args[0]->get_ref<const json::string_t&>()) == std::string::npos);
      } else {
        make_result(std::find(args[1]->begin(), args[1]->end(), *args[0]) == args[1]->end());
      }
    } break;
    case Op::Equal: {
      const auto args = get_arguments<2>(node);
      make_result(*args[0] == *args[1]);
    } break;
    case Op::NotEqual: {
      const auto args = get_arguments<2>(node);
      make_result(*args[0] != *args[1]);
    } break;
    case Op::Greater: {
      const auto args = get_arguments<2>(node);
      make_result(*args[0] > *args[1]);
    } break;
    case Op::GreaterEqual: {
      const auto args = get_arguments<2>(node);
      make_result(*args[0] >= *args[1]);
    } break;
    case Op::Less: {
      const auto args = get_arguments<2>(node);
      make_result(*args[0] < *args[1]);
    } break;
    case Op::LessEqual: {
      const auto args = get_arguments<2>(node);
      make_result(*args[0] <= *args[1]);
    } break;
    case Op::Add: {
      const auto args = get_arguments<2>(node);
      if (args[0]->is_string() && args[1]->is_string()) {
        make_result(args[0]->get_ref<const json::string_t&>() + args[1]->get_ref<const json::string_t&>());
      } else if (args[0]->is_array() && args[1]->is_array()) {
        auto result = *args[0];
        result.insert(result.end(), args[1]->begin(), args[1]->end());
        make_result(std::move(result));
      } else if (args[0]->is_number_integer() && args[1]->is_number_integer()) {
        make_result(args[0]->get<const json::number_integer_t>() + args[1]->get<const json::number_integer_t>());
      } else {
        make_result(args[0]->get<const json::number_float_t>() + args[1]->get<const json::number_float_t>());
      }
    } break;
    case Op::Concat: {
      const auto args = get_arguments<2>(node);
      make_result(string_value(*args[0]) + string_value(*args[1]));
    } break;
    case Op::Subtract: {
      const auto args = get_arguments<2>(node);
      if (args[0]->is_number_integer() && args[1]->is_number_integer()) {
        make_result(args[0]->get<const json::number_integer_t>() - args[1]->get<const json::number_integer_t>());
      } else {
        make_result(args[0]->get<const json::number_float_t>() - args[1]->get<const json::number_float_t>());
      }
    } break;
    case Op::Multiplication: {
      const auto args = get_arguments<2>(node);
      if ((args[0]->is_string() || args[0]->is_array()) && args[1]->is_number_integer()) {
        const auto count = args[1]->get<const json::number_integer_t>();
        if (count < 0) {
          make_result(args[0]->is_string() ? json("") : json::array());
          break;
        }
        json result = args[0]->is_string() ? json("") : json::array();
        for (json::number_integer_t index = 0; index < count; ++index) {
          if (args[0]->is_string()) {
            result.get_ref<std::string&>() += args[0]->get_ref<const std::string&>();
          } else {
            result.insert(result.end(), args[0]->begin(), args[0]->end());
          }
        }
        make_result(std::move(result));
      } else if (args[0]->is_number_integer() && args[1]->is_number_integer()) {
        make_result(args[0]->get<const json::number_integer_t>() * args[1]->get<const json::number_integer_t>());
      } else {
        make_result(args[0]->get<const json::number_float_t>() * args[1]->get<const json::number_float_t>());
      }
    } break;
    case Op::Division: {
      const auto args = get_arguments<2>(node);
      if (args[1]->get<const json::number_float_t>() == 0) {
        throw_renderer_error("division by zero", node);
      }
      make_result(args[0]->get<const json::number_float_t>() / args[1]->get<const json::number_float_t>());
    } break;
    case Op::Power: {
      const auto args = get_arguments<2>(node);
      if (args[0]->is_number_integer() && args[1]->get<const json::number_integer_t>() >= 0) {
        const auto result = static_cast<json::number_integer_t>(std::pow(args[0]->get<const json::number_integer_t>(), args[1]->get<const json::number_integer_t>()));
        make_result(result);
      } else {
        const auto result = std::pow(args[0]->get<const json::number_float_t>(), args[1]->get<const json::number_integer_t>());
        make_result(result);
      }
    } break;
    case Op::Modulo: {
      const auto args = get_arguments<2>(node);
      make_result(args[0]->get<const json::number_integer_t>() % args[1]->get<const json::number_integer_t>());
    } break;
    case Op::AtId: {
      const auto container = get_arguments<1, 0, false>(node)[0];
      node.arguments[1]->accept(*this);
      if (not_found_stack.empty()) {
        throw_renderer_error("could not find element with given name", node);
      }
      const auto id_node = dynamic_cast<const DataNode*>(not_found_stack.top());
      not_found_stack.pop();
      data_eval_stack.pop();
      if (id_node == nullptr) {
        throw_renderer_error("invalid member access", node);
      }
      data_eval_stack.push(&container->at(id_node->name));
    } break;
    case Op::At: {
      const auto args = get_arguments<2, 0, false>(node);
      if (args[0] == nullptr || args[1] == nullptr) {
        data_eval_stack.push(nullptr);
        not_found_stack.push(&node);
        return;
      }
      if (args[0]->is_object()) {
        const auto key = string_value(*args[1]);
        const auto value = args[0]->find(key);
        data_eval_stack.push(value == args[0]->end() ? nullptr : &*value);
        if (value == args[0]->end()) {
          not_found_stack.push(&node);
        }
      } else {
        auto index = args[1]->get<int64_t>();
        const auto size = static_cast<int64_t>(args[0]->size());
        if (index < 0) {
          index += size;
        }
        const bool missing = index < 0 || index >= size;
        data_eval_stack.push(missing ? nullptr : &args[0]->at(static_cast<size_t>(index)));
        if (missing) {
          not_found_stack.push(&node);
        }
      }
    } break;
    case Op::Slice: {
      const auto args = get_arguments<4>(node);
      if (!args[0]->is_array() && !args[0]->is_string()) {
        throw_renderer_error("slice requires an array or string", node);
      }
      const auto size = static_cast<int64_t>(args[0]->size());
      const auto step = args[3]->is_null() ? int64_t{1} : args[3]->get<int64_t>();
      if (step == 0) {
        throw_renderer_error("slice step cannot be zero", node);
      }
      int64_t start = args[1]->is_null() ? (step > 0 ? 0 : size - 1) : args[1]->get<int64_t>();
      int64_t stop = args[2]->is_null() ? (step > 0 ? size : -1) : args[2]->get<int64_t>();
      if (!args[1]->is_null() && start < 0) start += size;
      if (!args[2]->is_null() && stop < 0) stop += size;
      if (step > 0) {
        start = std::clamp(start, int64_t{0}, size);
        stop = std::clamp(stop, int64_t{0}, size);
      } else {
        start = std::clamp(start, int64_t{-1}, std::max(int64_t{-1}, size - 1));
        stop = std::clamp(stop, int64_t{-1}, std::max(int64_t{-1}, size - 1));
      }
      json result = args[0]->is_string() ? json("") : json::array();
      for (auto index = start; step > 0 ? index < stop : index > stop; index += step) {
        if (args[0]->is_string()) {
          result.get_ref<std::string&>().push_back(args[0]->get_ref<const std::string&>()[static_cast<size_t>(index)]);
        } else {
          result.push_back((*args[0])[static_cast<size_t>(index)]);
        }
      }
      make_result(std::move(result));
    } break;
    case Op::Conditional: {
      const auto condition = get_arguments<1, 0, false>(node)[0];
      node.arguments[truthy(condition) ? 1 : 2]->accept(*this);
    } break;
    case Op::Array: {
      json result = json::array();
      for (const auto* value : get_argument_vector(node)) {
        result.push_back(*value);
      }
      make_result(std::move(result));
    } break;
    case Op::Object: {
      const auto values = get_argument_vector(node);
      json result = json::object();
      result[std::string(object_order_key)] = json::array();
      for (size_t index = 0; index < values.size(); index += 2) {
        const auto key = string_value(*values[index]);
        if (key != object_order_key && !result.contains(key)) {
          result[std::string(object_order_key)].push_back(key);
        }
        result[key] = *values[index + 1];
      }
      make_result(std::move(result));
    } break;
    case Op::Capitalize: {
      auto result = get_arguments<1>(node)[0]->get<json::string_t>();
      result[0] = static_cast<char>(::toupper(result[0]));
      std::transform(result.begin() + 1, result.end(), result.begin() + 1, [](char c) { return static_cast<char>(::tolower(c)); });
      make_result(std::move(result));
    } break;
    case Op::Default: {
      const auto test_arg = get_arguments<1, 0, false>(node)[0];
      bool use_default = test_arg == nullptr;
      if (!use_default && node.arguments.size() == 3) {
        use_default = get_arguments<1, 2>(node)[0]->get<bool>() && !truthy(test_arg);
      }
      data_eval_stack.push(use_default ? get_arguments<1, 1>(node)[0] : test_arg);
    } break;
    case Op::DivisibleBy: {
      const auto args = get_arguments<2>(node);
      const auto divisor = args[1]->get<const json::number_integer_t>();
      make_result((divisor != 0) && (args[0]->get<const json::number_integer_t>() % divisor == 0));
    } break;
    case Op::Even: {
      make_result(get_arguments<1>(node)[0]->get<const json::number_integer_t>() % 2 == 0);
    } break;
    case Op::Exists: {
      auto&& name = get_arguments<1>(node)[0]->get_ref<const json::string_t&>();
      make_result(data_input->contains(json::json_pointer(DataNode::convert_dot_to_ptr(name))));
    } break;
    case Op::ExistsInObject: {
      const auto args = get_arguments<2>(node);
      auto&& name = args[1]->get_ref<const json::string_t&>();
      make_result(args[0]->find(name) != args[0]->end());
    } break;
    case Op::First: {
      const auto result = &get_arguments<1>(node)[0]->front();
      data_eval_stack.push(result);
    } break;
    case Op::Float: {
      make_result(std::stod(get_arguments<1>(node)[0]->get_ref<const json::string_t&>()));
    } break;
    case Op::Int: {
      make_result(std::stoi(get_arguments<1>(node)[0]->get_ref<const json::string_t&>()));
    } break;
    case Op::Last: {
      const auto result = &get_arguments<1>(node)[0]->back();
      data_eval_stack.push(result);
    } break;
    case Op::Length: {
      const auto val = get_arguments<1>(node)[0];
      if (val->is_string()) {
        make_result(val->get_ref<const json::string_t&>().length());
      } else {
        make_result(visible_size(*val));
      }
    } break;
    case Op::Lower: {
      auto result = get_arguments<1>(node)[0]->get<json::string_t>();
      std::transform(result.begin(), result.end(), result.begin(), [](char c) { return static_cast<char>(::tolower(c)); });
      make_result(std::move(result));
    } break;
    case Op::Max: {
      const auto args = get_arguments<1>(node);
      const auto result = std::max_element(args[0]->begin(), args[0]->end());
      data_eval_stack.push(&(*result));
    } break;
    case Op::Min: {
      const auto args = get_arguments<1>(node);
      const auto result = std::min_element(args[0]->begin(), args[0]->end());
      data_eval_stack.push(&(*result));
    } break;
    case Op::Odd: {
      make_result(get_arguments<1>(node)[0]->get<const json::number_integer_t>() % 2 != 0);
    } break;
    case Op::Range: {
      std::vector<int> result(get_arguments<1>(node)[0]->get<const json::number_integer_t>());
      std::iota(result.begin(), result.end(), 0);
      make_result(std::move(result));
    } break;
    case Op::Replace: {
      const auto args = get_arguments<3>(node);
      auto result = args[0]->get<std::string>();
      replace_substring(result, args[1]->get<std::string>(), args[2]->get<std::string>());
      make_result(std::move(result));
    } break;
    case Op::Round: {
      const auto args = get_arguments<2>(node);
      const auto precision = args[1]->get<const json::number_integer_t>();
      const double result = std::round(args[0]->get<const json::number_float_t>() * std::pow(10.0, precision)) / std::pow(10.0, precision);
      if (precision == 0) {
        make_result(static_cast<int>(result));
      } else {
        make_result(result);
      }
    } break;
    case Op::Sort: {
      auto result_ptr = std::make_shared<json>(get_arguments<1>(node)[0]->get<std::vector<json>>());
      std::sort(result_ptr->begin(), result_ptr->end());
      data_tmp_stack.push_back(result_ptr);
      data_eval_stack.push(result_ptr.get());
    } break;
    case Op::Upper: {
      auto result = get_arguments<1>(node)[0]->get<json::string_t>();
      std::transform(result.begin(), result.end(), result.begin(), [](char c) { return static_cast<char>(::toupper(c)); });
      make_result(std::move(result));
    } break;
    case Op::IsBoolean: {
      make_result(get_arguments<1>(node)[0]->is_boolean());
    } break;
    case Op::IsDefined: {
      make_result(get_arguments<1, 0, false>(node)[0] != nullptr);
    } break;
    case Op::IsUndefined: {
      make_result(get_arguments<1, 0, false>(node)[0] == nullptr);
    } break;
    case Op::IsNone: {
      const auto value = get_arguments<1, 0, false>(node)[0];
      make_result(value != nullptr && value->is_null());
    } break;
    case Op::IsIterable: {
      const auto value = get_arguments<1, 0, false>(node)[0];
      make_result(value != nullptr && (value->is_array() || value->is_object() || value->is_string()));
    } break;
    case Op::IsMapping: {
      const auto value = get_arguments<1, 0, false>(node)[0];
      make_result(value != nullptr && value->is_object());
    } break;
    case Op::IsNumber: {
      make_result(get_arguments<1>(node)[0]->is_number());
    } break;
    case Op::IsInteger: {
      make_result(get_arguments<1>(node)[0]->is_number_integer());
    } break;
    case Op::IsFloat: {
      make_result(get_arguments<1>(node)[0]->is_number_float());
    } break;
    case Op::IsObject: {
      make_result(get_arguments<1>(node)[0]->is_object());
    } break;
    case Op::IsSequence: {
      const auto value = get_arguments<1, 0, false>(node)[0];
      make_result(value != nullptr && (value->is_array() || value->is_object() || value->is_string()));
    } break;
    case Op::IsArray: {
      make_result(get_arguments<1>(node)[0]->is_array());
    } break;
    case Op::IsString: {
      const auto value = get_arguments<1, 0, false>(node)[0];
      make_result(value != nullptr && value->is_string());
    } break;
    case Op::IsTrue: {
      const auto value = get_arguments<1, 0, false>(node)[0];
      make_result(value != nullptr && value->is_boolean() && value->get<bool>());
    } break;
    case Op::IsFalse: {
      const auto value = get_arguments<1, 0, false>(node)[0];
      make_result(value != nullptr && value->is_boolean() && !value->get<bool>());
    } break;
    case Op::MacroCall: {
      const auto macro_it = current_template->macro_storage.find(node.name);
      if (macro_it == current_template->macro_storage.end()) {
        throw_renderer_error("unknown macro '" + node.name + "'", node);
      }
      const auto& macro = *macro_it->second;
      const auto argument_ptrs = get_argument_vector<false>(node);
      std::vector<json> arguments;
      std::vector<bool> argument_defined;
      arguments.reserve(argument_ptrs.size());
      argument_defined.reserve(argument_ptrs.size());
      for (const auto* argument : argument_ptrs) {
        argument_defined.push_back(argument != nullptr);
        arguments.push_back(argument == nullptr ? json(nullptr) : *argument);
      }

      begin_scope();
      bool scope_active = true;
      std::vector<bool> bound(macro.parameters.size(), false);
      size_t next_positional = 0;
      try {
        for (size_t index = 0; index < arguments.size(); ++index) {
          const auto& argument_name = node.argument_names[index];
          size_t parameter_index = next_positional;
          if (!argument_name.empty()) {
            const auto parameter = std::find_if(macro.parameters.begin(), macro.parameters.end(), [&](const auto& candidate) {
              return candidate.name == argument_name;
            });
            if (parameter == macro.parameters.end()) {
              throw_renderer_error("unknown argument '" + argument_name + "' for macro '" + node.name + "'", node);
            }
            parameter_index = static_cast<size_t>(std::distance(macro.parameters.begin(), parameter));
          } else {
            while (parameter_index < bound.size() && bound[parameter_index]) ++parameter_index;
            next_positional = parameter_index + 1;
          }
          if (parameter_index >= bound.size() || bound[parameter_index]) {
            throw_renderer_error("invalid arguments for macro '" + node.name + "'", node);
          }
          if (argument_defined[index]) {
            set_value(macro.parameters[parameter_index].name, arguments[index]);
          } else {
            unset_value(macro.parameters[parameter_index].name);
          }
          bound[parameter_index] = true;
        }
        for (size_t index = 0; index < macro.parameters.size(); ++index) {
          if (!bound[index]) {
            if (!macro.parameters[index].default_value) {
              throw_renderer_error("missing argument '" + macro.parameters[index].name + "' for macro '" + node.name + "'", node);
            }
            set_value(macro.parameters[index].name, *eval_expression(macro.parameters[index].default_value));
          }
        }

        std::ostringstream output;
        auto* previous_output = output_stream;
        output_stream = &output;
        try {
          macro.body.accept(*this);
        } catch (...) {
          output_stream = previous_output;
          throw;
        }
        output_stream = previous_output;
        end_scope();
        scope_active = false;
        make_result(output.str());
      } catch (...) {
        if (scope_active) {
          end_scope();
        }
        throw;
      }
    } break;
    case Op::Callback: {
      auto args = get_argument_vector(node);
      make_result(node.callback(args));
    } break;
    case Op::Super: {
      const auto args = get_argument_vector(node);
      const size_t old_level = current_level;
      const size_t level_diff = (args.size() == 1) ? args[0]->get<int>() : 1;
      const size_t level = current_level + level_diff;

      if (block_statement_stack.empty()) {
        throw_renderer_error("super() call is not within a block", node);
      }

      if (level < 1 || level > template_stack.size() - 1) {
        throw_renderer_error("level of super() call does not match parent templates (between 1 and " + std::to_string(template_stack.size() - 1) + ")", node);
      }

      const auto current_block_statement = block_statement_stack.back();
      const Template* new_template = template_stack.at(level);
      const Template* old_template = current_template;
      const auto block_it = new_template->block_storage.find(current_block_statement->name);
      if (block_it != new_template->block_storage.end()) {
        current_template = new_template;
        current_level = level;
        block_it->second->block.accept(*this);
        current_level = old_level;
        current_template = old_template;
      } else {
        throw_renderer_error("could not find block with name '" + current_block_statement->name + "'", node);
      }
      make_result(nullptr);
    } break;
    case Op::Join: {
      const auto args = get_arguments<2>(node);
      const auto separator = args[1]->get<json::string_t>();
      std::ostringstream os;
      std::string sep;
      for (const auto& value : *args[0]) {
        os << sep;
        if (value.is_string()) {
          os << value.get<std::string>(); // otherwise the value is surrounded with ""
        } else {
          os << python_repr(value);
        }
        sep = separator;
      }
      make_result(os.str());
    } break;
    case Op::None:
      break;
    }
  }

  void visit(const ExpressionListNode& node) override {
    print_data(eval_expression_list(node));
  }

  void visit(const StatementNode&) override {}

  void visit(const ForStatementNode&) override {}

  void visit(const ForArrayStatementNode& node) override {
    bool is_undefined = false;
    const auto result = eval_expression_list(node.condition, &is_undefined);
    if (is_undefined) {
      return;
    }
    if (!result->is_array() && !result->is_object() && !result->is_string()) {
      throw_renderer_error("for loop requires an iterable", node);
    }

    json values = json::array();
    if (result->is_object()) {
      for (const auto& key : object_keys(*result)) {
        values.push_back(key);
      }
    } else if (result->is_string()) {
      for (const auto ch : result->get_ref<const std::string&>()) {
        values.push_back(std::string(1, ch));
      }
    } else {
      for (const auto& value : *result) {
        values.push_back(value);
      }
    }
    json selected_values = json::array();
    for (const auto& value : values) {
      begin_scope();
      set_value(node.value, value);
      const bool selected = !node.has_filter || truthy(eval_expression_list(node.filter).get());
      end_scope();
      if (selected) selected_values.push_back(value);
    }

    for (size_t index = 0; index < selected_values.size(); ++index) {
      begin_scope();
      set_value(node.value, selected_values[index]);
      json loop = {
          {"index0", index},
          {"index", index + 1},
          {"index1", index + 1},
          {"revindex0", selected_values.size() - index - 1},
          {"revindex", selected_values.size() - index},
          {"first", index == 0},
          {"last", index + 1 == selected_values.size()},
          {"is_first", index == 0},
          {"is_last", index + 1 == selected_values.size()},
          {"length", selected_values.size()},
          {"previtem", index == 0 ? json(nullptr) : selected_values[index - 1]},
          {"nextitem", index + 1 == selected_values.size() ? json(nullptr) : selected_values[index + 1]},
      };
      set_value("loop", std::move(loop));
      current_loop_data = &additional_data["loop"];
      node.body.accept(*this);
      end_scope();
      current_loop_data = &additional_data["loop"];
      if (loop_control == LoopControl::Break) {
        loop_control = LoopControl::None;
        break;
      }
      if (loop_control == LoopControl::Continue) {
        loop_control = LoopControl::None;
      }
    }
  }

  void visit(const ForObjectStatementNode& node) override {
    bool is_undefined = false;
    const auto result = eval_expression_list(node.condition, &is_undefined);
    if (is_undefined) {
      return;
    }
    json values = json::array();
    if (result->is_object()) {
      for (const auto& key : object_keys(*result)) {
        values.push_back(json::array({key, result->at(key)}));
      }
    } else if (result->is_array()) {
      for (const auto& value : *result) {
        if (!value.is_array() || value.size() != 2) {
          throw_renderer_error("tuple loop requires two-element values", node);
        }
        values.push_back(value);
      }
    } else {
      throw_renderer_error("tuple loop requires an object or pair sequence", node);
    }

    json selected_values = json::array();
    for (const auto& value : values) {
      begin_scope();
      set_value(node.key, value[0]);
      set_value(node.value, value[1]);
      const bool selected = !node.has_filter || truthy(eval_expression_list(node.filter).get());
      end_scope();
      if (selected) selected_values.push_back(value);
    }

    for (size_t index = 0; index < selected_values.size(); ++index) {
      const auto& pair = selected_values[index];
      begin_scope();
      set_value(node.key, pair[0]);
      set_value(node.value, pair[1]);
      json loop = {
          {"index0", index},
          {"index", index + 1},
          {"index1", index + 1},
          {"revindex0", selected_values.size() - index - 1},
          {"revindex", selected_values.size() - index},
          {"first", index == 0},
          {"last", index + 1 == selected_values.size()},
          {"is_first", index == 0},
          {"is_last", index + 1 == selected_values.size()},
          {"length", selected_values.size()},
          {"previtem", index == 0 ? json(nullptr) : selected_values[index - 1]},
          {"nextitem", index + 1 == selected_values.size() ? json(nullptr) : selected_values[index + 1]},
      };
      set_value("loop", std::move(loop));
      current_loop_data = &additional_data["loop"];
      node.body.accept(*this);
      end_scope();
      current_loop_data = &additional_data["loop"];
      if (loop_control == LoopControl::Break) {
        loop_control = LoopControl::None;
        break;
      }
      if (loop_control == LoopControl::Continue) {
        loop_control = LoopControl::None;
      }
    }
  }

  void visit(const IfStatementNode& node) override {
    const auto result = eval_expression_list(node.condition);
    if (truthy(result.get())) {
      node.true_statement.accept(*this);
    } else if (node.has_false_statement) {
      node.false_statement.accept(*this);
    }
  }

  void visit(const IncludeStatementNode& node) override {
    auto sub_renderer = Renderer(config, template_storage, function_storage);
    const auto included_template_it = template_storage.find(node.file);
    if (included_template_it != template_storage.end()) {
      sub_renderer.render_to(*output_stream, included_template_it->second, *data_input, &additional_data);
    } else if (config.throw_at_missing_includes) {
      throw_renderer_error("include '" + node.file + "' not found", node);
    }
  }

  void visit(const ExtendsStatementNode& node) override {
    const auto included_template_it = template_storage.find(node.file);
    if (included_template_it != template_storage.end()) {
      const Template* parent_template = &included_template_it->second;
      render_to(*output_stream, *parent_template, *data_input, &additional_data);
      break_rendering = true;
    } else if (config.throw_at_missing_includes) {
      throw_renderer_error("extends '" + node.file + "' not found", node);
    }
  }

  void visit(const BlockStatementNode& node) override {
    const size_t old_level = current_level;
    current_level = 0;
    current_template = template_stack.front();
    const auto block_it = current_template->block_storage.find(node.name);
    if (block_it != current_template->block_storage.end()) {
      block_statement_stack.emplace_back(&node);
      block_it->second->block.accept(*this);
      block_statement_stack.pop_back();
    }
    current_level = old_level;
    current_template = template_stack.back();
  }

  void visit(const SetStatementNode& node) override {
    set_value(node.key, *eval_expression_list(node.expression));
  }

  void visit(const SetBlockStatementNode& node) override {
    std::ostringstream output;
    auto* previous_output = output_stream;
    output_stream = &output;
    try {
      node.body.accept(*this);
    } catch (...) {
      output_stream = previous_output;
      throw;
    }
    output_stream = previous_output;
    set_value(node.key, output.str());
  }

  void visit(const MacroStatementNode&) override {}

  void visit(const LoopControlStatementNode& node) override {
    loop_control = node.control == LoopControlStatementNode::Control::Break ? LoopControl::Break : LoopControl::Continue;
  }

public:
  explicit Renderer(const RenderConfig& config, const TemplateStorage& template_storage, const FunctionStorage& function_storage)
      : config(config), template_storage(template_storage), function_storage(function_storage) {}

  void render_to(std::ostream& os, const Template& tmpl, const json& data, json* loop_data = nullptr) {
    output_stream = &os;
    current_template = &tmpl;
    data_input = &data;
    if (loop_data != nullptr) {
      additional_data = *loop_data;
      current_loop_data = &additional_data["loop"];
    }

    template_stack.emplace_back(current_template);
    current_template->root.accept(*this);

    data_tmp_stack.clear();
  }
};

} // namespace inja

#endif // INCLUDE_INJA_RENDERER_HPP_

// #include "template.hpp"

// #include "throw.hpp"


namespace inja {

/*!
 * \brief Class for changing the configuration.
 */
class Environment {
  FunctionStorage function_storage;
  TemplateStorage template_storage;

protected:
  LexerConfig lexer_config;
  ParserConfig parser_config;
  RenderConfig render_config;

  std::filesystem::path input_path;
  std::filesystem::path output_path;

public:
  Environment(): Environment("") {}
  explicit Environment(const std::filesystem::path& global_path): input_path(global_path), output_path(global_path) {}
  Environment(const std::filesystem::path& input_path, const std::filesystem::path& output_path): input_path(input_path), output_path(output_path) {}

  /// Sets the opener and closer for template statements
  void set_statement(const std::string& open, const std::string& close) {
    lexer_config.statement_open = open;
    lexer_config.statement_open_no_lstrip = open + "+";
    lexer_config.statement_open_force_lstrip = open + "-";
    lexer_config.statement_close = close;
    lexer_config.statement_close_force_rstrip = "-" + close;
    lexer_config.update_open_chars();
  }

  /// Sets the opener for template line statements
  void set_line_statement(const std::string& open) {
    lexer_config.line_statement = open;
    lexer_config.update_open_chars();
  }

  /// Sets the opener and closer for template expressions
  void set_expression(const std::string& open, const std::string& close) {
    lexer_config.expression_open = open;
    lexer_config.expression_open_force_lstrip = open + "-";
    lexer_config.expression_close = close;
    lexer_config.expression_close_force_rstrip = "-" + close;
    lexer_config.update_open_chars();
  }

  /// Sets the opener and closer for template comments
  void set_comment(const std::string& open, const std::string& close) {
    lexer_config.comment_open = open;
    lexer_config.comment_open_force_lstrip = open + "-";
    lexer_config.comment_close = close;
    lexer_config.comment_close_force_rstrip = "-" + close;
    lexer_config.update_open_chars();
  }

  /// Sets whether to remove the first newline after a block
  void set_trim_blocks(bool trim_blocks) {
    lexer_config.trim_blocks = trim_blocks;
  }

  /// Sets whether to strip the spaces and tabs from the start of a line to a block
  void set_lstrip_blocks(bool lstrip_blocks) {
    lexer_config.lstrip_blocks = lstrip_blocks;
  }

  /// Sets the element notation syntax
  void set_search_included_templates_in_files(bool search_in_files) {
    parser_config.search_included_templates_in_files = search_in_files;
  }

  /// Sets whether a missing include will throw an error
  void set_throw_at_missing_includes(bool will_throw) {
    render_config.throw_at_missing_includes = will_throw;
  }

  /// Sets whether we'll automatically perform HTML escape
  void set_html_autoescape(bool will_escape) {
    render_config.html_autoescape = will_escape;
  }

  Template parse(std::string_view input) {
    Parser parser(parser_config, lexer_config, template_storage, function_storage);
    return parser.parse(input, input_path);
  }

  Template parse_template(const std::filesystem::path& filename) {
    Parser parser(parser_config, lexer_config, template_storage, function_storage);
    auto result = Template(Parser::load_file(input_path / filename));
    parser.parse_into_template(result, (input_path / filename).string());
    return result;
  }

  Template parse_file(const std::filesystem::path& filename) {
    return parse_template(filename);
  }

  std::string render(std::string_view input, const json& data) {
    return render(parse(input), data);
  }

  std::string render(const Template& tmpl, const json& data) {
    std::stringstream os;
    render_to(os, tmpl, data);
    return os.str();
  }

  std::string render_file(const std::filesystem::path& filename, const json& data) {
    return render(parse_template(filename), data);
  }

  std::string render_file_with_json_file(const std::filesystem::path& filename, const std::string& filename_data) {
    const json data = load_json(filename_data);
    return render_file(filename, data);
  }

  void write(const std::filesystem::path& filename, const json& data, const std::string& filename_out) {
    std::ofstream file(output_path / filename_out);
    file << render_file(filename, data);
    file.close();
  }

  void write(const Template& temp, const json& data, const std::string& filename_out) {
    std::ofstream file(output_path / filename_out);
    file << render(temp, data);
    file.close();
  }

  void write_with_json_file(const std::filesystem::path& filename, const std::string& filename_data, const std::string& filename_out) {
    const json data = load_json(filename_data);
    write(filename, data, filename_out);
  }

  void write_with_json_file(const Template& temp, const std::string& filename_data, const std::string& filename_out) {
    const json data = load_json(filename_data);
    write(temp, data, filename_out);
  }

  std::ostream& render_to(std::ostream& os, const Template& tmpl, const json& data) {
    Renderer(render_config, template_storage, function_storage).render_to(os, tmpl, data);
    return os;
  }

  std::ostream& render_to(std::ostream& os, const std::string_view input, const json& data) {
    return render_to(os, parse(input), data);
  }

  std::string load_file(const std::string& filename) {
    const Parser parser(parser_config, lexer_config, template_storage, function_storage);
    return Parser::load_file(input_path / filename);
  }

  json load_json(const std::string& filename) {
    std::ifstream file;
    file.open(input_path / filename);
    if (file.fail()) {
      INJA_THROW(FileError("failed accessing file at '" + (input_path / filename).string() + "'"));
    }

    return json::parse(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
  }

  /*!
  @brief Adds a variadic callback
  */
  void add_callback(const std::string& name, const CallbackFunction& callback) {
    add_callback(name, -1, callback);
  }

  /*!
  @brief Adds a variadic void callback
  */
  void add_void_callback(const std::string& name, const VoidCallbackFunction& callback) {
    add_void_callback(name, -1, callback);
  }

  /*!
  @brief Adds a callback with given number or arguments
  */
  void add_callback(const std::string& name, int num_args, const CallbackFunction& callback) {
    function_storage.add_callback(name, num_args, callback);
  }

  /*!
  @brief Adds a void callback with given number or arguments
  */
  void add_void_callback(const std::string& name, int num_args, const VoidCallbackFunction& callback) {
    function_storage.add_callback(name, num_args, [callback](Arguments& args) {
      callback(args);
      return json();
    });
  }

  /** Includes a template with a given name into the environment.
   * Then, a template can be rendered in another template using the
   * include "<name>" syntax.
   */
  void include_template(const std::string& name, const Template& tmpl) {
    template_storage[name] = tmpl;
  }

  /*!
  @brief Sets a function that is called when an included file is not found
  */
  void set_include_callback(const std::function<Template(const std::filesystem::path&, const std::string&)>& callback) {
    parser_config.include_callback = callback;
  }
};

/*!
@brief render with default settings to a string
*/
inline std::string render(std::string_view input, const json& data) {
  return Environment().render(input, data);
}

/*!
@brief render with default settings to the given output stream
*/
inline void render_to(std::ostream& os, std::string_view input, const json& data) {
  Environment env;
  env.render_to(os, env.parse(input), data);
}

} // namespace inja

#endif // INCLUDE_INJA_ENVIRONMENT_HPP_

// #include "exceptions.hpp"

// #include "parser.hpp"

// #include "renderer.hpp"

// #include "template.hpp"


#endif // INCLUDE_INJA_INJA_HPP_
