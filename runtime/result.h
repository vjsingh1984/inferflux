#pragma once

#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace inferflux {

/// Structured error value returned by Result<T>.
struct Error {
  std::string message;

  enum class Category {
    kGeneric,
    kNotFound,
    kUnavailable,
    kInvalid,
  };
  Category category{Category::kGeneric};

  explicit Error(std::string msg, Category cat = Category::kGeneric)
      : message(std::move(msg)), category(cat) {}
};

/// Result<T> — a value-or-error return type.
///
/// Replaces ad-hoc patterns (bool + side-channel, optional, enum) with
/// a single vocabulary type.  Intentionally lightweight: no exceptions,
/// no allocations beyond the contained T/Error.
///
/// Usage:
///   Result<int> r = Ok(42);
///   Result<int> r = Err("something broke");
///   if (r) use(r.value());
///   else   log(r.error().message);
template <typename T>
class Result {
public:
  // Implicit construction from T (success).
  Result(T val) : data_(std::move(val)) {} // NOLINT(google-explicit-constructor)

  // Implicit construction from Error (failure).
  Result(Error err) : data_(std::move(err)) {} // NOLINT(google-explicit-constructor)

  bool ok() const { return std::holds_alternative<T>(data_); }
  explicit operator bool() const { return ok(); }

  T &value() & { return std::get<T>(data_); }
  const T &value() const & { return std::get<T>(data_); }
  T &&value() && { return std::get<T>(std::move(data_)); }

  const Error &error() const { return std::get<Error>(data_); }

private:
  std::variant<T, Error> data_;
};

/// Result<void> specialisation for operations that produce no value.
template <>
class Result<void> {
public:
  Result() : error_(std::nullopt) {}

  Result(Error err) : error_(std::move(err)) {} // NOLINT(google-explicit-constructor)

  bool ok() const { return !error_.has_value(); }
  explicit operator bool() const { return ok(); }

  void value() const {} // No-op, for generic code.

  const Error &error() const { return *error_; }

private:
  std::optional<Error> error_;
};

// --- Free-function factories ---

/// Construct a successful Result<T>.
template <typename T>
Result<std::decay_t<T>> Ok(T &&val) {
  return Result<std::decay_t<T>>(std::forward<T>(val));
}

/// Construct a successful Result<void>.
inline Result<void> OkVoid() { return Result<void>(); }

/// Construct a failed Result with a message and optional category.
inline Error Err(std::string msg,
                 Error::Category cat = Error::Category::kGeneric) {
  return Error(std::move(msg), cat);
}

} // namespace inferflux
