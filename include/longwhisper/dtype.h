#pragma once

#include <expected>
#include <string>
#include <string_view>

namespace longwhisper {

enum class DType {
  kUnknown = 0,
  kFloat16,
  kBFloat16,
  kFloat32,
  kInt32,
  kInt64,
};

const char* DTypeName(DType dtype);
std::expected<DType, std::string> ParseDType(std::string_view text);

}  // namespace longwhisper
