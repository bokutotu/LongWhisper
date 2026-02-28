#include "longwhisper/dtype.h"

namespace longwhisper {

const char* DTypeName(DType dtype) {
  switch (dtype) {
    case DType::kFloat16:
      return "float16";
    case DType::kBFloat16:
      return "bfloat16";
    case DType::kFloat32:
      return "float32";
    case DType::kInt32:
      return "int32";
    case DType::kInt64:
      return "int64";
    case DType::kUnknown:
      return "unknown";
  }
  return "unknown";
}

std::expected<DType, std::string> ParseDType(std::string_view text) {
  if (text == "float16") {
    return DType::kFloat16;
  }
  if (text == "bfloat16") {
    return DType::kBFloat16;
  }
  if (text == "float32") {
    return DType::kFloat32;
  }
  if (text == "int32") {
    return DType::kInt32;
  }
  if (text == "int64") {
    return DType::kInt64;
  }
  return std::unexpected("Unsupported dtype: " + std::string(text));
}

}  // namespace longwhisper
