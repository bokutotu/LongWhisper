#include "longwhisper/assets_config.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>

namespace longwhisper {

namespace {

namespace fs = std::filesystem;

constexpr char kManifestFormat[] = "longwhisper.packed_weights.v1";
constexpr int kExpectedDModel = 1280;
constexpr int kExpectedHeads = 20;
constexpr int kExpectedEncoderLayers = 32;
constexpr int kExpectedDecoderLayers = 32;
constexpr int kExpectedVocabSize = 51866;
constexpr int kExpectedEncoderCtx = 1500;
constexpr int kExpectedDecoderCtx = 448;
constexpr int kExpectedSampleRate = 16000;
constexpr int kExpectedNfft = 400;
constexpr int kExpectedHopLength = 160;
constexpr int kExpectedChunkLength = 30;
constexpr int kExpectedMelBins = 128;
constexpr int kExpectedMaxFrames = 3000;

constexpr int kExpectedJaToken = 50266;
constexpr int kExpectedTranscribeToken = 50360;
constexpr int kExpectedStartToken = 50258;
constexpr int kExpectedNoTimestampsToken = 50364;
constexpr int kExpectedEndOfTextToken = 50257;

LoadAssetsResult Error(std::string message) {
  return std::unexpected(std::move(message));
}

std::string PathString(const fs::path& path) {
  return path.string();
}

bool ParseJsonFile(const fs::path& path, rapidjson::Document* doc,
                   std::string* error) {
  std::ifstream in(path);
  if (!in.is_open()) {
    *error = "Failed to open JSON file: " + PathString(path);
    return false;
  }

  rapidjson::IStreamWrapper stream(in);
  doc->ParseStream(stream);
  if (doc->HasParseError()) {
    std::ostringstream oss;
    oss << "Failed to parse JSON file: " << PathString(path) << " ("
        << rapidjson::GetParseError_En(doc->GetParseError()) << " at offset "
        << doc->GetErrorOffset() << ")";
    *error = oss.str();
    return false;
  }
  if (!doc->IsObject()) {
    *error = "JSON root must be object: " + PathString(path);
    return false;
  }
  return true;
}

bool RequireMember(const rapidjson::Value& obj, std::string_view key,
                   const rapidjson::Value** out, std::string* error,
                   std::string_view context) {
  auto it = obj.FindMember(rapidjson::StringRef(key.data(), key.size()));
  if (it == obj.MemberEnd()) {
    *error = std::string(context) + " missing required key: " +
             std::string(key);
    return false;
  }
  *out = &it->value;
  return true;
}

bool ReadInt(const rapidjson::Value& obj, std::string_view key, int* out,
             std::string* error, std::string_view context) {
  const rapidjson::Value* value = nullptr;
  if (!RequireMember(obj, key, &value, error, context)) {
    return false;
  }
  if (!value->IsInt()) {
    *error = std::string(context) + " key '" + std::string(key) +
             "' must be int";
    return false;
  }
  *out = value->GetInt();
  return true;
}

bool ReadUint64(const rapidjson::Value& obj, std::string_view key, uint64_t* out,
                std::string* error, std::string_view context) {
  const rapidjson::Value* value = nullptr;
  if (!RequireMember(obj, key, &value, error, context)) {
    return false;
  }
  if (!value->IsUint64()) {
    *error = std::string(context) + " key '" + std::string(key) +
             "' must be uint64";
    return false;
  }
  *out = value->GetUint64();
  return true;
}

bool ReadString(const rapidjson::Value& obj, std::string_view key,
                std::string* out, std::string* error,
                std::string_view context) {
  const rapidjson::Value* value = nullptr;
  if (!RequireMember(obj, key, &value, error, context)) {
    return false;
  }
  if (!value->IsString()) {
    *error = std::string(context) + " key '" + std::string(key) +
             "' must be string";
    return false;
  }
  *out = value->GetString();
  return true;
}

bool ExpectEqual(std::string_view label, int actual, int expected,
                 std::string* error) {
  if (actual == expected) {
    return true;
  }
  std::ostringstream oss;
  oss << "Unexpected value for " << label << ": got " << actual
      << ", expected " << expected;
  *error = oss.str();
  return false;
}

bool ReadTokenId(const rapidjson::Value& added_tokens, std::string_view token,
                 int* out, std::string* error) {
  if (!added_tokens.IsObject()) {
    *error = "added_tokens.json root must be object";
    return false;
  }
  auto it = added_tokens.FindMember(rapidjson::StringRef(token.data(), token.size()));
  if (it == added_tokens.MemberEnd()) {
    *error = "added_tokens.json missing token: " + std::string(token);
    return false;
  }
  if (!it->value.IsInt()) {
    *error = "added_tokens.json token must map to int: " + std::string(token);
    return false;
  }
  *out = it->value.GetInt();
  return true;
}

bool CheckExistsRegularFile(const fs::path& path, std::string* error) {
  if (!fs::exists(path)) {
    *error = "Missing required file: " + PathString(path);
    return false;
  }
  if (!fs::is_regular_file(path)) {
    *error = "Required path is not a regular file: " + PathString(path);
    return false;
  }
  return true;
}

}  // namespace

LoadAssetsResult LoadAssets(const fs::path& model_dir, const fs::path& packed_dir) {
  const fs::path config_path = model_dir / "config.json";
  const fs::path generation_config_path = model_dir / "generation_config.json";
  const fs::path preprocessor_config_path = model_dir / "preprocessor_config.json";
  const fs::path added_tokens_path = model_dir / "added_tokens.json";
  const fs::path manifest_path = packed_dir / "manifest.json";

  std::string error;
  for (const auto& required : {config_path, generation_config_path,
                               preprocessor_config_path, added_tokens_path,
                               manifest_path}) {
    if (!CheckExistsRegularFile(required, &error)) {
      return Error(std::move(error));
    }
  }

  rapidjson::Document config_doc;
  rapidjson::Document generation_doc;
  rapidjson::Document preprocessor_doc;
  rapidjson::Document added_tokens_doc;
  rapidjson::Document manifest_doc;

  if (!ParseJsonFile(config_path, &config_doc, &error)) {
    return Error(std::move(error));
  }
  if (!ParseJsonFile(generation_config_path, &generation_doc, &error)) {
    return Error(std::move(error));
  }
  if (!ParseJsonFile(preprocessor_config_path, &preprocessor_doc, &error)) {
    return Error(std::move(error));
  }
  if (!ParseJsonFile(added_tokens_path, &added_tokens_doc, &error)) {
    return Error(std::move(error));
  }
  if (!ParseJsonFile(manifest_path, &manifest_doc, &error)) {
    return Error(std::move(error));
  }

  RuntimeAssets assets;

  if (!ReadInt(config_doc, "d_model", &assets.model.d_model, &error,
               "config.json")) {
    return Error(std::move(error));
  }
  int encoder_heads = 0;
  int decoder_heads = 0;
  if (!ReadInt(config_doc, "encoder_attention_heads", &encoder_heads, &error,
               "config.json")) {
    return Error(std::move(error));
  }
  if (!ReadInt(config_doc, "decoder_attention_heads", &decoder_heads, &error,
               "config.json")) {
    return Error(std::move(error));
  }
  assets.model.n_heads = encoder_heads;

  if (!ReadInt(config_doc, "encoder_layers", &assets.model.encoder_layers, &error,
               "config.json")) {
    return Error(std::move(error));
  }
  if (!ReadInt(config_doc, "decoder_layers", &assets.model.decoder_layers, &error,
               "config.json")) {
    return Error(std::move(error));
  }
  if (!ReadInt(config_doc, "vocab_size", &assets.model.vocab_size, &error,
               "config.json")) {
    return Error(std::move(error));
  }
  if (!ReadInt(config_doc, "max_source_positions", &assets.model.encoder_ctx,
               &error, "config.json")) {
    return Error(std::move(error));
  }
  if (!ReadInt(config_doc, "max_target_positions", &assets.model.decoder_ctx,
               &error, "config.json")) {
    return Error(std::move(error));
  }

  if (!ExpectEqual("config.d_model", assets.model.d_model, kExpectedDModel,
                   &error)) {
    return Error(std::move(error));
  }
  if (!ExpectEqual("config.encoder_attention_heads", encoder_heads, kExpectedHeads,
                   &error)) {
    return Error(std::move(error));
  }
  if (!ExpectEqual("config.decoder_attention_heads", decoder_heads, kExpectedHeads,
                   &error)) {
    return Error(std::move(error));
  }
  if (!ExpectEqual("config.encoder_layers", assets.model.encoder_layers,
                   kExpectedEncoderLayers, &error)) {
    return Error(std::move(error));
  }
  if (!ExpectEqual("config.decoder_layers", assets.model.decoder_layers,
                   kExpectedDecoderLayers, &error)) {
    return Error(std::move(error));
  }
  if (!ExpectEqual("config.vocab_size", assets.model.vocab_size,
                   kExpectedVocabSize, &error)) {
    return Error(std::move(error));
  }
  if (!ExpectEqual("config.max_source_positions", assets.model.encoder_ctx,
                   kExpectedEncoderCtx, &error)) {
    return Error(std::move(error));
  }
  if (!ExpectEqual("config.max_target_positions", assets.model.decoder_ctx,
                   kExpectedDecoderCtx, &error)) {
    return Error(std::move(error));
  }

  if (!ReadInt(preprocessor_doc, "sampling_rate", &assets.audio.sample_rate,
               &error, "preprocessor_config.json")) {
    return Error(std::move(error));
  }
  if (!ReadInt(preprocessor_doc, "n_fft", &assets.audio.n_fft, &error,
               "preprocessor_config.json")) {
    return Error(std::move(error));
  }
  if (!ReadInt(preprocessor_doc, "hop_length", &assets.audio.hop_length, &error,
               "preprocessor_config.json")) {
    return Error(std::move(error));
  }
  if (!ReadInt(preprocessor_doc, "chunk_length", &assets.audio.chunk_length_sec,
               &error, "preprocessor_config.json")) {
    return Error(std::move(error));
  }
  if (!ReadInt(preprocessor_doc, "feature_size", &assets.audio.mel_bins, &error,
               "preprocessor_config.json")) {
    return Error(std::move(error));
  }
  if (!ReadInt(preprocessor_doc, "nb_max_frames", &assets.audio.nb_max_frames,
               &error, "preprocessor_config.json")) {
    return Error(std::move(error));
  }

  if (!ExpectEqual("preprocessor.sampling_rate", assets.audio.sample_rate,
                   kExpectedSampleRate, &error)) {
    return Error(std::move(error));
  }
  if (!ExpectEqual("preprocessor.n_fft", assets.audio.n_fft, kExpectedNfft,
                   &error)) {
    return Error(std::move(error));
  }
  if (!ExpectEqual("preprocessor.hop_length", assets.audio.hop_length,
                   kExpectedHopLength, &error)) {
    return Error(std::move(error));
  }
  if (!ExpectEqual("preprocessor.chunk_length", assets.audio.chunk_length_sec,
                   kExpectedChunkLength, &error)) {
    return Error(std::move(error));
  }
  if (!ExpectEqual("preprocessor.feature_size", assets.audio.mel_bins,
                   kExpectedMelBins, &error)) {
    return Error(std::move(error));
  }
  if (!ExpectEqual("preprocessor.nb_max_frames", assets.audio.nb_max_frames,
                   kExpectedMaxFrames, &error)) {
    return Error(std::move(error));
  }

  if (!ReadInt(generation_doc, "bos_token_id", &assets.tokens.bos, &error,
               "generation_config.json")) {
    return Error(std::move(error));
  }
  if (!ReadInt(generation_doc, "eos_token_id", &assets.tokens.eos, &error,
               "generation_config.json")) {
    return Error(std::move(error));
  }
  if (!ReadInt(generation_doc, "no_timestamps_token_id",
               &assets.tokens.no_timestamps, &error, "generation_config.json")) {
    return Error(std::move(error));
  }

  if (!ReadTokenId(added_tokens_doc, "<|ja|>", &assets.tokens.ja, &error)) {
    return Error(std::move(error));
  }
  if (!ReadTokenId(added_tokens_doc, "<|transcribe|>", &assets.tokens.transcribe,
                   &error)) {
    return Error(std::move(error));
  }
  if (!ReadTokenId(added_tokens_doc, "<|startoftranscript|>",
                   &assets.tokens.startoftranscript, &error)) {
    return Error(std::move(error));
  }
  int added_no_timestamps = 0;
  if (!ReadTokenId(added_tokens_doc, "<|notimestamps|>", &added_no_timestamps,
                   &error)) {
    return Error(std::move(error));
  }
  int end_of_text = 0;
  if (!ReadTokenId(added_tokens_doc, "<|endoftext|>", &end_of_text, &error)) {
    return Error(std::move(error));
  }

  if (!ExpectEqual("token <|ja|>", assets.tokens.ja, kExpectedJaToken, &error)) {
    return Error(std::move(error));
  }
  if (!ExpectEqual("token <|transcribe|>", assets.tokens.transcribe,
                   kExpectedTranscribeToken, &error)) {
    return Error(std::move(error));
  }
  if (!ExpectEqual("token <|startoftranscript|>", assets.tokens.startoftranscript,
                   kExpectedStartToken, &error)) {
    return Error(std::move(error));
  }
  if (!ExpectEqual("token <|notimestamps|>", assets.tokens.no_timestamps,
                   kExpectedNoTimestampsToken, &error)) {
    return Error(std::move(error));
  }
  if (!ExpectEqual("added token <|notimestamps|>", added_no_timestamps,
                   kExpectedNoTimestampsToken, &error)) {
    return Error(std::move(error));
  }
  if (!ExpectEqual("token <|endoftext|>", end_of_text, kExpectedEndOfTextToken,
                   &error)) {
    return Error(std::move(error));
  }
  if (!ExpectEqual("bos_token_id", assets.tokens.bos, kExpectedEndOfTextToken,
                   &error)) {
    return Error(std::move(error));
  }
  if (!ExpectEqual("eos_token_id", assets.tokens.eos, kExpectedEndOfTextToken,
                   &error)) {
    return Error(std::move(error));
  }

  assets.tokens.fixed_ja_prefix_ids = {
      assets.tokens.startoftranscript,
      assets.tokens.ja,
      assets.tokens.transcribe,
      assets.tokens.no_timestamps,
  };

  std::string manifest_format;
  if (!ReadString(manifest_doc, "format", &manifest_format, &error,
                  "manifest.json")) {
    return Error(std::move(error));
  }
  if (manifest_format != kManifestFormat) {
    return Error("Unsupported manifest format: " + manifest_format);
  }
  if (!ReadUint64(manifest_doc, "alignment", &assets.alignment, &error,
                  "manifest.json")) {
    return Error(std::move(error));
  }
  if (assets.alignment == 0) {
    return Error("manifest.json alignment must be > 0");
  }

  std::string weights_file;
  if (!ReadString(manifest_doc, "weights_file", &weights_file, &error,
                  "manifest.json")) {
    return Error(std::move(error));
  }
  assets.weights_path = fs::absolute(packed_dir / weights_file);
  if (!CheckExistsRegularFile(assets.weights_path, &error)) {
    return Error(std::move(error));
  }
  assets.weights_size_bytes = fs::file_size(assets.weights_path);

  uint64_t num_tensors = 0;
  if (!ReadUint64(manifest_doc, "num_tensors", &num_tensors, &error,
                  "manifest.json")) {
    return Error(std::move(error));
  }
  const rapidjson::Value* tensors = nullptr;
  if (!RequireMember(manifest_doc, "tensors", &tensors, &error,
                     "manifest.json")) {
    return Error(std::move(error));
  }
  if (!tensors->IsArray()) {
    return Error("manifest.json key 'tensors' must be array");
  }
  if (num_tensors != tensors->Size()) {
    std::ostringstream oss;
    oss << "manifest.json num_tensors mismatch: field=" << num_tensors
        << ", actual=" << tensors->Size();
    return Error(oss.str());
  }

  struct Interval {
    uint64_t begin = 0;
    uint64_t end = 0;
    std::string name;
  };
  std::vector<Interval> ranges;
  ranges.reserve(tensors->Size());
  assets.tensors.reserve(tensors->Size());

  for (rapidjson::SizeType i = 0; i < tensors->Size(); ++i) {
    const auto& value = (*tensors)[i];
    if (!value.IsObject()) {
      return Error("manifest.json tensor entry must be object");
    }

    TensorMeta tensor;
    if (!ReadString(value, "name", &tensor.name, &error, "manifest tensor")) {
      return Error(std::move(error));
    }
    std::string dtype_text;
    if (!ReadString(value, "dtype", &dtype_text, &error, "manifest tensor")) {
      return Error(std::move(error));
    }
    const auto parsed_dtype = ParseDType(dtype_text);
    if (!parsed_dtype.has_value()) {
      std::ostringstream oss;
      oss << parsed_dtype.error() << " for tensor " << tensor.name;
      return Error(oss.str());
    }
    tensor.dtype = *parsed_dtype;
    if (!ReadUint64(value, "offset", &tensor.offset, &error, "manifest tensor")) {
      return Error(std::move(error));
    }
    if (!ReadUint64(value, "nbytes", &tensor.nbytes, &error, "manifest tensor")) {
      return Error(std::move(error));
    }
    if (!ReadString(value, "source_file", &tensor.source_file, &error,
                    "manifest tensor")) {
      return Error(std::move(error));
    }

    const rapidjson::Value* shape = nullptr;
    if (!RequireMember(value, "shape", &shape, &error, "manifest tensor")) {
      return Error(std::move(error));
    }
    if (!shape->IsArray()) {
      return Error("manifest tensor key 'shape' must be array");
    }
    tensor.shape.reserve(shape->Size());
    for (rapidjson::SizeType j = 0; j < shape->Size(); ++j) {
      if (!(*shape)[j].IsInt64()) {
        return Error("manifest tensor shape values must be int64");
      }
      const int64_t dim = (*shape)[j].GetInt64();
      if (dim <= 0) {
        return Error("manifest tensor shape values must be > 0");
      }
      tensor.shape.push_back(dim);
    }

    if ((tensor.offset % assets.alignment) != 0) {
      std::ostringstream oss;
      oss << "Tensor offset is not aligned (" << assets.alignment
          << "): " << tensor.name;
      return Error(oss.str());
    }
    if (tensor.offset > assets.weights_size_bytes ||
        tensor.nbytes > assets.weights_size_bytes - tensor.offset) {
      std::ostringstream oss;
      oss << "Tensor range out of bounds for weights.bin: " << tensor.name;
      return Error(oss.str());
    }

    auto [it, inserted] = assets.tensors.emplace(tensor.name, tensor);
    if (!inserted) {
      return Error("Duplicate tensor name in manifest: " + tensor.name);
    }

    ranges.push_back(
        Interval{tensor.offset, tensor.offset + tensor.nbytes, tensor.name});
  }

  std::sort(ranges.begin(), ranges.end(),
            [](const Interval& a, const Interval& b) { return a.begin < b.begin; });
  for (size_t i = 1; i < ranges.size(); ++i) {
    if (ranges[i].begin < ranges[i - 1].end) {
      return Error("Overlapping tensor ranges in manifest: " + ranges[i - 1].name +
                   " and " + ranges[i].name);
    }
  }

  return assets;
}

}  // namespace longwhisper
