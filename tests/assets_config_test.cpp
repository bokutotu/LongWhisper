#include "longwhisper/assets_config.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace longwhisper {
namespace {

namespace fs = std::filesystem;

fs::path SourceRoot() { return fs::path(LONGWHISPER_SOURCE_DIR); }

fs::path MakeTempDir(const std::string& label) {
  const auto seed = static_cast<uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  const fs::path dir = fs::temp_directory_path() / "longwhisper_tests" /
                       (label + "_" + std::to_string(seed));
  fs::create_directories(dir);
  return dir;
}

void WriteText(const fs::path& path, const std::string& content) {
  std::ofstream out(path);
  ASSERT_TRUE(out.is_open()) << "Failed to open " << path.string();
  out << content;
  ASSERT_TRUE(out.good()) << "Failed to write " << path.string();
}

void WriteBinary(const fs::path& path, size_t bytes) {
  std::ofstream out(path, std::ios::binary);
  ASSERT_TRUE(out.is_open()) << "Failed to open " << path.string();
  std::vector<char> data(bytes, '\0');
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
  ASSERT_TRUE(out.good()) << "Failed to write " << path.string();
}

TEST(AssetsConfigTest, LoadsRealLargeV3Assets) {
  const fs::path root = SourceRoot();
  const fs::path model_dir = root / "third_party/whisper-large-v3";
  const fs::path packed_dir = root / "artifacts/whisper-large-v3-packed";

  const LoadAssetsResult result = LoadAssets(model_dir, packed_dir);
  ASSERT_TRUE(result.has_value()) << result.error();

  EXPECT_EQ(result->model.d_model, 1280);
  EXPECT_EQ(result->model.n_heads, 20);
  EXPECT_EQ(result->model.encoder_layers, 32);
  EXPECT_EQ(result->model.decoder_layers, 32);
  EXPECT_EQ(result->model.vocab_size, 51866);
  EXPECT_EQ(result->model.encoder_ctx, 1500);
  EXPECT_EQ(result->model.decoder_ctx, 448);

  EXPECT_EQ(result->audio.sample_rate, 16000);
  EXPECT_EQ(result->audio.n_fft, 400);
  EXPECT_EQ(result->audio.hop_length, 160);
  EXPECT_EQ(result->audio.chunk_length_sec, 30);
  EXPECT_EQ(result->audio.mel_bins, 128);
  EXPECT_EQ(result->audio.nb_max_frames, 3000);

  EXPECT_EQ(result->tokens.ja, 50266);
  EXPECT_EQ(result->tokens.transcribe, 50360);
  EXPECT_EQ(result->tokens.startoftranscript, 50258);
  EXPECT_EQ(result->tokens.no_timestamps, 50364);
  EXPECT_EQ(result->tokens.bos, 50257);
  EXPECT_EQ(result->tokens.eos, 50257);

  const std::vector<int> expected_prefix = {50258, 50266, 50360, 50364};
  EXPECT_EQ(result->tokens.fixed_ja_prefix_ids, expected_prefix);

  EXPECT_GT(result->tensors.size(), 1000u);
  EXPECT_GT(result->weights_size_bytes, 0u);
  ASSERT_TRUE(result->weights_path.is_absolute());

  const auto it = result->tensors.find("model.decoder.embed_tokens.weight");
  ASSERT_NE(it, result->tensors.end());
  EXPECT_EQ(it->second.dtype, DType::kFloat16);
  EXPECT_STREQ(DTypeName(it->second.dtype), "float16");
  EXPECT_EQ(it->second.shape.size(), 2u);
  EXPECT_EQ(it->second.shape[0], 51866);
  EXPECT_EQ(it->second.shape[1], 1280);
}

TEST(AssetsConfigTest, MissingFileFailsFast) {
  const fs::path root = SourceRoot();
  const fs::path missing_model_dir = root / "third_party/does-not-exist";
  const fs::path packed_dir = root / "artifacts/whisper-large-v3-packed";

  const LoadAssetsResult result = LoadAssets(missing_model_dir, packed_dir);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().find("config.json"), std::string::npos);
}

TEST(AssetsConfigTest, CorruptManifestOutOfBoundsFails) {
  const fs::path root = SourceRoot();
  const fs::path model_dir = root / "third_party/whisper-large-v3";
  const fs::path packed_dir = MakeTempDir("packed_corrupt");

  WriteBinary(packed_dir / "weights.bin", 64);
  WriteText(
      packed_dir / "manifest.json",
      R"json({
  "format": "longwhisper.packed_weights.v1",
  "weights_file": "weights.bin",
  "alignment": 64,
  "num_tensors": 1,
  "tensors": [
    {
      "name": "dummy.weight",
      "dtype": "float16",
      "shape": [1],
      "offset": 64,
      "nbytes": 64,
      "source_file": "dummy.safetensors"
    }
  ]
}
)json");

  const LoadAssetsResult result = LoadAssets(model_dir, packed_dir);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().find("out of bounds"), std::string::npos);
}

TEST(AssetsConfigTest, UnsupportedDTypeFails) {
  const fs::path root = SourceRoot();
  const fs::path model_dir = root / "third_party/whisper-large-v3";
  const fs::path packed_dir = MakeTempDir("packed_bad_dtype");

  WriteBinary(packed_dir / "weights.bin", 64);
  WriteText(
      packed_dir / "manifest.json",
      R"json({
  "format": "longwhisper.packed_weights.v1",
  "weights_file": "weights.bin",
  "alignment": 64,
  "num_tensors": 1,
  "tensors": [
    {
      "name": "dummy.weight",
      "dtype": "uint8",
      "shape": [1],
      "offset": 0,
      "nbytes": 8,
      "source_file": "dummy.safetensors"
    }
  ]
}
)json");

  const LoadAssetsResult result = LoadAssets(model_dir, packed_dir);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().find("Unsupported dtype"),
            std::string::npos);
}

TEST(AssetsConfigTest, TokenMismatchFails) {
  const fs::path root = SourceRoot();
  const fs::path model_dir = MakeTempDir("model_token_mismatch");
  const fs::path packed_dir = root / "artifacts/whisper-large-v3-packed";

  WriteText(
      model_dir / "config.json",
      R"json({
  "d_model": 1280,
  "encoder_attention_heads": 20,
  "decoder_attention_heads": 20,
  "encoder_layers": 32,
  "decoder_layers": 32,
  "vocab_size": 51866,
  "max_source_positions": 1500,
  "max_target_positions": 448
}
)json");
  WriteText(
      model_dir / "generation_config.json",
      R"json({
  "bos_token_id": 50257,
  "eos_token_id": 50257,
  "no_timestamps_token_id": 50364
}
)json");
  WriteText(
      model_dir / "preprocessor_config.json",
      R"json({
  "sampling_rate": 16000,
  "n_fft": 400,
  "hop_length": 160,
  "chunk_length": 30,
  "feature_size": 128,
  "nb_max_frames": 3000
}
)json");
  WriteText(
      model_dir / "added_tokens.json",
      R"json({
  "<|ja|>": 12345,
  "<|transcribe|>": 50360,
  "<|startoftranscript|>": 50258,
  "<|notimestamps|>": 50364,
  "<|endoftext|>": 50257
}
)json");

  const LoadAssetsResult result = LoadAssets(model_dir, packed_dir);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().find("<|ja|>"), std::string::npos);
}

}  // namespace
}  // namespace longwhisper
