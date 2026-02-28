#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "longwhisper/dtype.h"

namespace longwhisper {

struct TensorMeta {
  std::string name;
  DType dtype = DType::kUnknown;
  std::vector<int64_t> shape;
  uint64_t offset = 0;
  uint64_t nbytes = 0;
  std::string source_file;
};

struct ModelConstants {
  int d_model = 0;
  int n_heads = 0;
  int encoder_layers = 0;
  int decoder_layers = 0;
  int vocab_size = 0;
  int encoder_ctx = 0;
  int decoder_ctx = 0;
};

struct AudioConstants {
  int sample_rate = 0;
  int n_fft = 0;
  int hop_length = 0;
  int chunk_length_sec = 0;
  int mel_bins = 0;
  int nb_max_frames = 0;
};

struct TokenIds {
  int ja = 0;
  int transcribe = 0;
  int startoftranscript = 0;
  int no_timestamps = 0;
  int bos = 0;
  int eos = 0;
  std::vector<int> fixed_ja_prefix_ids;
};

struct RuntimeAssets {
  ModelConstants model;
  AudioConstants audio;
  TokenIds tokens;
  std::unordered_map<std::string, TensorMeta> tensors;
  std::filesystem::path weights_path;
  uint64_t weights_size_bytes = 0;
  uint64_t alignment = 0;
};

using LoadAssetsResult = std::expected<RuntimeAssets, std::string>;

LoadAssetsResult LoadAssets(const std::filesystem::path& model_dir,
                            const std::filesystem::path& packed_dir);

}  // namespace longwhisper
