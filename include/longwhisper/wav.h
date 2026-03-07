#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace longwhisper {

struct WavMetadata {
  int sample_rate_hz = 0;
  int channel_count = 0;
  int bits_per_sample = 0;
  int64_t frame_count = 0;
  double duration_seconds = 0.0;
};

struct WavFile {
  WavMetadata metadata;
  std::vector<std::vector<float>> channel_audio;
};

using LoadWavResult = std::expected<WavFile, std::string>;

LoadWavResult LoadWav(const std::filesystem::path& path);

}  // namespace longwhisper
