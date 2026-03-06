#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace longwhisper {

struct VadConfig {
  int frame_length = 2048;
  int hop_length = 512;
  float amin = 1e-5f;
  float top_db = 60.0f;
  float low_confidence_threshold = 0.35f;
};

struct VadSegment {
  int64_t frame_begin = 0;
  int64_t frame_end = 0;  // Exclusive.
  int64_t sample_begin = 0;
  int64_t sample_end = 0;  // Exclusive.

  float mean_db = 0.0f;
  float peak_db = 0.0f;
  float priority_score = 0.0f;
  bool low_confidence = false;
};

struct VadResult {
  int sample_rate_hz = 0;
  int frame_length = 0;
  int hop_length = 0;

  std::vector<float> frame_rms;
  std::vector<float> frame_db;
  std::vector<float> noise_floor_db;
  std::vector<float> confidence;
  std::vector<uint8_t> speech_mask;

  std::vector<VadSegment> segments;
};

using RunVadResult = std::expected<VadResult, std::string>;

// Mono volume VAD modeled after librosa.effects.split:
// centered RMS, peak-relative dB, and frame-edge interval conversion.
RunVadResult RunVolumeVad(std::span<const float> mono_pcm,
                          int sample_rate_hz,
                          const VadConfig& config = {});

}  // namespace longwhisper
