#include "longwhisper/vad.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace longwhisper {
namespace {

float Clamp(float value, float low, float high) {
  return std::max(low, std::min(high, value));
}

float DbFromAmplitudeRelative(float amplitude, float reference, float amin) {
  const float safe_amplitude = std::max(amplitude, amin);
  const float safe_reference = std::max(reference, amin);
  return 20.0f * std::log10(safe_amplitude) - 20.0f * std::log10(safe_reference);
}

std::expected<void, std::string> ValidateConfig(const VadConfig& config) {
  if (config.frame_length <= 0) {
    return std::unexpected("VadConfig.frame_length must be > 0");
  }
  if (config.hop_length <= 0) {
    return std::unexpected("VadConfig.hop_length must be > 0");
  }
  if (config.amin <= 0.0f) {
    return std::unexpected("VadConfig.amin must be > 0");
  }
  if (!std::isfinite(config.top_db) || config.top_db < 0.0f) {
    return std::unexpected("VadConfig.top_db must be finite and >= 0");
  }
  if (config.low_confidence_threshold < 0.0f ||
      config.low_confidence_threshold > 1.0f) {
    return std::unexpected(
        "VadConfig.low_confidence_threshold must be in [0, 1]");
  }
  return {};
}

std::expected<size_t, std::string> ComputeFrameCount(int64_t sample_count,
                                                     int frame_length,
                                                     int hop_length) {
  const int64_t pad = frame_length / 2;
  const int64_t padded_sample_count = sample_count + 2 * pad;
  if (padded_sample_count < frame_length) {
    return std::unexpected("Input is too short for centered frame analysis");
  }

  return static_cast<size_t>(
      (padded_sample_count - frame_length) / hop_length + 1);
}

float ComputeCenteredFrameRms(std::span<const float> mono_pcm,
                              int64_t sample_count,
                              int frame_length,
                              int hop_length,
                              size_t frame_index) {
  const int64_t centered_start =
      static_cast<int64_t>(frame_index) * hop_length - frame_length / 2;
  const int64_t centered_end = centered_start + frame_length;

  const int64_t copy_begin = std::max<int64_t>(0, centered_start);
  const int64_t copy_end = std::min(centered_end, sample_count);

  double sum_sq = 0.0;
  for (int64_t i = copy_begin; i < copy_end; ++i) {
    const double value = static_cast<double>(mono_pcm[static_cast<size_t>(i)]);
    sum_sq += value * value;
  }

  return static_cast<float>(std::sqrt(sum_sq / frame_length));
}

std::vector<VadSegment> BuildSegments(const std::vector<uint8_t>& mask,
                                      const std::vector<float>& frame_db,
                                      const std::vector<float>& confidence,
                                      int hop_length,
                                      int64_t sample_count,
                                      float low_confidence_threshold) {
  std::vector<int64_t> edges;
  edges.reserve(mask.size() + 1);

  for (size_t i = 0; i + 1 < mask.size(); ++i) {
    if (mask[i] != mask[i + 1]) {
      edges.push_back(static_cast<int64_t>(i) + 1);
    }
  }

  if (!mask.empty() && mask.front() != 0) {
    edges.insert(edges.begin(), 0);
  }
  if (!mask.empty() && mask.back() != 0) {
    edges.push_back(static_cast<int64_t>(mask.size()));
  }

  std::vector<VadSegment> segments;
  segments.reserve(edges.size() / 2);

  for (size_t i = 0; i + 1 < edges.size(); i += 2) {
    const int64_t frame_begin = edges[i];
    const int64_t frame_end = edges[i + 1];

    float sum_db = 0.0f;
    float peak_db = -std::numeric_limits<float>::infinity();
    float sum_conf = 0.0f;
    float peak_conf = 0.0f;

    for (int64_t frame = frame_begin; frame < frame_end; ++frame) {
      const size_t index = static_cast<size_t>(frame);
      sum_db += frame_db[index];
      peak_db = std::max(peak_db, frame_db[index]);
      sum_conf += confidence[index];
      peak_conf = std::max(peak_conf, confidence[index]);
    }

    const float inv_frames = 1.0f / static_cast<float>(frame_end - frame_begin);
    const float mean_db = sum_db * inv_frames;
    const float mean_conf = sum_conf * inv_frames;

    VadSegment segment;
    segment.frame_begin = frame_begin;
    segment.frame_end = frame_end;
    segment.sample_begin =
        std::min(sample_count, frame_begin * static_cast<int64_t>(hop_length));
    segment.sample_end =
        std::min(sample_count, frame_end * static_cast<int64_t>(hop_length));
    segment.mean_db = mean_db;
    segment.peak_db = peak_db;
    segment.priority_score =
        Clamp(0.7f * mean_conf + 0.3f * peak_conf, 0.0f, 1.0f);
    segment.low_confidence =
        segment.priority_score < low_confidence_threshold;

    segments.push_back(segment);
  }

  return segments;
}

}  // namespace

RunVadResult RunVolumeVad(std::span<const float> mono_pcm,
                          int sample_rate_hz,
                          const VadConfig& config) {
  if (sample_rate_hz <= 0) {
    return std::unexpected("sample_rate_hz must be > 0");
  }

  if (auto config_status = ValidateConfig(config); !config_status.has_value()) {
    return std::unexpected(std::move(config_status.error()));
  }

  VadResult result;
  result.sample_rate_hz = sample_rate_hz;
  result.frame_length = config.frame_length;
  result.hop_length = config.hop_length;

  const int64_t sample_count = static_cast<int64_t>(mono_pcm.size());
  const auto frame_count_status = ComputeFrameCount(sample_count,
                                                    result.frame_length,
                                                    result.hop_length);
  if (!frame_count_status.has_value()) {
    return std::unexpected(std::move(frame_count_status.error()));
  }
  const size_t frame_count = *frame_count_status;

  result.frame_rms.resize(frame_count, 0.0f);
  result.frame_db.resize(frame_count, 0.0f);
  result.noise_floor_db.resize(frame_count, -config.top_db);
  result.confidence.resize(frame_count, 0.0f);
  result.speech_mask.resize(frame_count, 0);

  for (size_t frame = 0; frame < frame_count; ++frame) {
    result.frame_rms[frame] = ComputeCenteredFrameRms(mono_pcm,
                                                      sample_count,
                                                      result.frame_length,
                                                      result.hop_length,
                                                      frame);
  }

  const float reference =
      *std::max_element(result.frame_rms.begin(), result.frame_rms.end());
  const float confidence_scale = std::max(config.top_db, 1e-12f);

  for (size_t frame = 0; frame < frame_count; ++frame) {
    const float db =
        DbFromAmplitudeRelative(result.frame_rms[frame], reference, config.amin);
    result.frame_db[frame] = db;
    result.speech_mask[frame] = (db > -config.top_db) ? 1 : 0;
    result.confidence[frame] =
        Clamp((db + config.top_db) / confidence_scale, 0.0f, 1.0f);
  }

  result.segments = BuildSegments(result.speech_mask,
                                  result.frame_db,
                                  result.confidence,
                                  result.hop_length,
                                  sample_count,
                                  config.low_confidence_threshold);

  return result;
}

}  // namespace longwhisper
