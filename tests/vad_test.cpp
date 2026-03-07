#include "longwhisper/vad.h"
#include "longwhisper/wav.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace longwhisper {
namespace {

constexpr double kPi = 3.14159265358979323846;

struct SampleInterval {
  int64_t begin = 0;
  int64_t end = 0;
};

std::filesystem::path SourceRoot() {
  return std::filesystem::path(LONGWHISPER_SOURCE_DIR);
}

std::filesystem::path RecitationWavPath() {
  return SourceRoot() / "RECITATION324_158.wav";
}

std::filesystem::path RecitationReferencePath() {
  return SourceRoot() / "tests" / "data" /
         "RECITATION324_158.librosa_split.txt";
}

std::expected<std::vector<SampleInterval>, std::string> LoadReferenceIntervals(
    const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    return std::unexpected("Failed to open interval file: " + path.string());
  }

  std::vector<SampleInterval> intervals;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::istringstream parser(line);
    SampleInterval interval;
    if (!(parser >> interval.begin >> interval.end) || interval.begin < 0 ||
        interval.end < interval.begin) {
      return std::unexpected("Malformed interval line in " + path.string() +
                             ": " + line);
    }
    intervals.push_back(interval);
  }

  if (intervals.empty()) {
    return std::unexpected("Interval file was empty: " + path.string());
  }
  return intervals;
}

std::vector<SampleInterval> ExtractIntervals(const VadResult& result) {
  std::vector<SampleInterval> intervals;
  intervals.reserve(result.segments.size());
  for (const VadSegment& segment : result.segments) {
    intervals.push_back({segment.sample_begin, segment.sample_end});
  }
  return intervals;
}

void AppendSilence(std::vector<float>* audio, int sample_rate_hz,
                   float seconds) {
  const size_t count =
      static_cast<size_t>(std::lround(seconds * sample_rate_hz));
  audio->insert(audio->end(), count, 0.0f);
}

void AppendTone(std::vector<float>* audio, int sample_rate_hz, float seconds,
                float amplitude, float frequency_hz) {
  const size_t count =
      static_cast<size_t>(std::lround(seconds * sample_rate_hz));
  const size_t offset = audio->size();
  audio->resize(offset + count, 0.0f);

  for (size_t i = 0; i < count; ++i) {
    const double t = static_cast<double>(i) / sample_rate_hz;
    const double value = amplitude * std::sin(2.0 * kPi * frequency_hz * t);
    (*audio)[offset + i] = static_cast<float>(value);
  }
}

VadConfig SyntheticConfig() {
  VadConfig config;
  config.frame_length = 400;
  config.hop_length = 160;
  config.amin = 1e-5f;
  config.top_db = 40.0f;
  config.low_confidence_threshold = 0.35f;
  return config;
}

TEST(VadTest, EmptyInputMatchesLibrosaZeroLengthInterval) {
  const std::vector<float> audio;
  const auto result = RunVolumeVad(audio, 16000, VadConfig{});

  ASSERT_TRUE(result.has_value()) << result.error();
  ASSERT_EQ(result->frame_rms.size(), 1u);
  ASSERT_EQ(result->frame_db.size(), 1u);
  ASSERT_EQ(result->speech_mask.size(), 1u);
  ASSERT_EQ(result->segments.size(), 1u);

  EXPECT_FLOAT_EQ(result->frame_rms[0], 0.0f);
  EXPECT_FLOAT_EQ(result->frame_db[0], 0.0f);
  EXPECT_EQ(result->speech_mask[0], 1);
  EXPECT_EQ(result->segments[0].frame_begin, 0);
  EXPECT_EQ(result->segments[0].frame_end, 1);
  EXPECT_EQ(result->segments[0].sample_begin, 0);
  EXPECT_EQ(result->segments[0].sample_end, 0);
}

TEST(VadTest, EmptyInputWithOddFrameLengthMatchesLibrosaError) {
  auto config = VadConfig{};
  config.frame_length = 5;
  config.hop_length = 2;

  const std::vector<float> audio;
  const auto result = RunVolumeVad(audio, 16000, config);

  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().find("too short"), std::string::npos);
}

TEST(VadTest, DetectsSpeechBetweenSilence) {
  std::vector<float> audio;
  AppendSilence(&audio, 16000, 1.0f);
  AppendTone(&audio, 16000, 1.0f, 0.15f, 220.0f);
  AppendSilence(&audio, 16000, 1.0f);

  const auto result = RunVolumeVad(audio, 16000, SyntheticConfig());
  ASSERT_TRUE(result.has_value()) << result.error();

  ASSERT_EQ(result->segments.size(), 1u);
  const VadSegment& segment = result->segments[0];
  const double start_sec =
      static_cast<double>(segment.sample_begin) / 16000.0;
  const double end_sec = static_cast<double>(segment.sample_end) / 16000.0;

  EXPECT_GT(start_sec, 0.95);
  EXPECT_LT(start_sec, 1.05);
  EXPECT_GT(end_sec, 1.95);
  EXPECT_LT(end_sec, 2.05);
  EXPECT_NEAR(*std::max_element(result->frame_db.begin(), result->frame_db.end()),
              0.0f, 1e-5f);
}

TEST(VadTest, UsesPeakRelativeThresholdToDropQuietRegion) {
  std::vector<float> audio;
  AppendSilence(&audio, 16000, 0.5f);
  AppendTone(&audio, 16000, 0.6f, 0.2f, 250.0f);
  AppendSilence(&audio, 16000, 0.3f);
  AppendTone(&audio, 16000, 0.6f, 0.001f, 250.0f);
  AppendSilence(&audio, 16000, 0.5f);

  auto config = SyntheticConfig();
  config.top_db = 40.0f;

  const auto result = RunVolumeVad(audio, 16000, config);
  ASSERT_TRUE(result.has_value()) << result.error();

  ASSERT_EQ(result->segments.size(), 1u);
  const double start_sec =
      static_cast<double>(result->segments[0].sample_begin) / 16000.0;
  const double end_sec =
      static_cast<double>(result->segments[0].sample_end) / 16000.0;
  EXPECT_GT(start_sec, 0.45);
  EXPECT_LT(start_sec, 0.55);
  EXPECT_GT(end_sec, 1.05);
  EXPECT_LT(end_sec, 1.15);
}

TEST(VadTest, OddFrameLengthMatchesLibrosaIntervals) {
  const std::vector<float> audio = {
      0.0f, 0.0f, 0.0012f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f};

  auto config = VadConfig{};
  config.frame_length = 5;
  config.hop_length = 4;
  config.top_db = 60.0f;

  const auto result = RunVolumeVad(audio, 16000, config);
  ASSERT_TRUE(result.has_value()) << result.error();

  const auto intervals = ExtractIntervals(*result);
  ASSERT_EQ(intervals.size(), 1u);
  EXPECT_EQ(intervals[0].begin, 0);
  EXPECT_EQ(intervals[0].end, 8);
}

TEST(VadTest, SeparatesSpeechAcrossLongSilence) {
  std::vector<float> audio;
  AppendSilence(&audio, 16000, 0.5f);
  AppendTone(&audio, 16000, 0.4f, 0.12f, 180.0f);
  AppendSilence(&audio, 16000, 0.6f);
  AppendTone(&audio, 16000, 0.4f, 0.12f, 350.0f);
  AppendSilence(&audio, 16000, 0.5f);

  const auto result = RunVolumeVad(audio, 16000, SyntheticConfig());
  ASSERT_TRUE(result.has_value()) << result.error();
  ASSERT_EQ(result->segments.size(), 2u);
}

TEST(VadTest, ProducesDeterministicResults) {
  std::vector<float> audio;
  AppendSilence(&audio, 16000, 0.3f);
  AppendTone(&audio, 16000, 0.5f, 0.08f, 180.0f);
  AppendSilence(&audio, 16000, 0.2f);
  AppendTone(&audio, 16000, 0.4f, 0.12f, 350.0f);

  const auto first = RunVolumeVad(audio, 16000, SyntheticConfig());
  const auto second = RunVolumeVad(audio, 16000, SyntheticConfig());

  ASSERT_TRUE(first.has_value()) << first.error();
  ASSERT_TRUE(second.has_value()) << second.error();

  ASSERT_EQ(first->frame_db.size(), second->frame_db.size());
  ASSERT_EQ(first->speech_mask.size(), second->speech_mask.size());
  ASSERT_EQ(first->segments.size(), second->segments.size());

  for (size_t i = 0; i < first->frame_db.size(); ++i) {
    EXPECT_FLOAT_EQ(first->frame_rms[i], second->frame_rms[i]);
    EXPECT_FLOAT_EQ(first->frame_db[i], second->frame_db[i]);
    EXPECT_FLOAT_EQ(first->noise_floor_db[i], second->noise_floor_db[i]);
    EXPECT_FLOAT_EQ(first->confidence[i], second->confidence[i]);
    EXPECT_EQ(first->speech_mask[i], second->speech_mask[i]);
  }

  for (size_t i = 0; i < first->segments.size(); ++i) {
    EXPECT_EQ(first->segments[i].frame_begin, second->segments[i].frame_begin);
    EXPECT_EQ(first->segments[i].frame_end, second->segments[i].frame_end);
    EXPECT_EQ(first->segments[i].sample_begin, second->segments[i].sample_begin);
    EXPECT_EQ(first->segments[i].sample_end, second->segments[i].sample_end);
    EXPECT_FLOAT_EQ(first->segments[i].mean_db, second->segments[i].mean_db);
    EXPECT_FLOAT_EQ(first->segments[i].peak_db, second->segments[i].peak_db);
    EXPECT_FLOAT_EQ(first->segments[i].priority_score,
                    second->segments[i].priority_score);
    EXPECT_EQ(first->segments[i].low_confidence,
              second->segments[i].low_confidence);
  }
}

TEST(VadTest, RejectsInvalidConfig) {
  std::vector<float> audio;
  AppendTone(&audio, 16000, 0.5f, 0.2f, 220.0f);

  auto config = VadConfig{};
  config.frame_length = 0;

  const auto result = RunVolumeVad(audio, 16000, config);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().find("frame_length"), std::string::npos);
}

TEST(VadTest, Recitation324MatchesLibrosaVolumeVadReference) {
  const auto wav = LoadWav(RecitationWavPath());
  ASSERT_TRUE(wav.has_value()) << wav.error();
  ASSERT_EQ(wav->metadata.channel_count, 1);

  const auto reference_intervals =
      LoadReferenceIntervals(RecitationReferencePath());
  ASSERT_TRUE(reference_intervals.has_value()) << reference_intervals.error();

  const auto result = RunVolumeVad(wav->channel_audio[0],
                                   wav->metadata.sample_rate_hz,
                                   VadConfig{});
  ASSERT_TRUE(result.has_value()) << result.error();

  const auto actual_intervals = ExtractIntervals(*result);
  ASSERT_EQ(actual_intervals.size(), reference_intervals->size());

  for (size_t i = 0; i < actual_intervals.size(); ++i) {
    EXPECT_LE(std::llabs(actual_intervals[i].begin -
                         reference_intervals->at(i).begin),
              static_cast<long long>(result->hop_length));
    EXPECT_LE(std::llabs(actual_intervals[i].end -
                         reference_intervals->at(i).end),
              static_cast<long long>(result->hop_length));
  }
}

}  // namespace
}  // namespace longwhisper
