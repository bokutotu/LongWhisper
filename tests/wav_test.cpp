#include "longwhisper/wav.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
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

void WriteBytes(const fs::path& path, const std::vector<unsigned char>& bytes) {
  std::ofstream output(path, std::ios::binary);
  ASSERT_TRUE(output.is_open()) << "Failed to open " << path.string();
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  ASSERT_TRUE(output.good()) << "Failed to write " << path.string();
}

void AppendAscii(std::vector<unsigned char>* bytes, std::string_view text) {
  bytes->insert(bytes->end(), text.begin(), text.end());
}

void AppendUint16(std::vector<unsigned char>* bytes, uint16_t value) {
  bytes->push_back(static_cast<unsigned char>(value & 0xff));
  bytes->push_back(static_cast<unsigned char>((value >> 8) & 0xff));
}

void AppendUint32(std::vector<unsigned char>* bytes, uint32_t value) {
  bytes->push_back(static_cast<unsigned char>(value & 0xff));
  bytes->push_back(static_cast<unsigned char>((value >> 8) & 0xff));
  bytes->push_back(static_cast<unsigned char>((value >> 16) & 0xff));
  bytes->push_back(static_cast<unsigned char>((value >> 24) & 0xff));
}

void PatchUint32(std::vector<unsigned char>* bytes, size_t offset,
                 uint32_t value) {
  ASSERT_GE(bytes->size(), offset + 4);
  (*bytes)[offset + 0] = static_cast<unsigned char>(value & 0xff);
  (*bytes)[offset + 1] = static_cast<unsigned char>((value >> 8) & 0xff);
  (*bytes)[offset + 2] = static_cast<unsigned char>((value >> 16) & 0xff);
  (*bytes)[offset + 3] = static_cast<unsigned char>((value >> 24) & 0xff);
}

std::vector<unsigned char> EncodePcm16(const std::vector<int16_t>& samples) {
  std::vector<unsigned char> bytes;
  bytes.reserve(samples.size() * sizeof(int16_t));
  for (const int16_t sample : samples) {
    AppendUint16(&bytes, static_cast<uint16_t>(sample));
  }
  return bytes;
}

void AppendChunk(std::vector<unsigned char>* bytes, std::string_view id,
                 const std::vector<unsigned char>& payload) {
  ASSERT_EQ(id.size(), 4u);
  AppendAscii(bytes, id);
  AppendUint32(bytes, static_cast<uint32_t>(payload.size()));
  bytes->insert(bytes->end(), payload.begin(), payload.end());
  if ((payload.size() % 2) != 0) {
    bytes->push_back(0);
  }
}

std::vector<unsigned char> MakeWavBytes(
    uint16_t audio_format,
    uint16_t channel_count,
    uint32_t sample_rate_hz,
    uint16_t bits_per_sample,
    const std::vector<unsigned char>& data_payload,
    const std::vector<std::pair<std::string, std::vector<unsigned char>>>&
        extra_chunks = {},
    bool include_fmt = true,
    bool include_data = true) {
  std::vector<unsigned char> bytes;
  AppendAscii(&bytes, "RIFF");
  AppendUint32(&bytes, 0);
  AppendAscii(&bytes, "WAVE");

  for (const auto& [chunk_id, payload] : extra_chunks) {
    AppendChunk(&bytes, chunk_id, payload);
  }

  if (include_fmt) {
    std::vector<unsigned char> fmt_payload;
    const uint16_t bytes_per_sample =
        static_cast<uint16_t>(bits_per_sample / 8);
    const uint16_t block_align =
        static_cast<uint16_t>(channel_count * bytes_per_sample);
    const uint32_t byte_rate = sample_rate_hz * block_align;

    AppendUint16(&fmt_payload, audio_format);
    AppendUint16(&fmt_payload, channel_count);
    AppendUint32(&fmt_payload, sample_rate_hz);
    AppendUint32(&fmt_payload, byte_rate);
    AppendUint16(&fmt_payload, block_align);
    AppendUint16(&fmt_payload, bits_per_sample);
    AppendChunk(&bytes, "fmt ", fmt_payload);
  }

  if (include_data) {
    AppendChunk(&bytes, "data", data_payload);
  }

  PatchUint32(&bytes, 4, static_cast<uint32_t>(bytes.size() - 8));
  return bytes;
}

TEST(WavTest, LoadsBundledMonoFixtureWithoutResampling) {
  const fs::path path = SourceRoot() / "RECITATION324_158.wav";

  const LoadWavResult wav = LoadWav(path);
  ASSERT_TRUE(wav.has_value()) << wav.error();

  EXPECT_EQ(wav->metadata.sample_rate_hz, 48000);
  EXPECT_EQ(wav->metadata.channel_count, 1);
  EXPECT_EQ(wav->metadata.bits_per_sample, 16);
  EXPECT_EQ(wav->metadata.frame_count, 544937);
  EXPECT_DOUBLE_EQ(wav->metadata.duration_seconds, 544937.0 / 48000.0);

  ASSERT_EQ(wav->channel_audio.size(), 1u);
  ASSERT_EQ(wav->channel_audio[0].size(), 544937u);

  EXPECT_FLOAT_EQ(wav->channel_audio[0][0], -5.0f / 32768.0f);
  EXPECT_FLOAT_EQ(wav->channel_audio[0][1], -5.0f / 32768.0f);
  EXPECT_FLOAT_EQ(wav->channel_audio[0][2], -6.0f / 32768.0f);
  EXPECT_FLOAT_EQ(wav->channel_audio[0].back(), 11.0f / 32768.0f);
}

TEST(WavTest, DeinterleavesStereoPcm16IntoPerChannelFloatBuffers) {
  const fs::path dir = MakeTempDir("wav_stereo");
  const fs::path path = dir / "stereo.wav";
  const std::vector<int16_t> interleaved = {
      1000, -1000, 2000, -2000, -32768, 32767, 0, 1234};
  WriteBytes(path, MakeWavBytes(1, 2, 48000, 16, EncodePcm16(interleaved)));

  const LoadWavResult wav = LoadWav(path);
  ASSERT_TRUE(wav.has_value()) << wav.error();

  EXPECT_EQ(wav->metadata.sample_rate_hz, 48000);
  EXPECT_EQ(wav->metadata.channel_count, 2);
  EXPECT_EQ(wav->metadata.bits_per_sample, 16);
  EXPECT_EQ(wav->metadata.frame_count, 4);
  EXPECT_DOUBLE_EQ(wav->metadata.duration_seconds, 4.0 / 48000.0);

  ASSERT_EQ(wav->channel_audio.size(), 2u);
  EXPECT_EQ(wav->channel_audio[0].size(), 4u);
  EXPECT_EQ(wav->channel_audio[1].size(), 4u);

  EXPECT_FLOAT_EQ(wav->channel_audio[0][0], 1000.0f / 32768.0f);
  EXPECT_FLOAT_EQ(wav->channel_audio[0][1], 2000.0f / 32768.0f);
  EXPECT_FLOAT_EQ(wav->channel_audio[0][2], -1.0f);
  EXPECT_FLOAT_EQ(wav->channel_audio[0][3], 0.0f);

  EXPECT_FLOAT_EQ(wav->channel_audio[1][0], -1000.0f / 32768.0f);
  EXPECT_FLOAT_EQ(wav->channel_audio[1][1], -2000.0f / 32768.0f);
  EXPECT_FLOAT_EQ(wav->channel_audio[1][2], 32767.0f / 32768.0f);
  EXPECT_FLOAT_EQ(wav->channel_audio[1][3], 1234.0f / 32768.0f);
}

TEST(WavTest, SkipsUnknownChunksAndOddPadding) {
  const fs::path dir = MakeTempDir("wav_odd_chunk");
  const fs::path path = dir / "odd_padding.wav";

  const std::vector<int16_t> samples = {1, -1, 2};
  const std::vector<std::pair<std::string, std::vector<unsigned char>>>
      extra_chunks = {{"JUNK", {0xaa, 0xbb, 0xcc}}};
  WriteBytes(path,
             MakeWavBytes(1, 1, 16000, 16, EncodePcm16(samples), extra_chunks));

  const LoadWavResult wav = LoadWav(path);
  ASSERT_TRUE(wav.has_value()) << wav.error();

  ASSERT_EQ(wav->channel_audio.size(), 1u);
  ASSERT_EQ(wav->channel_audio[0].size(), 3u);
  EXPECT_FLOAT_EQ(wav->channel_audio[0][0], 1.0f / 32768.0f);
  EXPECT_FLOAT_EQ(wav->channel_audio[0][1], -1.0f / 32768.0f);
  EXPECT_FLOAT_EQ(wav->channel_audio[0][2], 2.0f / 32768.0f);
}

TEST(WavTest, RejectsUnsupportedFormats) {
  const fs::path dir = MakeTempDir("wav_unsupported");

  const fs::path float_path = dir / "float.wav";
  WriteBytes(float_path,
             MakeWavBytes(3, 1, 16000, 16, EncodePcm16({1, 2, 3, 4})));

  const LoadWavResult float_wav = LoadWav(float_path);
  ASSERT_FALSE(float_wav.has_value());
  EXPECT_NE(float_wav.error().find("PCM WAV"), std::string::npos);

  const fs::path pcm8_path = dir / "pcm8.wav";
  WriteBytes(pcm8_path, MakeWavBytes(1, 1, 16000, 8, {0x00, 0x80, 0xff, 0x7f}));

  const LoadWavResult pcm8_wav = LoadWav(pcm8_path);
  ASSERT_FALSE(pcm8_wav.has_value());
  EXPECT_NE(pcm8_wav.error().find("16-bit"), std::string::npos);
}

TEST(WavTest, RejectsMalformedFiles) {
  const fs::path dir = MakeTempDir("wav_malformed");

  struct TestCase {
    std::string name;
    std::vector<unsigned char> bytes;
    std::string expected_error;
  };

  std::vector<TestCase> cases;
  cases.push_back({"missing_fmt",
                   MakeWavBytes(1, 1, 16000, 16, EncodePcm16({1, 2, 3}),
                                {}, false, true),
                   "missing fmt"});
  cases.push_back({"missing_data",
                   MakeWavBytes(1, 1, 16000, 16, {}, {}, true, false),
                   "missing data"});
  cases.push_back({"misaligned_data",
                   MakeWavBytes(1, 1, 16000, 16, {0x01, 0x02, 0x03}),
                   "aligned"});

  std::vector<unsigned char> truncated = MakeWavBytes(
      1, 1, 16000, 16, EncodePcm16({1, 2, 3, 4}));
  ASSERT_GE(truncated.size(), 2u);
  truncated.pop_back();
  truncated.pop_back();
  PatchUint32(&truncated, 4, static_cast<uint32_t>(truncated.size() - 8));
  cases.push_back({"truncated_data", std::move(truncated), "RIFF size"});

  std::vector<unsigned char> riff_size_exceeds_file =
      MakeWavBytes(1, 1, 16000, 16, EncodePcm16({1, 2, 3, 4}));
  PatchUint32(&riff_size_exceeds_file, 4,
              static_cast<uint32_t>(riff_size_exceeds_file.size() + 16 - 8));
  cases.push_back({"riff_size_exceeds_file", std::move(riff_size_exceeds_file),
                   "RIFF chunk size"});

  for (const TestCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const fs::path path = dir / (test_case.name + ".wav");
    WriteBytes(path, test_case.bytes);

    const LoadWavResult wav = LoadWav(path);
    ASSERT_FALSE(wav.has_value());
    EXPECT_NE(wav.error().find(test_case.expected_error), std::string::npos)
        << wav.error();
  }
}

}  // namespace
}  // namespace longwhisper
