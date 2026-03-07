#include "longwhisper/wav.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace longwhisper {
namespace {

namespace fs = std::filesystem;

constexpr size_t kDecodeFramesPerChunk = 4096;

struct FormatMetadata {
  uint16_t audio_format = 0;
  uint16_t channel_count = 0;
  uint32_t sample_rate_hz = 0;
  uint32_t byte_rate = 0;
  uint16_t block_align = 0;
  uint16_t bits_per_sample = 0;
};

struct RiffHeader {
  uint32_t chunk_size = 0;
  uint64_t end_offset = 0;
};

struct ChunkHeader {
  std::array<char, 4> id{};
  uint32_t size = 0;
  std::streamoff payload_offset = 0;

  std::string IdString() const { return std::string(id.data(), id.size()); }
};

struct DataChunk {
  std::streamoff offset = 0;
  uint32_t size = 0;
};

struct WavLayout {
  FormatMetadata format_metadata;
  DataChunk data_chunk;
};

LoadWavResult Error(std::string message) {
  return std::unexpected(std::move(message));
}

std::string PathString(const fs::path& path) {
  return path.string();
}

std::expected<uint64_t, std::string> ReadFileSize(const fs::path& path) {
  std::error_code file_size_error;
  const uintmax_t raw_file_size = fs::file_size(path, file_size_error);
  if (file_size_error) {
    return std::unexpected("Failed to determine WAV file size: " +
                           PathString(path));
  }
  return static_cast<uint64_t>(raw_file_size);
}

bool ReadBytes(std::ifstream* input, void* data, size_t size) {
  return input->read(static_cast<char*>(data), static_cast<std::streamsize>(size))
      .good();
}

bool ReadUint16(std::ifstream* input, uint16_t* value) {
  std::array<unsigned char, 2> bytes{};
  if (!ReadBytes(input, bytes.data(), bytes.size())) {
    return false;
  }

  *value = static_cast<uint16_t>(bytes[0]) |
           (static_cast<uint16_t>(bytes[1]) << 8);
  return true;
}

bool ReadUint32(std::ifstream* input, uint32_t* value) {
  std::array<unsigned char, 4> bytes{};
  if (!ReadBytes(input, bytes.data(), bytes.size())) {
    return false;
  }

  *value = static_cast<uint32_t>(bytes[0]) |
           (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) |
           (static_cast<uint32_t>(bytes[3]) << 24);
  return true;
}

bool SkipBytes(std::ifstream* input, uint64_t count) {
  if (count == 0) {
    return true;
  }
  if (count >
      static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
    return false;
  }

  input->clear();
  input->seekg(static_cast<std::streamoff>(count), std::ios::cur);
  return input->good();
}

std::expected<RiffHeader, std::string> ReadRiffHeader(std::ifstream* input,
                                                      uint64_t file_size,
                                                      const fs::path& path) {
  std::array<char, 4> riff{};
  std::array<char, 4> wave{};
  RiffHeader header;
  if (!ReadBytes(input, riff.data(), riff.size()) ||
      !ReadUint32(input, &header.chunk_size) ||
      !ReadBytes(input, wave.data(), wave.size())) {
    return std::unexpected("Failed to read WAV header: " + PathString(path));
  }

  if (std::string(riff.data(), riff.size()) != "RIFF" ||
      std::string(wave.data(), wave.size()) != "WAVE") {
    return std::unexpected("Not a RIFF/WAVE file: " + PathString(path));
  }
  if (header.chunk_size < wave.size()) {
    return std::unexpected("RIFF chunk is too small: " + PathString(path));
  }

  header.end_offset = static_cast<uint64_t>(header.chunk_size) + 8;
  if (header.end_offset > file_size) {
    return std::unexpected("RIFF chunk size exceeds file size: " +
                           PathString(path));
  }

  return header;
}

std::expected<std::optional<ChunkHeader>, std::string> ReadNextChunkHeader(
    std::ifstream* input, uint64_t riff_end_offset, const fs::path& path) {
  const std::streamoff chunk_offset = input->tellg();
  if (chunk_offset < 0) {
    return std::unexpected("Failed to locate WAV chunk header: " +
                           PathString(path));
  }

  const uint64_t chunk_offset_u64 = static_cast<uint64_t>(chunk_offset);
  if (chunk_offset_u64 == riff_end_offset) {
    return std::optional<ChunkHeader>();
  }
  if (chunk_offset_u64 > riff_end_offset ||
      (riff_end_offset - chunk_offset_u64) < 8) {
    return std::unexpected("Truncated WAV chunk header: " + PathString(path));
  }

  ChunkHeader header;
  if (!ReadBytes(input, header.id.data(), header.id.size())) {
    return std::unexpected("Failed to read WAV chunk header: " +
                           PathString(path));
  }
  if (!ReadUint32(input, &header.size)) {
    return std::unexpected("Failed to read WAV chunk size: " +
                           PathString(path));
  }

  header.payload_offset = input->tellg();
  if (header.payload_offset < 0) {
    return std::unexpected("Failed to locate WAV chunk payload: " +
                           PathString(path));
  }

  const uint64_t padded_chunk_size =
      static_cast<uint64_t>(header.size) + (header.size % 2);
  if (padded_chunk_size >
      (riff_end_offset - static_cast<uint64_t>(header.payload_offset))) {
    return std::unexpected("WAV chunk exceeds RIFF size: " +
                           PathString(path));
  }

  return header;
}

std::expected<FormatMetadata, std::string> ParseFmtMetadata(
    std::ifstream* input, uint32_t chunk_size, const fs::path& path) {
  if (chunk_size < 16) {
    return std::unexpected("fmt chunk is too small: " + PathString(path));
  }

  FormatMetadata metadata;
  if (!ReadUint16(input, &metadata.audio_format) ||
      !ReadUint16(input, &metadata.channel_count) ||
      !ReadUint32(input, &metadata.sample_rate_hz) ||
      !ReadUint32(input, &metadata.byte_rate) ||
      !ReadUint16(input, &metadata.block_align) ||
      !ReadUint16(input, &metadata.bits_per_sample)) {
    return std::unexpected("Failed to read fmt chunk: " + PathString(path));
  }

  if (chunk_size > 16 &&
      !SkipBytes(input, static_cast<uint64_t>(chunk_size - 16))) {
    return std::unexpected("Truncated fmt chunk: " + PathString(path));
  }

  return metadata;
}

std::expected<void, std::string> ConsumeChunkPadding(std::ifstream* input,
                                                     uint32_t chunk_size,
                                                     const fs::path& path) {
  if ((chunk_size % 2) == 0) {
    return {};
  }
  if (!SkipBytes(input, 1)) {
    return std::unexpected("Truncated chunk padding: " + PathString(path));
  }
  return {};
}

std::expected<WavLayout, std::string> ParseWavLayout(std::ifstream* input,
                                                     uint64_t file_size,
                                                     const fs::path& path) {
  auto riff_header = ReadRiffHeader(input, file_size, path);
  if (!riff_header.has_value()) {
    return std::unexpected(std::move(riff_header.error()));
  }

  std::optional<FormatMetadata> format_metadata;
  std::optional<DataChunk> data_chunk;

  while (true) {
    auto chunk_header =
        ReadNextChunkHeader(input, riff_header->end_offset, path);
    if (!chunk_header.has_value()) {
      return std::unexpected(std::move(chunk_header.error()));
    }
    if (!chunk_header->has_value()) {
      break;
    }

    const ChunkHeader& header = chunk_header->value();
    const std::string chunk_id = header.IdString();
    if (chunk_id == "fmt ") {
      if (format_metadata.has_value()) {
        return std::unexpected("Duplicate fmt chunk in WAV file: " +
                               PathString(path));
      }

      auto parsed_metadata = ParseFmtMetadata(input, header.size, path);
      if (!parsed_metadata.has_value()) {
        return std::unexpected(std::move(parsed_metadata.error()));
      }
      format_metadata = *parsed_metadata;
    } else if (chunk_id == "data") {
      if (data_chunk.has_value()) {
        return std::unexpected("Duplicate data chunk in WAV file: " +
                               PathString(path));
      }

      DataChunk recorded_data_chunk;
      recorded_data_chunk.offset = header.payload_offset;
      recorded_data_chunk.size = header.size;
      data_chunk = recorded_data_chunk;
      if (!SkipBytes(input, header.size)) {
        return std::unexpected("Failed to skip WAV data chunk: " +
                               PathString(path));
      }
    } else {
      if (!SkipBytes(input, header.size)) {
        return std::unexpected("Truncated WAV chunk body: " + PathString(path));
      }
    }

    auto padding_status = ConsumeChunkPadding(input, header.size, path);
    if (!padding_status.has_value()) {
      return std::unexpected(std::move(padding_status.error()));
    }
  }

  if (!format_metadata.has_value()) {
    return std::unexpected("WAV file is missing fmt chunk: " +
                           PathString(path));
  }
  if (!data_chunk.has_value()) {
    return std::unexpected("WAV file is missing data chunk: " +
                           PathString(path));
  }

  WavLayout layout;
  layout.format_metadata = *format_metadata;
  layout.data_chunk = *data_chunk;
  return layout;
}

std::expected<void, std::string> ValidateFormatMetadata(
    const FormatMetadata& format_metadata, const fs::path& path) {
  if (format_metadata.audio_format != 1) {
    return std::unexpected("Only PCM WAV files are supported: " +
                           PathString(path));
  }
  if (format_metadata.channel_count == 0) {
    return std::unexpected("WAV channel_count must be > 0: " +
                           PathString(path));
  }
  if (format_metadata.sample_rate_hz == 0) {
    return std::unexpected("WAV sample_rate_hz must be > 0: " +
                           PathString(path));
  }
  if (format_metadata.bits_per_sample != 16) {
    return std::unexpected("Only 16-bit PCM WAV files are supported: " +
                           PathString(path));
  }

  const uint16_t expected_block_align =
      static_cast<uint16_t>(format_metadata.channel_count * sizeof(int16_t));
  if (format_metadata.block_align != expected_block_align) {
    return std::unexpected("Invalid WAV block_align: " + PathString(path));
  }

  const uint32_t expected_byte_rate =
      format_metadata.sample_rate_hz *
      static_cast<uint32_t>(format_metadata.block_align);
  if (format_metadata.byte_rate != expected_byte_rate) {
    return std::unexpected("Invalid WAV byte_rate: " + PathString(path));
  }

  return {};
}

std::expected<void, std::string> DecodePcm16(std::ifstream* input,
                                             std::streamoff data_offset,
                                             uint32_t data_size,
                                             const FormatMetadata& format_metadata,
                                             const fs::path& path,
                                             WavFile* wav) {
  const uint64_t block_align = format_metadata.block_align;
  if (data_size == 0) {
    return std::unexpected("WAV data chunk is empty: " + PathString(path));
  }
  if ((data_size % block_align) != 0) {
    return std::unexpected("WAV data chunk size is not aligned to frames: " +
                           PathString(path));
  }

  const uint64_t frame_count = data_size / block_align;
  if (frame_count >
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return std::unexpected("WAV frame_count exceeds supported range: " +
                           PathString(path));
  }

  wav->metadata.sample_rate_hz =
      static_cast<int>(format_metadata.sample_rate_hz);
  wav->metadata.channel_count =
      static_cast<int>(format_metadata.channel_count);
  wav->metadata.bits_per_sample =
      static_cast<int>(format_metadata.bits_per_sample);
  wav->metadata.frame_count = static_cast<int64_t>(frame_count);
  wav->metadata.duration_seconds =
      static_cast<double>(frame_count) / format_metadata.sample_rate_hz;

  wav->channel_audio.resize(format_metadata.channel_count);
  for (std::vector<float>& channel : wav->channel_audio) {
    channel.resize(static_cast<size_t>(frame_count), 0.0f);
  }

  input->clear();
  input->seekg(data_offset);
  if (!input->good()) {
    return std::unexpected("Failed to seek to WAV data chunk: " +
                           PathString(path));
  }

  const size_t decode_chunk_bytes = static_cast<size_t>(
      std::min<uint64_t>(static_cast<uint64_t>(data_size),
                         static_cast<uint64_t>(kDecodeFramesPerChunk) *
                             block_align));
  std::vector<unsigned char> buffer(decode_chunk_bytes, 0);

  uint64_t frame_index = 0;
  while (frame_index < frame_count) {
    const uint64_t frames_this_chunk =
        std::min<uint64_t>(frame_count - frame_index, kDecodeFramesPerChunk);
    const size_t bytes_this_chunk =
        static_cast<size_t>(frames_this_chunk * block_align);

    if (!ReadBytes(input, buffer.data(), bytes_this_chunk)) {
      return std::unexpected("Truncated WAV data chunk: " + PathString(path));
    }

    for (uint64_t frame = 0; frame < frames_this_chunk; ++frame) {
      const size_t frame_offset =
          static_cast<size_t>(frame * block_align);
      for (uint16_t channel = 0; channel < format_metadata.channel_count;
           ++channel) {
        const size_t sample_offset =
            frame_offset + static_cast<size_t>(channel) * sizeof(int16_t);
        const uint16_t raw =
            static_cast<uint16_t>(buffer[sample_offset]) |
            (static_cast<uint16_t>(buffer[sample_offset + 1]) << 8);
        const int16_t sample = static_cast<int16_t>(raw);

        wav->channel_audio[channel][static_cast<size_t>(frame_index + frame)] =
            static_cast<float>(sample) / 32768.0f;
      }
    }

    frame_index += frames_this_chunk;
  }

  return {};
}

}  // namespace

LoadWavResult LoadWav(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    return Error("Failed to open WAV file: " + PathString(path));
  }

  auto file_size = ReadFileSize(path);
  if (!file_size.has_value()) {
    return Error(std::move(file_size.error()));
  }

  auto wav_layout = ParseWavLayout(&input, *file_size, path);
  if (!wav_layout.has_value()) {
    return Error(std::move(wav_layout.error()));
  }

  auto format_status = ValidateFormatMetadata(wav_layout->format_metadata, path);
  if (!format_status.has_value()) {
    return Error(std::move(format_status.error()));
  }

  WavFile wav;
  auto decode_status = DecodePcm16(&input, wav_layout->data_chunk.offset,
                                   wav_layout->data_chunk.size,
                                   wav_layout->format_metadata, path, &wav);
  if (!decode_status.has_value()) {
    return Error(std::move(decode_status.error()));
  }

  return wav;
}

}  // namespace longwhisper
