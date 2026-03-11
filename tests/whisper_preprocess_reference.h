#pragma once

#include <complex>
#include <vector>

namespace longwhisper::test {

// Readable CPU reference frontend for Whisper. This lives under tests on purpose.
using Matrix = std::vector<std::vector<float>>;

constexpr int kWhisperSampleRate = 16000;
constexpr int kWhisperChunkSamples = 480000;
constexpr int kWhisperNfft = 400;
constexpr int kWhisperHopLength = 160;
constexpr int kWhisperFrequencyBins = 201;
constexpr int kWhisperMelBins = 128;
constexpr int kWhisperFrameCount = 3000;

struct WhisperPreprocessOptions {
  int frame_count = kWhisperFrameCount;
  bool pad_or_trim_to_chunk = true;
};

std::vector<float> MakeHannWindow400();
std::vector<float> ApplyHann400(const std::vector<float>& frame);
std::vector<std::complex<float>> Dft400(const std::vector<float>& windowed_frame);
std::vector<float> PowerSpectrum201(
    const std::vector<std::complex<float>>& dft_bins);
Matrix MakeMelFilterBank128();
std::vector<float> Mel128(const std::vector<float>& power_spectrum,
                          const Matrix& mel_filter_bank);
void Log10ClampInPlace(std::vector<float>* mel_energies, float peak_log_value);
void AffineNormalizeInPlace(std::vector<float>* log_mel_energies);

// Returns a log-mel spectrogram with shape [128][frame_count].
Matrix WhisperPreprocess(
    const std::vector<float>& audio,
    const WhisperPreprocessOptions& options = {});

}  // namespace longwhisper::test
