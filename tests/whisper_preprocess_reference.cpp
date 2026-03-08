#include "whisper_preprocess_reference.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace longwhisper::test {
namespace {

constexpr int kWhisperCenterPad = kWhisperNfft / 2;
constexpr float kMinimumMelEnergy = 1e-10f;
constexpr float kDynamicRangeInLog10 = 8.0f;
constexpr float kAffineBias = 4.0f;
constexpr float kAffineDivisor = 4.0f;

std::vector<float> PadOrTrimToWhisperChunk(const std::vector<float>& audio) {
  std::vector<float> padded(kWhisperChunkSamples, 0.0f);
  const size_t copy_count =
      std::min(audio.size(), static_cast<size_t>(kWhisperChunkSamples));
  std::copy_n(audio.begin(), copy_count, padded.begin());
  return padded;
}

size_t ReflectIndex(int index, size_t size) {
  if (size <= 1) {
    return 0;
  }

  const int period = static_cast<int>(2 * size - 2);
  int reflected = index % period;
  if (reflected < 0) {
    reflected += period;
  }
  if (reflected >= static_cast<int>(size)) {
    reflected = period - reflected;
  }
  return static_cast<size_t>(reflected);
}

std::vector<float> ReflectPad(const std::vector<float>& audio, int pad) {
  // This matches centered STFT framing with reflect padding on both sides.
  std::vector<float> padded(audio.size() + static_cast<size_t>(pad * 2), 0.0f);
  for (size_t i = 0; i < padded.size(); ++i) {
    const int source_index = static_cast<int>(i) - pad;
    padded[i] = audio[ReflectIndex(source_index, audio.size())];
  }
  return padded;
}

std::vector<float> ExtractFrame(const std::vector<float>& centered_audio,
                                int frame_index) {
  const size_t begin = static_cast<size_t>(frame_index * kWhisperHopLength);
  return std::vector<float>(centered_audio.begin() + begin,
                            centered_audio.begin() + begin + kWhisperNfft);
}

double HzToMel(double hz) {
  constexpr double kMinLogHz = 1000.0;
  constexpr double kMinLogMel = 15.0;
  constexpr double kLinearHzPerMel = 200.0 / 3.0;
  constexpr double kLogStep = std::log(6.4) / 27.0;

  if (hz < kMinLogHz) {
    return hz / kLinearHzPerMel;
  }
  return kMinLogMel + std::log(hz / kMinLogHz) / kLogStep;
}

double MelToHz(double mel) {
  constexpr double kMinLogHz = 1000.0;
  constexpr double kMinLogMel = 15.0;
  constexpr double kLinearHzPerMel = 200.0 / 3.0;
  constexpr double kLogStep = std::log(6.4) / 27.0;

  if (mel < kMinLogMel) {
    return mel * kLinearHzPerMel;
  }
  return kMinLogHz * std::exp((mel - kMinLogMel) * kLogStep);
}

std::vector<double> Linspace(double begin, double end, int count) {
  std::vector<double> values(count, begin);
  if (count == 1) {
    return values;
  }

  const double step = (end - begin) / static_cast<double>(count - 1);
  for (int i = 0; i < count; ++i) {
    values[static_cast<size_t>(i)] = begin + step * static_cast<double>(i);
  }
  return values;
}

}  // namespace

std::vector<float> MakeHannWindow400() {
  std::vector<float> window(kWhisperNfft, 0.0f);
  for (int i = 0; i < kWhisperNfft; ++i) {
    const double phase =
        2.0 * std::numbers::pi_v<double> * static_cast<double>(i) / kWhisperNfft;
    window[static_cast<size_t>(i)] =
        static_cast<float>(0.5 - 0.5 * std::cos(phase));
  }
  return window;
}

std::vector<float> ApplyHann400(const std::vector<float>& frame) {
  if (frame.size() != static_cast<size_t>(kWhisperNfft)) {
    return {};
  }

  const std::vector<float> window = MakeHannWindow400();
  std::vector<float> windowed(frame.size(), 0.0f);
  for (size_t i = 0; i < frame.size(); ++i) {
    windowed[i] = frame[i] * window[i];
  }
  return windowed;
}

std::vector<std::complex<float>> Dft400(const std::vector<float>& windowed_frame) {
  if (windowed_frame.size() != static_cast<size_t>(kWhisperNfft)) {
    return {};
  }

  std::vector<std::complex<float>> dft_bins(kWhisperNfft, {0.0f, 0.0f});
  for (int k = 0; k < kWhisperNfft; ++k) {
    double real = 0.0;
    double imag = 0.0;
    for (int n = 0; n < kWhisperNfft; ++n) {
      const double angle =
          -2.0 * std::numbers::pi_v<double> * static_cast<double>(k * n) /
          kWhisperNfft;
      real += static_cast<double>(windowed_frame[static_cast<size_t>(n)]) *
              std::cos(angle);
      imag += static_cast<double>(windowed_frame[static_cast<size_t>(n)]) *
              std::sin(angle);
    }
    dft_bins[static_cast<size_t>(k)] =
        std::complex<float>(static_cast<float>(real), static_cast<float>(imag));
  }
  return dft_bins;
}

std::vector<float> PowerSpectrum201(
    const std::vector<std::complex<float>>& dft_bins) {
  if (dft_bins.size() != static_cast<size_t>(kWhisperNfft)) {
    return {};
  }

  std::vector<float> power_spectrum(kWhisperFrequencyBins, 0.0f);
  for (int i = 0; i < kWhisperFrequencyBins; ++i) {
    power_spectrum[static_cast<size_t>(i)] =
        std::norm(dft_bins[static_cast<size_t>(i)]);
  }
  return power_spectrum;
}

Matrix MakeMelFilterBank128() {
  Matrix filter_bank(kWhisperMelBins,
                     std::vector<float>(kWhisperFrequencyBins, 0.0f));

  const std::vector<double> fft_bin_hz =
      Linspace(0.0, kWhisperSampleRate / 2.0, kWhisperFrequencyBins);
  const std::vector<double> mel_points =
      Linspace(HzToMel(0.0), HzToMel(kWhisperSampleRate / 2.0),
               kWhisperMelBins + 2);

  std::vector<double> hz_points(mel_points.size(), 0.0);
  for (size_t i = 0; i < mel_points.size(); ++i) {
    hz_points[i] = MelToHz(mel_points[i]);
  }

  for (int mel_index = 0; mel_index < kWhisperMelBins; ++mel_index) {
    const double left = hz_points[static_cast<size_t>(mel_index)];
    const double center = hz_points[static_cast<size_t>(mel_index + 1)];
    const double right = hz_points[static_cast<size_t>(mel_index + 2)];
    const double area_scale = 2.0 / (right - left);

    for (int bin_index = 0; bin_index < kWhisperFrequencyBins; ++bin_index) {
      const double hz = fft_bin_hz[static_cast<size_t>(bin_index)];
      const double up_slope = (hz - left) / (center - left);
      const double down_slope = (right - hz) / (right - center);
      const double triangle = std::max(0.0, std::min(up_slope, down_slope));
      filter_bank[static_cast<size_t>(mel_index)][static_cast<size_t>(bin_index)] =
          static_cast<float>(triangle * area_scale);
    }
  }

  return filter_bank;
}

std::vector<float> Mel128(const std::vector<float>& power_spectrum,
                          const Matrix& mel_filter_bank) {
  if (power_spectrum.size() != static_cast<size_t>(kWhisperFrequencyBins) ||
      mel_filter_bank.size() != static_cast<size_t>(kWhisperMelBins)) {
    return {};
  }

  std::vector<float> mel_energies(kWhisperMelBins, 0.0f);
  for (int mel_index = 0; mel_index < kWhisperMelBins; ++mel_index) {
    double sum = 0.0;
    for (int bin_index = 0; bin_index < kWhisperFrequencyBins; ++bin_index) {
      sum += static_cast<double>(
                 mel_filter_bank[static_cast<size_t>(mel_index)]
                                [static_cast<size_t>(bin_index)]) *
             static_cast<double>(power_spectrum[static_cast<size_t>(bin_index)]);
    }
    mel_energies[static_cast<size_t>(mel_index)] = static_cast<float>(sum);
  }
  return mel_energies;
}

void Log10ClampInPlace(std::vector<float>* mel_energies, float peak_log_value) {
  if (mel_energies == nullptr) {
    return;
  }

  for (float& value : *mel_energies) {
    value = std::log10(std::max(value, kMinimumMelEnergy));
    value = std::max(value, peak_log_value - kDynamicRangeInLog10);
  }
}

void AffineNormalizeInPlace(std::vector<float>* log_mel_energies) {
  if (log_mel_energies == nullptr) {
    return;
  }

  for (float& value : *log_mel_energies) {
    value = (value + kAffineBias) / kAffineDivisor;
  }
}

Matrix WhisperPreprocess(const std::vector<float>& audio) {
  const std::vector<float> chunk_audio = PadOrTrimToWhisperChunk(audio);
  const std::vector<float> centered_audio =
      ReflectPad(chunk_audio, kWhisperCenterPad);
  const Matrix mel_filter_bank = MakeMelFilterBank128();

  Matrix mel_spectrogram(kWhisperMelBins,
                         std::vector<float>(kWhisperFrameCount, 0.0f));
  float peak_log_value = std::log10(kMinimumMelEnergy);

  // Centered 30-second audio produces 3001 STFT frames; Whisper drops the last one.
  for (int frame_index = 0; frame_index < kWhisperFrameCount; ++frame_index) {
    const std::vector<float> frame = ExtractFrame(centered_audio, frame_index);
    const std::vector<float> windowed = ApplyHann400(frame);
    const std::vector<std::complex<float>> dft_bins = Dft400(windowed);
    const std::vector<float> power_spectrum = PowerSpectrum201(dft_bins);
    const std::vector<float> mel_energies = Mel128(power_spectrum, mel_filter_bank);

    for (int mel_index = 0; mel_index < kWhisperMelBins; ++mel_index) {
      const float mel_energy = mel_energies[static_cast<size_t>(mel_index)];
      mel_spectrogram[static_cast<size_t>(mel_index)]
                     [static_cast<size_t>(frame_index)] = mel_energy;
      peak_log_value =
          std::max(peak_log_value, std::log10(std::max(mel_energy, kMinimumMelEnergy)));
    }
  }

  for (std::vector<float>& mel_band : mel_spectrogram) {
    Log10ClampInPlace(&mel_band, peak_log_value);
    AffineNormalizeInPlace(&mel_band);
  }

  return mel_spectrogram;
}

}  // namespace longwhisper::test
