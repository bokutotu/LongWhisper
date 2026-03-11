#include "preprocess/frontend.cuh"
#include "preprocess/preprocess_cuda.h"
#include "whisper_preprocess_reference.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace longwhisper {
namespace {

constexpr double kMaxStftAbsError = 5e-3;
constexpr double kMaxPowerAbsError = 5e-2;
constexpr double kMaxPowerRelError = 1e-4;
constexpr double kMaxMelAbsError = 2e-2;
constexpr double kMaxMelRelError = 1e-4;
constexpr double kMaxPeakAbsError = 2e-2;
constexpr double kMaxPeakRelError = 1e-4;
constexpr double kMaxNormalizedAbsError = 5e-3;
constexpr double kMaxNormalizedRelError = 1e-4;
constexpr int kHopLength = detail::kPreprocess400HopLength;

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      (void)cudaFree(ptr_);
    }
  }

  cudaError_t Allocate(size_t count) {
    return cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T));
  }

  T* get() const { return ptr_; }

 private:
  T* ptr_ = nullptr;
};

std::string GetCudaAvailabilityFailure() {
  int device_count = 0;
  const cudaError_t device_count_error = cudaGetDeviceCount(&device_count);
  if (device_count_error != cudaSuccess) {
    return std::string("CUDA device unavailable: ") +
           cudaGetErrorString(device_count_error);
  }
  if (device_count == 0) {
    return "No CUDA device available";
  }
  return {};
}

struct Frontend400StageWorkspace {
  float* device_mel_scratch = nullptr;
  float* device_frame_peak_output = nullptr;
};

__global__ void InspectFrontend400Kernel(const float* __restrict__ input,
                                         int sample_count,
                                         float2* __restrict__ stft_output,
                                         float* __restrict__ power_output,
                                         float* __restrict__ mel_output) {
  __shared__ Frontend400SharedStorage shared;

  const int tid = static_cast<int>(threadIdx.x);
  const int frame_index = static_cast<int>(blockIdx.x);

  Frontend400(input, sample_count, &shared);

  if (tid < detail::kFft400OutputBins) {
    const size_t base =
        static_cast<size_t>(frame_index) * detail::kFft400OutputBins;
    if (stft_output != nullptr) {
      stft_output[base + static_cast<size_t>(tid)] = shared.fft[tid];
    }
    if (power_output != nullptr) {
      power_output[base + static_cast<size_t>(tid)] = shared.power[tid];
    }
  }

  if (tid < detail::kPreprocess400MelBins && mel_output != nullptr) {
    mel_output[static_cast<size_t>(frame_index) * detail::kPreprocess400MelBins +
               static_cast<size_t>(tid)] = shared.mel[tid];
  }
}

cudaError_t RunInspectFrontend400(const float* device_input, int sample_count,
                                  int frame_count, float2* device_stft_output,
                                  float* device_power_output,
                                  float* device_mel_output,
                                  cudaStream_t stream = nullptr) {
  if (sample_count < 0 || frame_count < 0) {
    return cudaErrorInvalidValue;
  }
  if (frame_count == 0) {
    return cudaSuccess;
  }
  if (device_input == nullptr || sample_count == 0) {
    return cudaErrorInvalidValue;
  }
  if (device_stft_output == nullptr && device_power_output == nullptr &&
      device_mel_output == nullptr) {
    return cudaErrorInvalidValue;
  }

  InspectFrontend400Kernel<<<frame_count, kFrontend400Threads, 0, stream>>>(
      device_input, sample_count, device_stft_output, device_power_output,
      device_mel_output);
  return cudaGetLastError();
}

__global__ void Frontend400StageKernel(const float* __restrict__ input,
                                       int sample_count,
                                       float* __restrict__ mel_output,
                                       float* __restrict__ frame_peak_output) {
  __shared__ Frontend400SharedStorage shared;
  __shared__ float warp_max[kFrontend400Threads / detail::kCudaWarpSize];

  const int tid = static_cast<int>(threadIdx.x);
  const int frame_index = static_cast<int>(blockIdx.x);

  Frontend400(input, sample_count, &shared);

  float thread_max = detail::NegativeInfinity();
  if (tid < detail::kPreprocess400MelBins) {
    const float mel_value = shared.mel[tid];
    mel_output[static_cast<size_t>(frame_index) * detail::kPreprocess400MelBins +
               static_cast<size_t>(tid)] = mel_value;
    thread_max = mel_value;
  }

  const float frame_max =
      detail::BlockReduceMax<kFrontend400Threads>(thread_max, warp_max);
  if (tid == 0) {
    frame_peak_output[frame_index] = frame_max;
  }
}

cudaError_t RunFrontend400StageForTest(const float* device_input,
                                       int sample_count, int frame_count,
                                       const Frontend400StageWorkspace& workspace,
                                       cudaStream_t stream = nullptr) {
  if (sample_count < 0 || frame_count < 0) {
    return cudaErrorInvalidValue;
  }
  if (frame_count == 0) {
    return cudaSuccess;
  }
  if (device_input == nullptr || sample_count == 0 ||
      workspace.device_mel_scratch == nullptr ||
      workspace.device_frame_peak_output == nullptr) {
    return cudaErrorInvalidValue;
  }

  Frontend400StageKernel<<<frame_count, kFrontend400Threads, 0, stream>>>(
      device_input, sample_count, workspace.device_mel_scratch,
      workspace.device_frame_peak_output);
  return cudaGetLastError();
}

__global__ void ReduceMaxTestKernel(const float* __restrict__ input,
                                    int value_count,
                                    float* __restrict__ max_output) {
  __shared__ float warp_max[kFrontend400Threads / detail::kCudaWarpSize];

  float thread_max = detail::NegativeInfinity();
  for (int index = static_cast<int>(threadIdx.x); index < value_count;
       index += static_cast<int>(blockDim.x)) {
    thread_max = fmaxf(thread_max, input[index]);
  }

  const float block_max =
      detail::BlockReduceMax<kFrontend400Threads>(thread_max, warp_max);
  if (threadIdx.x == 0) {
    max_output[0] = block_max;
  }
}

cudaError_t RunReduceMaxForTest(const float* device_input, int value_count,
                                float* device_max_output,
                                cudaStream_t stream = nullptr) {
  if (value_count <= 0) {
    return cudaErrorInvalidValue;
  }
  if (device_input == nullptr || device_max_output == nullptr) {
    return cudaErrorInvalidValue;
  }

  ReduceMaxTestKernel<<<1, kFrontend400Threads, 0, stream>>>(
      device_input, value_count, device_max_output);
  return cudaGetLastError();
}

int ReflectIndex(int index, int sample_count) {
  if (sample_count <= 1) {
    return 0;
  }

  const int period = 2 * sample_count - 2;
  int reflected = index % period;
  if (reflected < 0) {
    reflected += period;
  }
  if (reflected >= sample_count) {
    reflected = period - reflected;
  }
  return reflected;
}

std::vector<float> ExtractCenteredFrame400(const std::vector<float>& input,
                                           int frame_index) {
  std::vector<float> frame(detail::kFft400InputSize, 0.0f);
  const int frame_begin =
      frame_index * kHopLength - detail::kPreprocess400CenterPad;

  for (int i = 0; i < detail::kFft400InputSize; ++i) {
    const int source_index =
        ReflectIndex(frame_begin + i, static_cast<int>(input.size()));
    frame[static_cast<size_t>(i)] = input[static_cast<size_t>(source_index)];
  }

  return frame;
}

std::vector<std::complex<float>> CpuReferenceStft400(
    const std::vector<float>& input, int frame_index) {
  const std::vector<float> frame = ExtractCenteredFrame400(input, frame_index);
  const std::vector<float> windowed_frame = test::ApplyHann400(frame);
  const std::vector<std::complex<float>> dft_bins = test::Dft400(windowed_frame);
  if (dft_bins.size() < static_cast<size_t>(detail::kFft400OutputBins)) {
    return {};
  }

  std::vector<std::complex<float>> output(detail::kFft400OutputBins,
                                          {0.0f, 0.0f});
  for (int k = 0; k < detail::kFft400OutputBins; ++k) {
    output[static_cast<size_t>(k)] = dft_bins[static_cast<size_t>(k)];
  }

  return output;
}

std::vector<float> CpuReferencePower400(const std::vector<float>& input,
                                        int frame_index) {
  const auto stft_bins = CpuReferenceStft400(input, frame_index);
  if (stft_bins.size() != static_cast<size_t>(detail::kFft400OutputBins)) {
    return {};
  }

  std::vector<float> output(detail::kFft400OutputBins, 0.0f);
  for (int k = 0; k < detail::kFft400OutputBins; ++k) {
    output[static_cast<size_t>(k)] = std::norm(stft_bins[static_cast<size_t>(k)]);
  }
  return output;
}

std::vector<float> CpuReferenceMel400(const std::vector<float>& input,
                                      int frame_index,
                                      const test::Matrix& mel_filter_bank) {
  const auto power_spectrum = CpuReferencePower400(input, frame_index);
  if (power_spectrum.size() != static_cast<size_t>(detail::kFft400OutputBins)) {
    return {};
  }
  return test::Mel128(power_spectrum, mel_filter_bank);
}

float CpuReferenceMelPeak400(const std::vector<float>& input, int frame_count,
                             const test::Matrix& mel_filter_bank) {
  float peak = 0.0f;
  for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
    const auto mel_energies =
        CpuReferenceMel400(input, frame_index, mel_filter_bank);
    if (mel_energies.size() !=
        static_cast<size_t>(detail::kPreprocess400MelBins)) {
      return -1.0f;
    }

    for (const float value : mel_energies) {
      peak = std::max(peak, value);
    }
  }
  return peak;
}

std::vector<float> CpuReferenceNormalizedMel400(
    const std::vector<float>& input, int frame_count) {
  test::WhisperPreprocessOptions options;
  options.frame_count = frame_count;
  options.pad_or_trim_to_chunk = false;

  const test::Matrix expected = test::WhisperPreprocess(input, options);
  if (expected.size() != static_cast<size_t>(detail::kPreprocess400MelBins)) {
    return {};
  }

  std::vector<float> output(static_cast<size_t>(frame_count) *
                                detail::kPreprocess400MelBins,
                            0.0f);
  for (const auto& mel_band : expected) {
    if (mel_band.size() != static_cast<size_t>(frame_count)) {
      return {};
    }
  }

  for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
    for (int mel_index = 0; mel_index < detail::kPreprocess400MelBins;
         ++mel_index) {
      output[static_cast<size_t>(frame_index) * detail::kPreprocess400MelBins +
             static_cast<size_t>(mel_index)] =
          expected[static_cast<size_t>(mel_index)]
                  [static_cast<size_t>(frame_index)];
    }
  }

  return output;
}

::testing::AssertionResult MatchesNaiveCpuStftReference(
    const std::vector<float>& input, const std::vector<float2>& gpu_stft_output,
    int frame_count) {
  for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
    const auto expected = CpuReferenceStft400(input, frame_index);
    if (expected.size() != static_cast<size_t>(detail::kFft400OutputBins)) {
      return ::testing::AssertionFailure()
             << "Reference STFT returned " << expected.size()
             << " bins, expected " << detail::kFft400OutputBins;
    }

    for (int bin = 0; bin < detail::kFft400OutputBins; ++bin) {
      const float2 actual =
          gpu_stft_output[static_cast<size_t>(frame_index) *
                              detail::kFft400OutputBins +
                          static_cast<size_t>(bin)];
      const double real_error =
          std::abs(static_cast<double>(actual.x) - expected[bin].real());
      const double imag_error =
          std::abs(static_cast<double>(actual.y) - expected[bin].imag());

      if (real_error > kMaxStftAbsError || imag_error > kMaxStftAbsError) {
        return ::testing::AssertionFailure()
               << "Mismatch at frame " << frame_index << ", bin " << bin
               << ": expected=(" << expected[bin].real() << ", "
               << expected[bin].imag() << "), actual=(" << actual.x << ", "
               << actual.y << "), errors=(" << real_error << ", " << imag_error
               << ")";
      }
    }
  }

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult MatchesNaiveCpuPowerReference(
    const std::vector<float>& input, const std::vector<float>& gpu_power_output,
    int frame_count) {
  for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
    const auto expected = CpuReferencePower400(input, frame_index);
    if (expected.size() != static_cast<size_t>(detail::kFft400OutputBins)) {
      return ::testing::AssertionFailure()
             << "Reference power spectrum returned " << expected.size()
             << " bins, expected " << detail::kFft400OutputBins;
    }

    for (int bin = 0; bin < detail::kFft400OutputBins; ++bin) {
      const double actual = static_cast<double>(
          gpu_power_output[static_cast<size_t>(frame_index) *
                               detail::kFft400OutputBins +
                           static_cast<size_t>(bin)]);
      const double expected_value = static_cast<double>(expected[bin]);
      const double error = std::abs(actual - expected_value);
      const double tolerance =
          kMaxPowerAbsError + kMaxPowerRelError * std::abs(expected_value);

      if (error > tolerance) {
        return ::testing::AssertionFailure()
               << "Power mismatch at frame " << frame_index << ", bin " << bin
               << ": expected=" << expected_value << ", actual=" << actual
               << ", error=" << error << ", tolerance=" << tolerance;
      }
    }
  }

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult MatchesNaiveCpuMelReference(
    const std::vector<float>& input, const test::Matrix& mel_filter_bank,
    const std::vector<float>& gpu_mel_output, int frame_count) {
  for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
    const auto expected = CpuReferenceMel400(input, frame_index, mel_filter_bank);
    if (expected.size() != static_cast<size_t>(detail::kPreprocess400MelBins)) {
      return ::testing::AssertionFailure()
             << "Reference mel spectrum returned " << expected.size()
             << " bins, expected " << detail::kPreprocess400MelBins;
    }

    for (int mel_index = 0; mel_index < detail::kPreprocess400MelBins;
         ++mel_index) {
      const double actual = static_cast<double>(
          gpu_mel_output[static_cast<size_t>(frame_index) *
                             detail::kPreprocess400MelBins +
                         static_cast<size_t>(mel_index)]);
      const double expected_value = static_cast<double>(expected[mel_index]);
      const double error = std::abs(actual - expected_value);
      const double tolerance =
          kMaxMelAbsError + kMaxMelRelError * std::abs(expected_value);

      if (error > tolerance) {
        return ::testing::AssertionFailure()
               << "Mel mismatch at frame " << frame_index << ", bin "
               << mel_index << ": expected=" << expected_value
               << ", actual=" << actual << ", error=" << error
               << ", tolerance=" << tolerance;
      }
    }
  }

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult MatchesNaiveCpuMelPeakReference(
    const std::vector<float>& input, const test::Matrix& mel_filter_bank,
    float gpu_peak_output, int frame_count) {
  const float expected =
      CpuReferenceMelPeak400(input, frame_count, mel_filter_bank);
  if (expected < 0.0f) {
    return ::testing::AssertionFailure()
           << "Reference mel peak computation failed for frame count "
           << frame_count;
  }

  const double actual = static_cast<double>(gpu_peak_output);
  const double expected_value = static_cast<double>(expected);
  const double error = std::abs(actual - expected_value);
  const double tolerance =
      kMaxPeakAbsError + kMaxPeakRelError * std::abs(expected_value);

  if (error > tolerance) {
    return ::testing::AssertionFailure()
           << "Mel peak mismatch: expected=" << expected_value
           << ", actual=" << actual << ", error=" << error
           << ", tolerance=" << tolerance;
  }

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult MatchesNaiveCpuNormalizedMelReference(
    const std::vector<float>& input,
    const std::vector<float>& gpu_normalized_output, int frame_count) {
  const auto expected = CpuReferenceNormalizedMel400(input, frame_count);
  if (expected.size() != gpu_normalized_output.size()) {
    return ::testing::AssertionFailure()
           << "Reference normalized mel output returned " << expected.size()
           << " values, expected " << gpu_normalized_output.size();
  }

  for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
    for (int mel_index = 0; mel_index < detail::kPreprocess400MelBins;
         ++mel_index) {
      const size_t index =
          static_cast<size_t>(frame_index) * detail::kPreprocess400MelBins +
          static_cast<size_t>(mel_index);
      const double actual = static_cast<double>(gpu_normalized_output[index]);
      const double expected_value = static_cast<double>(expected[index]);
      const double error = std::abs(actual - expected_value);
      const double tolerance = kMaxNormalizedAbsError +
                               kMaxNormalizedRelError *
                                   std::abs(expected_value);

      if (error > tolerance) {
        return ::testing::AssertionFailure()
               << "Normalized mel mismatch at frame " << frame_index
               << ", bin " << mel_index << ": expected=" << expected_value
               << ", actual=" << actual << ", error=" << error
               << ", tolerance=" << tolerance;
      }
    }
  }

  return ::testing::AssertionSuccess();
}

}  // namespace

TEST(Frontend400Test, RejectsInvalidArguments) {
  const Frontend400StageWorkspace empty_frontend_workspace;
  const Frontend400StageWorkspace full_frontend_workspace{
      reinterpret_cast<float*>(0x1), reinterpret_cast<float*>(0x1)};
  const PreprocessWorkspace empty_preprocess_workspace;
  const PreprocessWorkspace full_preprocess_workspace{
      reinterpret_cast<float*>(0x1), reinterpret_cast<float*>(0x1),
      reinterpret_cast<float*>(0x1)};

  EXPECT_EQ(RunInspectFrontend400(nullptr, -1, 0, nullptr, nullptr, nullptr),
            cudaErrorInvalidValue);
  EXPECT_EQ(RunInspectFrontend400(nullptr, 0, 0, nullptr, nullptr, nullptr),
            cudaSuccess);
  EXPECT_EQ(RunInspectFrontend400(nullptr, 1, 1, nullptr, nullptr, nullptr),
            cudaErrorInvalidValue);
  EXPECT_EQ(
      RunFrontend400StageForTest(nullptr, -1, 0, empty_frontend_workspace),
      cudaErrorInvalidValue);
  EXPECT_EQ(RunFrontend400StageForTest(nullptr, 0, 0, empty_frontend_workspace),
            cudaSuccess);
  EXPECT_EQ(RunFrontend400StageForTest(nullptr, 1, 1, empty_frontend_workspace),
            cudaErrorInvalidValue);
  EXPECT_EQ(RunFrontend400StageForTest(reinterpret_cast<const float*>(0x1), 0, 1,
                                       full_frontend_workspace),
            cudaErrorInvalidValue);
  EXPECT_EQ(
      RunPreprocess(reinterpret_cast<const float*>(0x1), -1, 0,
                    empty_preprocess_workspace,
                    reinterpret_cast<float*>(0x1)),
            cudaErrorInvalidValue);
  EXPECT_EQ(RunPreprocess(nullptr, 0, 0, empty_preprocess_workspace,
                          reinterpret_cast<float*>(0x1)),
            cudaSuccess);
  EXPECT_EQ(RunPreprocess(reinterpret_cast<const float*>(0x1), 1, 1,
                          empty_preprocess_workspace,
                          reinterpret_cast<float*>(0x1)),
            cudaErrorInvalidValue);
  EXPECT_EQ(RunPreprocess(reinterpret_cast<const float*>(0x1), 0, 1,
                          full_preprocess_workspace,
                          reinterpret_cast<float*>(0x1)),
            cudaErrorInvalidValue);
  EXPECT_EQ(RunPreprocess(reinterpret_cast<const float*>(0x1), 1, 1,
                          full_preprocess_workspace, nullptr),
            cudaErrorInvalidValue);
  EXPECT_EQ(RunReduceMaxForTest(nullptr, 0, nullptr), cudaErrorInvalidValue);
  EXPECT_EQ(RunReduceMaxForTest(nullptr, 1, reinterpret_cast<float*>(0x1)),
            cudaErrorInvalidValue);
  EXPECT_EQ(RunReduceMaxForTest(reinterpret_cast<const float*>(0x1), 1, nullptr),
            cudaErrorInvalidValue);
}

TEST(Frontend400Test, MatchesNaiveCpuStftPowerAndMelForRandomInput) {
  const std::string cuda_failure = GetCudaAvailabilityFailure();
  if (!cuda_failure.empty()) {
    GTEST_SKIP() << cuda_failure;
  }

  std::mt19937 rng(123456789u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const std::vector<int> sample_counts = {1, 257, 400, 731, 2048};
  const test::Matrix mel_filter_bank = test::MakeMelFilterBank128();

  for (const int sample_count : sample_counts) {
    const int frame_count = sample_count / kHopLength + 1;
    SCOPED_TRACE(sample_count);
    SCOPED_TRACE(frame_count);

    std::vector<float> host_input(static_cast<size_t>(sample_count), 0.0f);
    for (float& sample : host_input) {
      sample = dist(rng);
    }

    std::vector<float2> host_stft_output(
        static_cast<size_t>(frame_count) * detail::kFft400OutputBins);
    std::vector<float> host_power_output(
        static_cast<size_t>(frame_count) * detail::kFft400OutputBins, 0.0f);
    std::vector<float> host_mel_output(
        static_cast<size_t>(frame_count) * detail::kPreprocess400MelBins,
        0.0f);

    DeviceBuffer<float> device_input;
    DeviceBuffer<float2> device_stft_output;
    DeviceBuffer<float> device_power_output;
    DeviceBuffer<float> device_mel_output;

    ASSERT_EQ(device_input.Allocate(host_input.size()), cudaSuccess);
    ASSERT_EQ(device_stft_output.Allocate(host_stft_output.size()), cudaSuccess);
    ASSERT_EQ(device_power_output.Allocate(host_power_output.size()), cudaSuccess);
    ASSERT_EQ(device_mel_output.Allocate(host_mel_output.size()), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_input.get(), host_input.data(),
                         host_input.size() * sizeof(float),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    ASSERT_EQ(
        RunInspectFrontend400(device_input.get(), sample_count, frame_count,
                              device_stft_output.get(),
                              device_power_output.get(),
                              device_mel_output.get()),
        cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(host_stft_output.data(), device_stft_output.get(),
                         host_stft_output.size() * sizeof(float2),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(host_power_output.data(), device_power_output.get(),
                         host_power_output.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(host_mel_output.data(), device_mel_output.get(),
                         host_mel_output.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    EXPECT_TRUE(
        MatchesNaiveCpuStftReference(host_input, host_stft_output, frame_count));
    EXPECT_TRUE(
        MatchesNaiveCpuPowerReference(host_input, host_power_output, frame_count));
    EXPECT_TRUE(MatchesNaiveCpuMelReference(host_input, mel_filter_bank,
                                            host_mel_output, frame_count));
  }
}

TEST(Frontend400Test, SupportsOptionalInspectionOutputs) {
  const std::string cuda_failure = GetCudaAvailabilityFailure();
  if (!cuda_failure.empty()) {
    GTEST_SKIP() << cuda_failure;
  }

  std::mt19937 rng(987654321u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const std::vector<int> sample_counts = {1, 257, 400, 731, 2048};

  for (const int sample_count : sample_counts) {
    const int frame_count = sample_count / kHopLength + 1;
    SCOPED_TRACE(sample_count);
    SCOPED_TRACE(frame_count);

    std::vector<float> host_input(static_cast<size_t>(sample_count), 0.0f);
    for (float& sample : host_input) {
      sample = dist(rng);
    }

    std::vector<float> host_power_output(
        static_cast<size_t>(frame_count) * detail::kFft400OutputBins, 0.0f);
    DeviceBuffer<float> device_input;
    DeviceBuffer<float> device_power_output;

    ASSERT_EQ(device_input.Allocate(host_input.size()), cudaSuccess);
    ASSERT_EQ(device_power_output.Allocate(host_power_output.size()), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_input.get(), host_input.data(),
                         host_input.size() * sizeof(float),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    ASSERT_EQ(RunInspectFrontend400(device_input.get(), sample_count,
                                    frame_count, nullptr,
                                    device_power_output.get(), nullptr),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(host_power_output.data(), device_power_output.get(),
                         host_power_output.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    EXPECT_TRUE(
        MatchesNaiveCpuPowerReference(host_input, host_power_output, frame_count));
  }
}

TEST(Frontend400Test, ProducesFramePeakThenChunkPeakForRandomInput) {
  const std::string cuda_failure = GetCudaAvailabilityFailure();
  if (!cuda_failure.empty()) {
    GTEST_SKIP() << cuda_failure;
  }

  std::mt19937 rng(42424242u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const std::vector<int> sample_counts = {1, 257, 400, 731, 2048};
  const test::Matrix mel_filter_bank = test::MakeMelFilterBank128();

  for (const int sample_count : sample_counts) {
    const int frame_count = sample_count / kHopLength + 1;
    const int mel_value_count = frame_count * detail::kPreprocess400MelBins;
    SCOPED_TRACE(sample_count);
    SCOPED_TRACE(frame_count);

    std::vector<float> host_input(static_cast<size_t>(sample_count), 0.0f);
    for (float& sample : host_input) {
      sample = dist(rng);
    }

    DeviceBuffer<float> device_input;
    DeviceBuffer<float> device_mel_output;
    DeviceBuffer<float> device_frame_peak_output;
    DeviceBuffer<float> device_chunk_peak_output;
    float host_chunk_peak_output = 0.0f;

    ASSERT_EQ(device_input.Allocate(host_input.size()), cudaSuccess);
    ASSERT_EQ(device_mel_output.Allocate(static_cast<size_t>(mel_value_count)),
              cudaSuccess);
    ASSERT_EQ(device_frame_peak_output.Allocate(static_cast<size_t>(frame_count)),
              cudaSuccess);
    ASSERT_EQ(device_chunk_peak_output.Allocate(1), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_input.get(), host_input.data(),
                         host_input.size() * sizeof(float),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    const Frontend400StageWorkspace workspace{device_mel_output.get(),
                                              device_frame_peak_output.get()};
    ASSERT_EQ(RunFrontend400StageForTest(device_input.get(), sample_count,
                                         frame_count, workspace),
              cudaSuccess);
    ASSERT_EQ(RunReduceMaxForTest(device_frame_peak_output.get(), frame_count,
                                  device_chunk_peak_output.get()),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&host_chunk_peak_output, device_chunk_peak_output.get(),
                         sizeof(host_chunk_peak_output), cudaMemcpyDeviceToHost),
              cudaSuccess);

    EXPECT_TRUE(MatchesNaiveCpuMelPeakReference(
        host_input, mel_filter_bank, host_chunk_peak_output, frame_count));
  }
}

TEST(Frontend400Test, MatchesSplitWhisperNormalizePipeline) {
  const std::string cuda_failure = GetCudaAvailabilityFailure();
  if (!cuda_failure.empty()) {
    GTEST_SKIP() << cuda_failure;
  }

  std::mt19937 rng(20260310u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const std::vector<int> sample_counts = {1, 257, 400, 731, 2048};

  for (const int sample_count : sample_counts) {
    const int frame_count = sample_count / kHopLength + 1;
    const int mel_value_count = frame_count * detail::kPreprocess400MelBins;
    SCOPED_TRACE(sample_count);
    SCOPED_TRACE(frame_count);

    std::vector<float> host_input(static_cast<size_t>(sample_count), 0.0f);
    for (float& sample : host_input) {
      sample = dist(rng);
    }

    std::vector<float> host_normalized_output(static_cast<size_t>(mel_value_count),
                                              0.0f);
    DeviceBuffer<float> device_input;
    DeviceBuffer<float> device_mel_output;
    DeviceBuffer<float> device_frame_peak_output;
    DeviceBuffer<float> device_chunk_peak_output;
    DeviceBuffer<float> device_normalized_output;

    ASSERT_EQ(device_input.Allocate(host_input.size()), cudaSuccess);
    ASSERT_EQ(device_mel_output.Allocate(static_cast<size_t>(mel_value_count)),
              cudaSuccess);
    ASSERT_EQ(device_frame_peak_output.Allocate(static_cast<size_t>(frame_count)),
              cudaSuccess);
    ASSERT_EQ(device_chunk_peak_output.Allocate(1), cudaSuccess);
    ASSERT_EQ(device_normalized_output.Allocate(
                  static_cast<size_t>(mel_value_count)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_input.get(), host_input.data(),
                         host_input.size() * sizeof(float),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    const PreprocessWorkspace workspace{
        device_mel_output.get(), device_frame_peak_output.get(),
        device_chunk_peak_output.get()};
    ASSERT_EQ(RunPreprocess(device_input.get(), sample_count, frame_count,
                            workspace, device_normalized_output.get()),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(host_normalized_output.data(),
                         device_normalized_output.get(),
                         host_normalized_output.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    EXPECT_TRUE(MatchesNaiveCpuNormalizedMelReference(
        host_input, host_normalized_output, frame_count));
  }
}

TEST(Frontend400Test, ReduceMaxHandlesAllNegativeInputs) {
  const std::string cuda_failure = GetCudaAvailabilityFailure();
  if (!cuda_failure.empty()) {
    GTEST_SKIP() << cuda_failure;
  }

  std::vector<float> host_input(detail::kCudaWarpSize + 1, 0.0f);
  for (size_t index = 0; index < host_input.size(); ++index) {
    host_input[index] = -100.0f - static_cast<float>(index);
  }

  DeviceBuffer<float> device_input;
  DeviceBuffer<float> device_max_output;
  float host_max_output = 0.0f;

  ASSERT_EQ(device_input.Allocate(host_input.size()), cudaSuccess);
  ASSERT_EQ(device_max_output.Allocate(1), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_input.get(), host_input.data(),
                       host_input.size() * sizeof(float),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  ASSERT_EQ(RunReduceMaxForTest(device_input.get(),
                                static_cast<int>(host_input.size()),
                                device_max_output.get()),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&host_max_output, device_max_output.get(),
                       sizeof(host_max_output), cudaMemcpyDeviceToHost),
            cudaSuccess);

  const float expected =
      *std::max_element(host_input.begin(), host_input.end());
  ASSERT_TRUE(std::isfinite(expected));
  EXPECT_FLOAT_EQ(host_max_output, expected);
  EXPECT_LT(host_max_output, 0.0f);
}

}  // namespace longwhisper
