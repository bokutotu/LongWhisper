#include "preprocess.cuh"
#include "whisper_preprocess_reference.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace longwhisper {
namespace {

constexpr double kMaxStftAbsError = 5e-3;
constexpr double kMaxPowerAbsError = 5e-2;
constexpr double kMaxPowerRelError = 1e-4;
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

template <bool kOutputStft>
__global__ void Preprocess400TestKernel(const float* __restrict__ input,
                                        int sample_count,
                                        float2* __restrict__ stft_output,
                                        float* __restrict__ power_output) {
  __shared__ Preprocess400SharedStorage shared;
  const int tid = static_cast<int>(threadIdx.x);
  const int frame_index = static_cast<int>(blockIdx.x);

  Preprocess400<kOutputStft>(input, sample_count, stft_output, &shared);

  if (tid < detail::kFft400OutputBins) {
    power_output[static_cast<size_t>(frame_index) * detail::kFft400OutputBins +
                 static_cast<size_t>(tid)] = shared.power[tid];
  }
}

template <bool kOutputStft>
cudaError_t RunPreprocess400(const float* device_input, int sample_count,
                             int frame_count,
                             float2* device_stft_output,
                             float* device_power_output,
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
  if (device_power_output == nullptr) {
    return cudaErrorInvalidValue;
  }
  if constexpr (kOutputStft) {
    if (device_stft_output == nullptr) {
      return cudaErrorInvalidValue;
    }
  }

  Preprocess400TestKernel<kOutputStft>
      <<<frame_count, kPreprocess400Threads, 0, stream>>>(device_input,
                                                          sample_count,
                                                          device_stft_output,
                                                          device_power_output);
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
                          bin];
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

}  // namespace

TEST(Preprocess400Test, RejectsInvalidArguments) {
  EXPECT_EQ(RunPreprocess400<true>(nullptr, -1, 0, nullptr, nullptr),
            cudaErrorInvalidValue);
  EXPECT_EQ(RunPreprocess400<true>(nullptr, 0, 0, nullptr, nullptr),
            cudaSuccess);
  EXPECT_EQ(RunPreprocess400<true>(nullptr, 1, 1, nullptr, nullptr),
            cudaErrorInvalidValue);
  EXPECT_EQ(RunPreprocess400<true>(reinterpret_cast<const float*>(0x1), 0, 1,
                                   reinterpret_cast<float2*>(0x1), nullptr),
            cudaErrorInvalidValue);
  EXPECT_EQ(RunPreprocess400<true>(reinterpret_cast<const float*>(0x1), 1, 1,
                                   nullptr, reinterpret_cast<float*>(0x1)),
            cudaErrorInvalidValue);
  EXPECT_EQ(RunPreprocess400<true>(reinterpret_cast<const float*>(0x1), 1, 1,
                                   reinterpret_cast<float2*>(0x1), nullptr),
            cudaErrorInvalidValue);
  EXPECT_EQ(RunPreprocess400<false>(nullptr, 0, 0, nullptr, nullptr),
            cudaSuccess);
}

TEST(Preprocess400Test, MatchesNaiveCpuStftAndPowerForRandomInput) {
  int device_count = 0;
  const cudaError_t device_count_error = cudaGetDeviceCount(&device_count);
  if (device_count_error != cudaSuccess) {
    GTEST_SKIP() << "CUDA device unavailable: "
                 << cudaGetErrorString(device_count_error);
  }
  if (device_count == 0) {
    GTEST_SKIP() << "No CUDA device available";
  }

  std::mt19937 rng(123456789u);
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

    std::vector<float2> host_stft_output(
        static_cast<size_t>(frame_count) * detail::kFft400OutputBins);
    std::vector<float> host_power_output(
        static_cast<size_t>(frame_count) * detail::kFft400OutputBins, 0.0f);
    DeviceBuffer<float> device_input;
    DeviceBuffer<float2> device_stft_output;
    DeviceBuffer<float> device_power_output;

    ASSERT_EQ(device_input.Allocate(host_input.size()), cudaSuccess);
    ASSERT_EQ(device_stft_output.Allocate(host_stft_output.size()), cudaSuccess);
    ASSERT_EQ(device_power_output.Allocate(host_power_output.size()), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_input.get(), host_input.data(),
                         host_input.size() * sizeof(float),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    ASSERT_EQ(
        RunPreprocess400<true>(device_input.get(), sample_count, frame_count,
                               device_stft_output.get(),
                               device_power_output.get()),
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

    EXPECT_TRUE(
        MatchesNaiveCpuStftReference(host_input, host_stft_output, frame_count));
    EXPECT_TRUE(
        MatchesNaiveCpuPowerReference(host_input, host_power_output, frame_count));
  }
}

TEST(Preprocess400Test, MatchesNaiveCpuPowerWithoutStftOutput) {
  int device_count = 0;
  const cudaError_t device_count_error = cudaGetDeviceCount(&device_count);
  if (device_count_error != cudaSuccess) {
    GTEST_SKIP() << "CUDA device unavailable: "
                 << cudaGetErrorString(device_count_error);
  }
  if (device_count == 0) {
    GTEST_SKIP() << "No CUDA device available";
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

    ASSERT_EQ(
        RunPreprocess400<false>(device_input.get(), sample_count, frame_count,
                                nullptr, device_power_output.get()),
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

}  // namespace longwhisper
