#pragma once

#include <cstdint>

#include "fft400.cuh"
#include "gemm.cuh"

namespace longwhisper {

namespace detail {

inline constexpr int kPreprocess400HopLength = 160;
inline constexpr int kPreprocess400CenterPad = kFft400InputSize / 2;
inline constexpr int kPreprocess400MelBins = 128;
inline constexpr int kPreprocess400MelFilterNonZeros = 394;
inline constexpr int kPreprocess400MelMaxNonZerosPerRow = 9;
inline constexpr int kCudaWarpSize = 32;
inline constexpr float kWhisperMinimumMelEnergy = 1.0e-10f;
inline constexpr float kWhisperLogMelDynamicRange = 8.0f;
inline constexpr float kWhisperLogMelAffineBias = 4.0f;
inline constexpr float kWhisperLogMelAffineDivisor = 4.0f;

extern __device__ __constant__ float g_preprocess400_hann[kFft400InputSize];
extern __device__ __constant__ std::uint16_t
    g_preprocess400_mel_row_offsets[kPreprocess400MelBins + 1];
extern __device__ __constant__ std::uint16_t
    g_preprocess400_mel_col_indices[kPreprocess400MelFilterNonZeros];
extern __device__ __constant__ float
    g_preprocess400_mel_values[kPreprocess400MelFilterNonZeros];

__device__ __forceinline__ int ReflectPadIndex(int index, int sample_count) {
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

__device__ __forceinline__ void LoadWindowedFrame400(
    const float* __restrict__ audio_input, int sample_count, int frame_index,
    float* __restrict__ shared_input) {
  const int tid = static_cast<int>(threadIdx.x);
  const int frame_begin =
      frame_index * kPreprocess400HopLength - kPreprocess400CenterPad;

  for (int offset = tid; offset < kFft400InputSize; offset += blockDim.x) {
    const int source_index =
        ReflectPadIndex(frame_begin + offset, sample_count);
    shared_input[offset] =
        audio_input[source_index] * g_preprocess400_hann[offset];
  }
}

__device__ __forceinline__ void PowerSpectrum201(
    const float2* __restrict__ fft_output, float* __restrict__ power_output) {
  const int tid = static_cast<int>(threadIdx.x);
  if (tid < kFft400OutputBins) {
    const float2 value = fft_output[tid];
    power_output[tid] = fmaf(value.x, value.x, value.y * value.y);
  }
}

__device__ __forceinline__ void MelFilter128(
    const float* __restrict__ power_input, float* __restrict__ mel_output) {
  gemm::CsrMatVec<kPreprocess400MelMaxNonZerosPerRow>(
      kPreprocess400MelBins, g_preprocess400_mel_row_offsets,
      g_preprocess400_mel_col_indices, g_preprocess400_mel_values, power_input,
      mel_output);
}

__device__ __forceinline__ float NegativeInfinity() {
  return __uint_as_float(0xff800000u);
}

__device__ __forceinline__ float WhisperMinimumMelLog10() {
  return log10f(kWhisperMinimumMelEnergy);
}

__device__ __forceinline__ float WarpReduceMax(float value) {
#pragma unroll
  for (int offset = kCudaWarpSize / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(__activemask(), value, offset));
  }
  return value;
}

template <int kBlockSize>
__device__ __forceinline__ float BlockReduceMax(
    float value, float* __restrict__ shared_warp_max) {
  static_assert(kBlockSize >= kCudaWarpSize);
  static_assert((kBlockSize % kCudaWarpSize) == 0);

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & (kCudaWarpSize - 1);
  const int warp = tid / kCudaWarpSize;
  const int warp_count = kBlockSize / kCudaWarpSize;

  value = WarpReduceMax(value);

  if (lane == 0) {
    shared_warp_max[warp] = value;
  }

  __syncthreads();

  if (warp == 0) {
    // Inactive lanes must contribute the max identity to preserve
    // all-negative reductions.
    value = (lane < warp_count) ? shared_warp_max[lane] : NegativeInfinity();
    value = WarpReduceMax(value);
    if (lane == 0) {
      shared_warp_max[0] = value;
    }
  }

  __syncthreads();
  return shared_warp_max[0];
}

__device__ __forceinline__ float Log10ClampWhisperMel(float value,
                                                      float peak_log_value) {
  value = log10f(fmaxf(value, kWhisperMinimumMelEnergy));
  return fmaxf(value, peak_log_value - kWhisperLogMelDynamicRange);
}

__device__ __forceinline__ float AffineNormalizeWhisperMel(float value) {
  return (value + kWhisperLogMelAffineBias) / kWhisperLogMelAffineDivisor;
}

__device__ __forceinline__ float NormalizeWhisperMel(float value,
                                                     float peak_log_value) {
  return AffineNormalizeWhisperMel(
      Log10ClampWhisperMel(value, peak_log_value));
}

}  // namespace detail

inline constexpr int kFrontend400Threads = 256;

struct Frontend400SharedStorage {
  float input[detail::kFft400InputSize];
  float2 stage25[detail::kFft400ComplexSize];
  float2 spec200[detail::kFft400ComplexSize];
  float2 fft[detail::kFft400OutputBins];
  float power[detail::kFft400OutputBins];
  float mel[detail::kPreprocess400MelBins];
};

static_assert(kFrontend400Threads >= detail::kFft400OutputBins);
static_assert(kFrontend400Threads >= detail::kPreprocess400MelBins);

__device__ __forceinline__ void Frontend400(
    const float* __restrict__ audio_input, int sample_count,
    Frontend400SharedStorage* __restrict__ shared) {
  const int frame_index = static_cast<int>(blockIdx.x);

  detail::LoadWindowedFrame400(audio_input, sample_count, frame_index,
                               shared->input);

  __syncthreads();

  detail::Fft400(shared->input, shared->stage25, shared->spec200, shared->fft);
  detail::PowerSpectrum201(shared->fft, shared->power);
  __syncthreads();
  detail::MelFilter128(shared->power, shared->mel);

  // Keep shared power and mel visible to the whole block for downstream stages.
  __syncthreads();
}

}  // namespace longwhisper
