#pragma once

#include "fft400.cuh"

namespace longwhisper {

inline constexpr int kPreprocess400Threads = 256;

struct Preprocess400SharedStorage {
  float input[detail::kFft400InputSize];
  float2 stage25[detail::kFft400ComplexSize];
  float2 spec200[detail::kFft400ComplexSize];
  float2 fft[detail::kFft400OutputBins];
  float power[detail::kFft400OutputBins];
};

static_assert(kPreprocess400Threads >= detail::kFft400OutputBins);

namespace detail {

inline constexpr int kPreprocess400HopLength = 160;
inline constexpr int kPreprocess400CenterPad = kFft400InputSize / 2;
extern __device__ __constant__ float g_preprocess400_hann[kFft400InputSize];

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

}  // namespace detail

template <bool kOutputStft>
__device__ __forceinline__ void Preprocess400(
    const float* __restrict__ audio_input, int sample_count,
    float2* __restrict__ stft_output,
    Preprocess400SharedStorage* __restrict__ shared) {
  const int tid = static_cast<int>(threadIdx.x);
  const int frame_index = static_cast<int>(blockIdx.x);

  detail::LoadWindowedFrame400(audio_input, sample_count, frame_index,
                               shared->input);

  __syncthreads();

  detail::Fft400(shared->input, shared->stage25, shared->spec200, shared->fft);
  detail::PowerSpectrum201(shared->fft, shared->power);

  // Keep shared power visible to the whole block for the next frontend stage.
  __syncthreads();

  if constexpr (kOutputStft) {
    if (tid < detail::kFft400OutputBins) {
      stft_output[static_cast<size_t>(frame_index) *
                      detail::kFft400OutputBins +
                  static_cast<size_t>(tid)] = shared->fft[tid];
    }
  }
}

}  // namespace longwhisper
