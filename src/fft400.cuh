#pragma once

#include <cuda_runtime.h>

namespace longwhisper::detail {

inline constexpr int kFft400InputSize = 400;
inline constexpr int kFft400ComplexSize = 200;
inline constexpr int kFft400OutputBins = 201;
inline constexpr int kFft400RadixA = 8;
inline constexpr int kFft400RadixB = 25;

static_assert(kFft400InputSize == 2 * kFft400ComplexSize);
static_assert(kFft400ComplexSize == kFft400RadixA * kFft400RadixB);
static_assert(kFft400OutputBins == kFft400InputSize / 2 + 1);

extern __device__ __constant__ float2 g_fft400_w5[5];
extern __device__ __constant__ float2 g_fft400_stage25_post[5 * 25];
extern __device__ __constant__ float2 g_fft400_twiddle200[8 * 25];
extern __device__ __constant__ float2 g_fft400_w8[8 * 8];
extern __device__ __constant__ float2 g_fft400_rfft_twiddle[kFft400OutputBins];

__device__ __forceinline__ float2 CAdd(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float2 CConj(float2 a) {
  return make_float2(a.x, -a.y);
}

__device__ __forceinline__ float2 CMul(float2 a, float2 b) {
  return make_float2(fmaf(-a.y, b.y, a.x * b.x),
                     fmaf(a.x, b.y, a.y * b.x));
}

__device__ __forceinline__ float2 LoadPacked200FromReal400(
    const float* __restrict__ input, int m) {
  return make_float2(input[2 * m], input[2 * m + 1]);
}

__device__ __forceinline__ int Stage25Index(int n1, int a, int c) {
  return ((n1 * 5 + a) * 5 + c);
}

__device__ __forceinline__ float2 Dft5PackedOne(
    const float* __restrict__ input, int base_m, int stride_m, int k) {
  float2 acc = make_float2(0.0f, 0.0f);

#pragma unroll
  for (int n = 0; n < 5; ++n) {
    const float2 x = LoadPacked200FromReal400(input, base_m + n * stride_m);
    const float2 w = g_fft400_w5[(n * k) % 5];
    acc = CAdd(acc, CMul(x, w));
  }

  return acc;
}

// FFT math only. Callers own shared-memory layout and output policy.
__device__ __forceinline__ void Fft400(const float* __restrict__ input,
                                       float2* __restrict__ stage25,
                                       float2* __restrict__ spec200,
                                       float2* __restrict__ output) {
  const int tid = static_cast<int>(threadIdx.x);

  if (tid < kFft400ComplexSize) {
    const int linear = tid;
    const int n1 = linear / kFft400RadixB;
    const int rem = linear - n1 * kFft400RadixB;
    const int a = rem / 5;
    const int c = rem - a * 5;

    const int base_m = n1 + kFft400RadixA * a;
    stage25[Stage25Index(n1, a, c)] = Dft5PackedOne(input, base_m, 40, c);
  }

  __syncthreads();

  float2 y = make_float2(0.0f, 0.0f);

  if (tid < kFft400ComplexSize) {
    const int k2 = tid / kFft400RadixA;
    const int n1 = tid - k2 * kFft400RadixA;
    const int c = k2 % 5;

    float2 acc = make_float2(0.0f, 0.0f);

#pragma unroll
    for (int a = 0; a < 5; ++a) {
      const float2 v = stage25[Stage25Index(n1, a, c)];
      const float2 w = g_fft400_stage25_post[a * 25 + k2];
      acc = CAdd(acc, CMul(v, w));
    }

    y = CMul(acc, g_fft400_twiddle200[n1 * 25 + k2]);
  }

  if (tid < kFft400ComplexSize) {
    const int k2 = tid / kFft400RadixA;
    const int k1 = tid & 7;
    const unsigned mask = __activemask();

    float2 z = make_float2(0.0f, 0.0f);

#pragma unroll
    for (int n1 = 0; n1 < kFft400RadixA; ++n1) {
      const float yr = __shfl_sync(mask, y.x, n1, kFft400RadixA);
      const float yi = __shfl_sync(mask, y.y, n1, kFft400RadixA);
      const float2 v = make_float2(yr, yi);
      const float2 w = g_fft400_w8[n1 * 8 + k1];
      z = CAdd(z, CMul(v, w));
    }

    const int k = k2 + 25 * k1;
    spec200[k] = z;
  }

  __syncthreads();

  if (tid < kFft400OutputBins) {
    const int k = tid;
    const int km =
        (k == 0 || k == kFft400ComplexSize) ? 0 : (kFft400ComplexSize - k);

    const float2 a = spec200[k % kFft400ComplexSize];
    const float2 b = CConj(spec200[km]);

    const float2 even =
        make_float2(0.5f * (a.x + b.x), 0.5f * (a.y + b.y));
    const float2 odd =
        make_float2(0.5f * (a.y - b.y), -0.5f * (a.x - b.x));

    float2 x = CAdd(even, CMul(g_fft400_rfft_twiddle[k], odd));
    if (k == 0 || k == kFft400ComplexSize) {
      x.y = 0.0f;
    }

    output[k] = x;
  }
}

}  // namespace longwhisper::detail
