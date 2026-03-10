#pragma once

#include <cuda_runtime.h>

namespace longwhisper::gemm {

template <int kMaxNonZerosPerRow, typename OffsetT, typename IndexT>
__device__ __forceinline__ void CsrMatVec(
    int row_count, const OffsetT* __restrict__ row_offsets,
    const IndexT* __restrict__ col_indices, const float* __restrict__ values,
    const float* __restrict__ input, float* __restrict__ output) {
  static_assert(kMaxNonZerosPerRow > 0);

  const int tid = static_cast<int>(threadIdx.x);

  for (int row = tid; row < row_count; row += static_cast<int>(blockDim.x)) {
    const int begin = static_cast<int>(row_offsets[row]);
    const int end = static_cast<int>(row_offsets[row + 1]);
    float acc = 0.0f;

#pragma unroll
    for (int i = 0; i < kMaxNonZerosPerRow; ++i) {
      const int index = begin + i;
      if (index < end) {
        acc = fmaf(values[index], input[static_cast<int>(col_indices[index])],
                   acc);
      }
    }

    output[row] = acc;
  }
}

}  // namespace longwhisper::gemm
