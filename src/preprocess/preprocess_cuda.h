#pragma once

#include <cuda_runtime.h>

namespace longwhisper {

struct PreprocessWorkspace {
  float* device_mel_scratch = nullptr;
  float* device_frame_peak_scratch = nullptr;
  float* device_chunk_peak_scratch = nullptr;
};

cudaError_t RunPreprocess(const float* device_input, int sample_count,
                          int frame_count,
                          const PreprocessWorkspace& workspace,
                          float* device_output,
                          cudaStream_t stream = nullptr);

}  // namespace longwhisper
