#include "preprocess_cuda.h"

#include "frontend.cuh"

namespace longwhisper {
namespace {

struct Frontend400Workspace {
  float* device_mel_scratch = nullptr;
  float* device_frame_peak_output = nullptr;
};

__global__ void Frontend400Kernel(const float* __restrict__ input,
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

__global__ void ReduceMaxKernel(const float* __restrict__ input,
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

__global__ void NormalizeWhisperMelKernel(const float* __restrict__ mel_input,
                                          int mel_value_count,
                                          const float* __restrict__ peak_input,
                                          float* __restrict__ output) {
  const float peak_log_value =
      log10f(fmaxf(peak_input[0], detail::kWhisperMinimumMelEnergy));

  for (int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                   static_cast<int>(threadIdx.x);
       index < mel_value_count;
       index += static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x)) {
    output[index] =
        detail::NormalizeWhisperMel(mel_input[index], peak_log_value);
  }
}

cudaError_t RunFrontend400(const float* device_input, int sample_count,
                           int frame_count,
                           const Frontend400Workspace& workspace,
                           cudaStream_t stream) {
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

  Frontend400Kernel<<<frame_count, kFrontend400Threads, 0, stream>>>(
      device_input, sample_count, workspace.device_mel_scratch,
      workspace.device_frame_peak_output);
  return cudaGetLastError();
}

cudaError_t RunReduceMax(const float* device_input, int value_count,
                         float* device_max_output, cudaStream_t stream) {
  if (value_count <= 0) {
    return cudaErrorInvalidValue;
  }
  if (device_input == nullptr || device_max_output == nullptr) {
    return cudaErrorInvalidValue;
  }

  ReduceMaxKernel<<<1, kFrontend400Threads, 0, stream>>>(device_input,
                                                         value_count,
                                                         device_max_output);
  return cudaGetLastError();
}

cudaError_t RunNormalizeWhisperMel(const float* device_mel_input,
                                   int mel_value_count,
                                   const float* device_peak_input,
                                   float* device_output,
                                   cudaStream_t stream) {
  if (mel_value_count <= 0) {
    return cudaErrorInvalidValue;
  }
  if (device_mel_input == nullptr || device_peak_input == nullptr ||
      device_output == nullptr) {
    return cudaErrorInvalidValue;
  }

  constexpr int kThreadsPerBlock = 256;
  const int block_count = (mel_value_count + kThreadsPerBlock - 1) /
                          kThreadsPerBlock;
  NormalizeWhisperMelKernel<<<block_count, kThreadsPerBlock, 0, stream>>>(
      device_mel_input, mel_value_count, device_peak_input, device_output);
  return cudaGetLastError();
}

}  // namespace

cudaError_t RunPreprocess(const float* device_input, int sample_count,
                          int frame_count,
                          const PreprocessWorkspace& workspace,
                          float* device_output, cudaStream_t stream) {
  if (sample_count < 0 || frame_count < 0) {
    return cudaErrorInvalidValue;
  }
  if (frame_count == 0) {
    return cudaSuccess;
  }
  if (device_output == nullptr || workspace.device_mel_scratch == nullptr ||
      workspace.device_frame_peak_scratch == nullptr ||
      workspace.device_chunk_peak_scratch == nullptr) {
    return cudaErrorInvalidValue;
  }

  Frontend400Workspace frontend_workspace;
  frontend_workspace.device_mel_scratch = workspace.device_mel_scratch;
  frontend_workspace.device_frame_peak_output =
      workspace.device_frame_peak_scratch;

  cudaError_t status = RunFrontend400(device_input, sample_count, frame_count,
                                      frontend_workspace, stream);
  if (status != cudaSuccess) {
    return status;
  }

  status = RunReduceMax(workspace.device_frame_peak_scratch, frame_count,
                        workspace.device_chunk_peak_scratch, stream);
  if (status != cudaSuccess) {
    return status;
  }

  return RunNormalizeWhisperMel(workspace.device_mel_scratch,
                                frame_count * detail::kPreprocess400MelBins,
                                workspace.device_chunk_peak_scratch,
                                device_output, stream);
}

}  // namespace longwhisper
