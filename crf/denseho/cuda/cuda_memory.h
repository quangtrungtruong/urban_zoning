
#ifndef CUDA_MEMORY_H
#define CUDA_MEMORY_H

#include "helper_cuda.h"

size_t GPU_MEMORY_ALLOCATION = 0;

#define CUDA_SAFE_CALL checkCudaErrors

void allocateCudaMemory(void ** pointer, size_t size) {
  CUDA_SAFE_CALL(cudaMalloc(pointer, size));
  GPU_MEMORY_ALLOCATION += size;
}

#endif
