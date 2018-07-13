#ifndef OMPD_TYPES_H_
#define OMPD_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif

// Values for omp_device_kind
#define OMP_DEVICE_KIND_HOST 1
#define OMP_DEVICE_KIND_CUDA 2

// Values for ompd_thread_id_t
#define OMPD_THREAD_ID_PTHREAD 0
#define OMPD_THREAD_ID_LWP 1
#define OMPD_THREAD_ID_WINTHREAD 2
#define OMPD_THREAD_ID_CUDALOGICAL 3
#define OMPD_THREAD_ID_MAX 4

/**
 * Logical coordinates of OMP target device threads
 */
typedef struct ompd_dim3_t {
  ompd_word_t x;
  ompd_word_t y;
  ompd_word_t z;
} ompd_dim3_t;

typedef struct ompd_cudathread_coord_t {
  ompd_addr_t cudaDevId;
  ompd_addr_t cudaContext;
  ompd_addr_t warpSize;
  ompd_addr_t gridId;
  ompd_addr_t kernelId; // TODO (MJM) - for some reason, cuda-gdb doesn't work
                        // with grids too well.
  ompd_dim3_t gridDim;
  ompd_dim3_t blockDim;
  ompd_dim3_t blockIdx;
  ompd_dim3_t threadIdx;
} ompd_cudathread_coord_t;

#ifdef __cplusplus
}
#endif

#endif /*OMPD_TYPES_H_*/
