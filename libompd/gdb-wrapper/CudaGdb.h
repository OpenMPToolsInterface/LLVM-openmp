/*
 * CudaGdb.h
 *
 * Created on: Apr 11, 2017
 *     Author: Marty McFadden
 *    Contact: mmcfadden8@llnl.gov
 *
 * This header file defines the container for all threads running on Cuda devices
 */
#ifndef CUDA_GDB_H_
#define CUDA_GDB_H_
#include <cstdint>
#include <vector>
#include <map>
#include "omp-tools.h"
#include "../src/ompd-private.h"

struct CudaThread {
  ompd_cudathread_coord_t coord;
};

class CudaGdb
{
public:
  CudaGdb();
  std::vector<CudaThread> threads;
};

#endif /* CUDA_GDB_H_*/
