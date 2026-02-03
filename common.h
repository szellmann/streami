// Copyright 2026-2026 Stefan Zellmann
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <iostream>
#include <stdio.h>

// os
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#ifdef OPAQUE
#undef OPAQUE
#endif
#endif
#ifdef __GNUC__
#include <execinfo.h>
#include <sys/time.h>
#endif

#ifndef __CUDACC__
# define __host__
# define __device__
#endif
#ifndef __both__
# define __both__ __host__ __device__
#endif

// ours
#include "vecmath.h"

namespace streami {

// ==================================================================
// RNG
// ==================================================================

template<unsigned int N=4>
struct LCG
{
  inline __both__ LCG()
  { /* intentionally empty so we can use it in device vars that
       don't allow dynamic initialization (ie, PRD) */
  }
  inline __both__ LCG(unsigned int val0, unsigned int val1)
  { init(val0,val1); }

  inline __both__ LCG(const vecmath::vec2i &seed)
  { init((unsigned)seed.x,(unsigned)seed.y); }
  inline __both__ LCG(const vecmath::vec2ui &seed)
  { init(seed.x,seed.y); }
  
  inline __both__ void init(unsigned int val0, unsigned int val1)
  {
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;
  
    for (unsigned int n = 0; n < N; n++) {
      s0 += 0x9e3779b9;
      v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
      v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
    }
    state = v0;
  }

  // Generate random unsigned int in [0, 2^24)
  inline __both__ float operator() ()
  {
    const uint32_t LCG_A = 1664525u;
    const uint32_t LCG_C = 1013904223u;
    state = (LCG_A * state + LCG_C);
    return (state & 0x00FFFFFF) / (float) 0x01000000;
  }

  // For compat. with visionaray
  inline __both__ float next()
  {
    return operator()();
  }

  uint32_t state;
};

typedef LCG<4> Random;

// ==================================================================
// cuda atomic
// ==================================================================

__device__
inline float atomicMin(float *address, float val) {
  int ret = __float_as_int(*address);
  while (val < __int_as_float(ret)) {
    int old = ret;
    if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
        break;
  }
  return __int_as_float(ret);
}

__device__
inline float atomicMax(float *address, float val) {
  int ret = __float_as_int(*address);
  while (val > __int_as_float(ret)) {
    int old = ret;
    if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
      break;
  }
  return __int_as_float(ret);
}


// ==================================================================
// cuda call
// ==================================================================

#ifndef NDEBUG
#define CUDA_SAFE_CALL(FUNC) { cuda_safe_call((FUNC), __FILE__, __LINE__); }
#else
#define CUDA_SAFE_CALL(FUNC) FUNC
#endif
#define CUDA_SAFE_CALL_X(FUNC) { cuda_safe_call((FUNC), __FILE__, __LINE__, true); }

inline void cuda_safe_call(
    cudaError_t code, const char *file, int line, bool fatal=false)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s %s:%i\n", cudaGetErrorString(code), file, line);
    if (fatal)
      exit(code);
  }
}

} // streami



