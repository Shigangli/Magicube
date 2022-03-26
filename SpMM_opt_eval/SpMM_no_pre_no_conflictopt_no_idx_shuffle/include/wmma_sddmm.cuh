#include "cuda_fp16.h"
#ifndef WMMA_SDDMM_H
#define WMMA_SDDMM_H

namespace sddmm{

cudaError_t wmmaSddmm(int m_vec, int k, int n, int nonzeros_vec,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const half* __restrict__ lhs_matrix,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_values, 
    int vec_length, cudaStream_t stream, int algorithm) ;


cudaError_t wmmaSddmm(int m_vec, int k, int n, int nonzeros_vec,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const half* __restrict__ lhs_matrix,
    const half* __restrict__ rhs_matrix,
    half* __restrict__ output_values, 
    int vec_length, cudaStream_t stream, int algorithm) ;

cudaError_t wmmaSddmm(int m_vec, int k, int n, int nonzeros_vec,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const float* __restrict__ lhs_matrix,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_values, 
    int vec_length, cudaStream_t stream, int algorithm) ;

} // namespace sddmm

#endif