#include "cuda_fp16.h"
#ifndef WMMA_SDDMM_H
#define WMMA_SDDMM_H

namespace sddmm{

cudaError_t wmmaSddmm_4b(int m_vec, int k, int n,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_values, 
    int vec_length);

cudaError_t wmmaSddmm_8b(int m_vec, int k, int n,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_values, 
    int vec_length);

cudaError_t wmmaSddmm_16b(int m_vec, int k, int n,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_values, 
    int vec_length);

} // namespace sddmm

#endif
