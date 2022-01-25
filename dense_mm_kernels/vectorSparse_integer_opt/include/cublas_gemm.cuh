#ifndef CUBLAS_GEMM_H
#define CUBLAS_GEMM_H
#include <cublas_v2.h>
#include "cuda_fp16.h"


cublasStatus_t cublasGeMM(cublasHandle_t handle, int m, int n, int k, 
    float* d_rhs_matrix, float* d_lhs_matrix, float* d_output_matrix);

cublasStatus_t cublasGeMM(cublasHandle_t handle, int m, int n, int k, 
    half* d_rhs_matrix, half* d_lhs_matrix, half* d_output_matrix);

cublasStatus_t cublasGeMMT(cublasHandle_t handle, int m, int n, int k, 
    float* d_rhs_matrix, float* d_lhs_matrix, float* d_output_matrix);

cublasStatus_t cublasGeMMT(cublasHandle_t handle, int m, int n, int k, 
    half* d_rhs_matrix, half* d_lhs_matrix, half* d_output_matrix);

#endif