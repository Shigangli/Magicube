#include <cublas_v2.h>
#include "../include/cublas_gemm.cuh"


cublasStatus_t cublasGeMM(cublasHandle_t handle, int m, int n, int k, 
    float* d_rhs_matrix, float* d_lhs_matrix, float* d_output_values){
        const float alpha = 1.0;
        const float beta = 0.0;
        return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, 
            &alpha, d_rhs_matrix, k, d_lhs_matrix, n, &beta, d_output_values, k);
        
}


cublasStatus_t cublasGeMM(cublasHandle_t handle, int m, int n, int k, 
    half* d_rhs_matrix, half* d_lhs_matrix, half* d_output_values){
        const half alpha = 1.0;
        const half beta = 0.0;
        return cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, 
            &alpha, d_rhs_matrix, k, d_lhs_matrix, n, &beta, d_output_values, k);
}


cublasStatus_t cublasGeMMT(cublasHandle_t handle, int m, int n, int k, 
    half* d_rhs_matrix, half* d_lhs_matrix, half* d_output_values){
        const half alpha = 1.0;
        const half beta = 0.0;
        return cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, 
            &alpha, d_rhs_matrix, k, d_lhs_matrix, k, &beta, d_output_values, n);
}

cublasStatus_t cublasGeMMT(cublasHandle_t handle, int m, int n, int k, 
    float* d_rhs_matrix, float* d_lhs_matrix, float* d_output_values){
        const float alpha = 1.0;
        const float beta = 0.0;
        return cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, 
            &alpha, d_rhs_matrix, k, d_lhs_matrix, k, &beta, d_output_values, n);
}