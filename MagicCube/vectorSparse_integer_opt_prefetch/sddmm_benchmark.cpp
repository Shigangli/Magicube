#include <cuda_runtime.h>
#include <random>
#include <assert.h>
#include <algorithm>
#include <stdio.h>
#include "include/bm_test_utils.h"
#include "include/cuda_sddmm.cuh"
#include "include/wmma_sddmm.cuh"
#include "include/cublas_gemm.cuh"
#include <fstream>
#include <string>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include "sputnik/sputnik.h"
#include <cusparse.h>


// Overloading the sputnik::CudaSddmm to support half precision
namespace sputnik {
    cudaError_t CudaSddmm(int m, int k, int n, int nonzeros,
        const int* __restrict__ row_indices,
        const int* __restrict__ row_offsets,
        const int* __restrict__ column_indices,
        const half* __restrict__ lhs_matrix,
        const half* __restrict__ rhs_matrix,
        half* __restrict__ output_values,
        cudaStream_t stream){
            return sddmm::cudaSddmm(m, k, n, nonzeros, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values, 1, 0);
        }
}

void cusparseSDDMM_(
    int m, int k, int n, int nonzeros,
    int* d_row_indices,
    int* d_row_offsets,
    int* d_col_indices,
    float* d_lhs_matrix,
    float* d_rhs_matrix,
    float* d_output_value)
{
    cusparseHandle_t handle;
    cusparseDnMatDescr_t lhs_dense, rhs_dense;
    cusparseSpMatDescr_t filter_sparse;

    cusparseCreate(&handle);

    float alpha = 1.0;
    float beta  = 0.0;

    cusparseCreateDnMat(
        &lhs_dense, m, k, k, d_lhs_matrix, CUDA_R_32F, CUSPARSE_ORDER_ROW
    );

    cusparseCreateDnMat(
        &rhs_dense, k, n, k, d_rhs_matrix, CUDA_R_32F, CUSPARSE_ORDER_COL
    );

    cusparseCreateCsr(
        &filter_sparse, m, n, nonzeros, d_row_offsets, d_col_indices, d_output_value,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F
    );

    
    size_t buffer_size;
    void* dBuffer = NULL;
    
    // get buffer
    cusparseSDDMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, lhs_dense, rhs_dense, &beta, filter_sparse, CUDA_R_32F, 
        CUSPARSE_SDDMM_ALG_DEFAULT, &buffer_size
    );

    checkCuda(cudaMalloc(&dBuffer, buffer_size));

    // execute Sddmm
    cusparseSDDMM(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, lhs_dense, rhs_dense, &beta, filter_sparse, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT,
        dBuffer
    );

    checkCuda(cudaFree(dBuffer));
    cusparseDestroyDnMat(lhs_dense);
    cusparseDestroyDnMat(rhs_dense);
    cusparseDestroySpMat(filter_sparse);
    cusparseDestroy(handle);
    /*
    cusparseHandle_t handle;
    cublasHandle_t handle_t;
    cusparseDnMatDescr_t lhs_dense, rhs_dense;
    cusparseSpMatDescr_t filter_sparse;

    cusparseCreate(&handle);
    cublasCreate(&handle_t);

    float* d_lhs_matrix_t;

    checkCuda(cudaMalloc(&d_lhs_matrix_t, (m * k) * sizeof(float)));

    float alpha = 1.0;
    float beta  = 0.0;

    // Transpose the lhs matrix
    // This transpose is necessary, as for v11.2, it only support
    // column-major lhs and rhs matrix without transpose,
    // And the output must be CSR
    cublasSgeam(handle_t, CUBLAS_OP_T, CUBLAS_OP_T, 
                m, k, &alpha, d_lhs_matrix, k, &beta, d_lhs_matrix, k, d_lhs_matrix_t, m);

    cusparseCreateDnMat(
        &lhs_dense, m, k, m, d_lhs_matrix_t, CUDA_R_32F, CUSPARSE_ORDER_COL
    );

    cusparseCreateDnMat(
        &rhs_dense, k, n, k, d_rhs_matrix, CUDA_R_32F, CUSPARSE_ORDER_COL
    );

    cusparseCreateCsr(
        &filter_sparse, m, n, nonzeros, d_row_offsets, d_col_indices, d_output_value,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F
    );

    
    size_t buffer_size;
    void* dBuffer = NULL;
    
    // get buffer
    cusparseConstrainedGeMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, lhs_dense, rhs_dense, &beta, filter_sparse, CUDA_R_32F, &buffer_size
    );

    checkCuda(cudaMalloc(&dBuffer, buffer_size));

    // execute Sddmm
    cusparseConstrainedGeMM(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, lhs_dense, rhs_dense, &beta, filter_sparse, CUDA_R_32F, dBuffer
    );

    checkCuda(cudaFree(dBuffer));
    checkCuda(cudaFree(d_lhs_matrix_t));
    cusparseDestroyDnMat(lhs_dense);
    cusparseDestroyDnMat(rhs_dense);
    cusparseDestroySpMat(filter_sparse);
    cusparseDestroy(handle);
    cublasDestroy(handle_t);
    */
}


void cusparseSDDMM_(
    int m, int k, int n, int nonzeros,
    int* d_row_indices,
    int* d_row_offsets,
    int* d_col_indices,
    half* d_lhs_matrix,
    half* d_rhs_matrix,
    half* d_output_value)
{
    printf("cuSPARSE SDDMM doesn't support half precision. \n");
}


// For benchmarking, as a set of sparse matrices are provided
// The Dim M, N, and number of nonzeros are determined by the benchmark
template <typename InType, typename OutType>
void BmFN(std::string benchmark, int dimK, int vec_length, int kernel, int alg, bool sorted, bool func, int sparse){
    // The SDDMM is D_MxN = A_MxK * B_KxN o C_MxN

    // Open the benchmark file
    std::ifstream infile(benchmark, std::ifstream::in);
    std::string line;

    // get the Size of the benchmark
    std::getline(infile, line, ',');
    const int m_vec = std::stoi(line);
    const int m = m_vec * vec_length;
    std::getline(infile, line, ',');
    const int n = std::stoi(line);
    std::getline(infile, line, '\n');
    const int nonzeros_vec = std::stoi(line);
    const int nonzeros = nonzeros_vec * vec_length;
    const int k = dimK;

    printf("Problem size: M: %d, N: %d, nnz: %d, K: %d\n", m, n, nonzeros, k);

    std::default_random_engine generator;

    if (sparse){
        // Host
        // Step 1: fetch the sparse matrix from benchmark file
        // The sparse matrix is under CSC format. 
        int *row_offsets = new int[m_vec + 1];
        for (int i = 0; i < m_vec + 1; i ++){
            if (i == m_vec) std::getline(infile, line, '\n');
            else std::getline(infile, line, ' ');
            row_offsets[i] = std::stoi(line);
        }

        int *col_indices = new int[nonzeros_vec];
        for (int i = 0; i < nonzeros_vec; i ++){
            std::getline(infile, line, ' ');
            col_indices[i] = std::stoi(line);
        }

        // Step 2: generate the lhs and rhs matrices
        InType* lhs_matrix = new InType[m * k];
        InType* rhs_matrix = new InType[n * k];

        MakeDenseMatrix<InType>(m, k, lhs_matrix, generator);
        MakeDenseMatrix<InType>(n, k, rhs_matrix, generator);

        // Step 3: generate the output matrix
        OutType *output_values = new OutType[nonzeros];

        if (func){
            // Step 4: Do the SDDMM on host
            Host_sddmm<InType, OutType>(m_vec, k, vec_length, row_offsets, col_indices, lhs_matrix, rhs_matrix, output_values);
        }

        // Device
        int *d_row_offsets, *d_col_indices, *d_row_indices;
        InType *d_lhs_matrix, *d_rhs_matrix;
        OutType *d_output_value;

        checkCuda(cudaMalloc(&d_row_offsets, (m_vec + 1) * sizeof(int)));
        checkCuda(cudaMalloc(&d_col_indices, nonzeros_vec * sizeof(int)));
        checkCuda(cudaMalloc(&d_lhs_matrix, m * k * sizeof(InType)));
        checkCuda(cudaMalloc(&d_rhs_matrix, n * k * sizeof(InType)));
        checkCuda(cudaMalloc(&d_output_value, nonzeros * sizeof(OutType)));
        checkCuda(cudaMalloc(&d_row_indices, m_vec * sizeof(int)));

        checkCuda(cudaMemcpy(d_row_offsets, row_offsets, (m_vec + 1) * sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_col_indices, col_indices, nonzeros_vec * sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_lhs_matrix, lhs_matrix, m * k * sizeof(InType), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix, n * k * sizeof(InType), cudaMemcpyHostToDevice));

        int *row_indices = new int[m_vec];
        if (sorted) {
            //printf("Sort CSR based on row length\n");
            SortedRowSwizzle(m_vec, row_offsets, row_indices);
        }
        else{
            //printf("Process the rows in order\n");
            IdentityRowSwizzle(m_vec, row_indices);
        }

        checkCuda(cudaMemcpy(d_row_indices, row_indices, m_vec * sizeof(int), cudaMemcpyHostToDevice));

        cudaProfilerStart();

        // TODO: Launch kernel
        if (kernel == 0){
            printf("Using WMMA \n");
            sddmm::wmmaSddmm(m_vec, k, n, nonzeros_vec, d_row_indices, d_row_offsets, d_col_indices, d_lhs_matrix, d_rhs_matrix, d_output_value, vec_length, 0, alg);
            // TODO: wmma
        }
        else if (kernel == 1){
            printf("Using CUDA \n");
            sddmm::cudaSddmm(m_vec, k, n, nonzeros_vec, d_row_indices, d_row_offsets, d_col_indices, d_lhs_matrix, d_rhs_matrix, d_output_value, vec_length, 0);
        }
        else if (kernel == 2){
            printf("Using Sputnik \n"); 
            sputnik::CudaSddmm(m_vec, k, n, nonzeros_vec, d_row_indices, d_row_offsets, d_col_indices, d_lhs_matrix, d_rhs_matrix, d_output_value, 0);
        }
        else if (kernel == 3){
            printf("Using cuSPARSE \n");
            cusparseSDDMM_(m_vec, k, n, nonzeros_vec, d_row_indices, d_row_offsets, d_col_indices, d_lhs_matrix, d_rhs_matrix, d_output_value);
        }
        else{
            printf("unsupported kernel\n");
            // TODO: sputnik
        }

        cudaProfilerStop();

        if (func){
            // Copy the result back to host
            OutType *output_value_cuda = new OutType[nonzeros];
            checkCuda(cudaMemcpy(output_value_cuda, d_output_value, nonzeros * sizeof(OutType), cudaMemcpyDeviceToHost)); 
            
            // Verify the result
            int errors = 0;
            for (int j=0; j < nonzeros; j++){
                // if (j < 300) printf("item %d, expect %.4f, got %.4f\n", j, (float)output_values[j], (float)output_value_cuda[j]);
                if (abs(output_value_cuda[j] - output_values[j]) > 0.5){
                    // printf("item %d, expect %.4f, got %.4f\n", j, (float)output_values[j], (float)output_value_cuda[j]);
                    errors ++;
                }
            }

            if (errors > 0) {
                printf(
                    "SDDMM does not agree with SEQUENTIAL! %d errors!\n",
                    errors);
            }else {
                printf("Results verified: they agree.\n");
            }
        }

        // Free the memory
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        cudaFree(d_row_indices);
        cudaFree(d_lhs_matrix);
        cudaFree(d_rhs_matrix);
        cudaFree(d_output_value);
        delete row_offsets;
        delete col_indices;
        delete lhs_matrix;
        delete rhs_matrix;
        delete output_values;
        delete row_indices;
    }
    // CuBLAS Dense GeMM
    else{
        // Create cublas handles
        cublasHandle_t handle;
        checkCublas(cublasCreate(&handle));
        // Initialize the input operands
        InType *lhs_matrix = new InType[m * k];
        InType *rhs_matrix = new InType[n * k];
        MakeDenseMatrix<InType>(m, k, lhs_matrix, generator);
        MakeDenseMatrix<InType>(n, k, rhs_matrix, generator);

        // Allocate and initialize device memory
        InType *d_lhs_matrix, *d_rhs_matrix;
        InType *d_output_values;

        checkCuda(cudaMalloc(&d_lhs_matrix, (m * k) * sizeof(InType)));
        checkCuda(cudaMalloc(&d_rhs_matrix, (n * k) * sizeof(InType)));
        checkCuda(cudaMalloc(&d_output_values, (m * n) * sizeof(InType)));
        
        checkCuda(cudaMemcpy(d_lhs_matrix, lhs_matrix, (m * k) * sizeof(InType), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix, (n * k) * sizeof(InType), cudaMemcpyHostToDevice));

        cudaProfilerStart();
        printf("Dense Baseline\n");
        cublasGeMMT(handle, m, n, k, d_rhs_matrix, d_lhs_matrix, d_output_values);
        cudaProfilerStop();

        InType * output_value_host = new InType[m * n];

        if (func){
            // All the rows in the output matrix
            for (int i=0; i < m; i++){
                // All the columns in the output matrix
                for (int j=0; j < n; j++){
                    // the inner product dimension
                    float out_temp = 0;
                    for (int v=0; v < k; v++){
                        out_temp += (float)lhs_matrix[i * k + v] * (float)rhs_matrix[j * k + v];
                    }
                    output_value_host[i * n + j] = (InType)out_temp;
                }
            }

            InType *output_value_cuda = new InType[m * n];
            checkCuda(cudaMemcpy(output_value_cuda, d_output_values, m * n * sizeof(InType), cudaMemcpyDeviceToHost));

            // Verify the result
            int errors = 0;
            for (int j=0; j < m * n; j++){
                // if (j < 256) printf("item %d, expect %.4f, got %.4f\n", j, (float)output_value_host[j], (float)output_value_cuda[j]);
                if (abs((float)output_value_cuda[j] - (float)output_value_host[j]) > 0.5){
                    // printf("item %d, expect %.4f, got %.4f\n", j, (float)output_value_host[j], (float)output_value_cuda[j]);
                    errors ++;
                }
            }
            if (errors > 0) {
                printf( "CuBLAS does not agree with SEQUENTIAL! %d errors!\n",errors);
            }else {
                printf("Results verified: they agree.\n");
            }
            delete output_value_cuda;
            delete output_value_host;
        }

        checkCublas(cublasDestroy(handle));

        cudaFree(d_lhs_matrix);
        cudaFree(d_rhs_matrix);
        cudaFree(d_output_values);

        delete lhs_matrix;
        delete rhs_matrix;
    }
}

int main(int argc, char **argv){
    // Helper function
    if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)){
        printf("This script does a D_mxn = A_mxk * B_nxk^T o C_mxn matrix multiplication.\n");
        printf("The C_mxn can be a sparse matrix in CSR format loaded from the benchmark [bm], or an Identity matrix.\n");
        printf("The A_mxk and B_nxk are row-major dense matrices.\n");
        printf("\n");
        printf("usage: ./sddmm_benchmark [bm] [k] [v] [kernel] [alg] [sort] [function] [sparse] [mixed]\n");
        printf("arguments\n");
        printf("bm      :   path to the sparse matrix benchmark.\n");
        printf("            e.g.: /raid/datasets/dlmc/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx\n");
        printf("k       :   the length of dimension k.\n");
        printf("v       :   the vector length of the column vector sparsity, can be {1, 2, 4, 8}. \n");
        printf("kernel  :   kernel = 0 & v=2, 4, 8,    the wmmaSpMM is used. \n");
        printf("            kernel = 1 & v=1, 2, 4, 8, the cudaSpMM is used. \n");
        printf("            kernel = 2 & v=1, the sputnik is used. \n");
        printf("            kernel = 3 & v=1, the cusparse is used. \n");
        printf("alg     :   This is used to switch between different wmma sddmm algorithms. \n");
        printf("            alg = 0, wmma + shared memory. \n");
        printf("            alg = 1, mma + additional register. \n");
        printf("            alg = 2, mma + warp shfl. \n");
        printf("            alg = 3, simulate the mma + modified tensor core. \n");
        printf("sort    :   sort = 1, the rows are sorted to balance the workload; \n");
        printf("            sort = 0, the rows are processed in order; \n");
        printf("function:   function = 1, the result of the kernel will be verified.\n");
        printf("            function = 0, the result verification is skipped\n");
        printf("sparse  :   sparse = 0, the dense version is executed as a baseline;\n");
        printf("            sparse = 1, the SpMM is executed;\n");
        printf("mixed   :   mixed = 0, use single precision; \n");
        printf("            mixed = 1, use half precision; \n");
    }
    else{
        std::string benchmark(argv[1]);
        int dimK = std::atoi(argv[2]);
        int vec_length = std::atoi(argv[3]);
        int kernel = std::atoi(argv[4]);
        int alg = std::atoi(argv[5]);
        int sorted = std::atoi(argv[6]);
        int func = std::atoi(argv[7]);
        int sparse = std::atoi(argv[8]);
        int mixed = std::atoi(argv[9]);
        if (mixed)BmFN<half, half>(benchmark, dimK, vec_length, kernel, alg, sorted, func, sparse);
        else BmFN<float, float>(benchmark, dimK, vec_length, kernel, alg, sorted, func, sparse);
    }
}