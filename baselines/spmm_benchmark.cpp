#include <cuda_runtime.h>
#include <random>
#include <assert.h>
#include <algorithm>
#include <stdio.h>
#include "include/bm_test_utils.h"
#include "include/cuda_spmm.cuh"
#include "include/wmma_spmm.cuh"
#include "include/cublas_gemm.cuh"
#include <fstream>
#include <string>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
//#include "sputnik/sputnik.h"
#include <cusparse.h>
#include <iostream>


template <typename InType, typename OutType, typename IndexType, typename DTypeVec, typename ITypeVec, cudaDataType_t DCuSPARSE>
void BmFN(std::string benchmark, int dimK, int vec_length, int kernel, bool sorted, bool func, int sparse){

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
    std::cout << "vec_len: " << vec_length << " m_vec: " << m_vec << " m: " << m << " n: " << k << " k: " << n << "\n" ;

    // Create the A column indices

    std::default_random_engine generator;

    // SpMM
    if (sparse == 1){
        int *row_offsets = new int[m_vec + 1];
        for (int i = 0; i < m_vec + 1; i ++){
            if (i == m_vec) std::getline(infile, line, '\n');
            else std::getline(infile, line, ' ');
            row_offsets[i] = std::stoi(line);
        }
        int *col_indices = new int[nonzeros_vec];
        IndexType *col_indices_sputnik = new IndexType[nonzeros_vec];
        for (int i = 0; i < nonzeros_vec; i ++){
            std::getline(infile, line, ' ');
            col_indices[i] = std::stoi(line);
            col_indices_sputnik[i] = (IndexType)std::stoi(line);
        }

        // Initialize the input operands
        InType *values = new InType[nonzeros];
        InType *rhs_matrix = new InType[n * k];
        MakeDenseMatrix<InType>(n, k, rhs_matrix, generator);
        MakeDenseMatrix<InType>(1, nonzeros, values, generator);


        // Allocate the host output
        float *output_value_host = new float[m * k];

        if (func){
            // Initialize the output matrix with 0
            for (int i=0; i < m * k; i++){
                output_value_host[i] = 0.0f;
            }
            
            // traverse all the vector rows
            for (int i=0; i < m_vec; i++){
                // traverse all the nonzero columns in this row
                for (int j=row_offsets[i]; j < row_offsets[i+1]; j++){
                    int col_idx = col_indices[j];
                    // traverse all the elements in the vector
                    for (int v=0; v < vec_length; v++){
                        int row_idx = i * vec_length + v;
                        for (int l=0; l < k; l++){
                            output_value_host[row_idx * k + l] += (float)values[j * vec_length + v] * (float)rhs_matrix[col_idx * k + l];
                        }
                    }
                }
            }
        }// end if func


        int *row_indices = new int[m_vec];
        if (sorted) {
            //printf("Sort CSR based on row length\n");
            SortedRowSwizzle(m_vec, row_offsets, row_indices);
        }
        else{
            //printf("Process the rows in order\n");
            IdentityRowSwizzle(m_vec, row_indices);
        }

        

        // Device
        int *d_row_offsets, *d_col_indices, *d_row_indices;
        IndexType *d_col_indices_sputnik;
        InType *d_value, *d_rhs_matrix;
        OutType *d_output_value;

        checkCuda(cudaMalloc(&d_row_offsets, (m_vec + 1) * sizeof(int)));
        checkCuda(cudaMalloc(&d_col_indices, nonzeros_vec * sizeof(int)));
        checkCuda(cudaMalloc(&d_col_indices_sputnik, nonzeros_vec * sizeof(IndexType)));
        checkCuda(cudaMalloc(&d_row_indices, m_vec * sizeof(int)));

        checkCuda(cudaMalloc(&d_value, nonzeros * sizeof(InType)));
        checkCuda(cudaMalloc(&d_rhs_matrix, (n * k) * sizeof(InType)));
        checkCuda(cudaMalloc(&d_output_value, (m * k) * sizeof(OutType)));

        checkCuda(cudaMemcpy(d_row_offsets, row_offsets, (m_vec + 1) * sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_col_indices, col_indices, nonzeros_vec * sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_col_indices_sputnik, col_indices_sputnik, nonzeros_vec * sizeof(IndexType), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_row_indices, row_indices, m_vec * sizeof(int), cudaMemcpyHostToDevice));

        checkCuda(cudaMemcpy(d_value, values, nonzeros * sizeof(InType), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix, n * k * sizeof(InType), cudaMemcpyHostToDevice));
        
        cudaProfilerStart();
	float spmm_ms_avg = 0.0f;
	int NUM_PROFILES = 512;
        if (kernel == 0){
            printf("Using WMMA \n");
	    for(int iter=0; iter<16; ++iter){
                spmm::wmmaSpmm(m_vec, vec_length, k, n, d_row_indices, d_row_offsets, d_col_indices, d_value, d_rhs_matrix, d_output_value);
	    }
	    for(int iter=0; iter<NUM_PROFILES; ++iter){
	        float spmm_ms = 0.0f;
	        cudaEvent_t spmm_start;
	        cudaEvent_t spmm_end;
	        cudaEventCreate(&spmm_start);
	        cudaEventCreate(&spmm_end);
	        cudaEventRecord(spmm_start);
                spmm::wmmaSpmm(m_vec, vec_length, k, n, d_row_indices, d_row_offsets, d_col_indices, d_value, d_rhs_matrix, d_output_value);
	        cudaEventRecord(spmm_end);
	        cudaEventSynchronize(spmm_end);
	        cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
                cudaEventDestroy(spmm_start);
                cudaEventDestroy(spmm_end);
                spmm_ms_avg += spmm_ms;
	    }
        }
        else if (kernel == 1){
            printf("Using CUDA \n");
            spmm::cudaSpmm(m_vec, vec_length, k, n, d_row_indices, d_row_offsets, d_col_indices, d_value, d_rhs_matrix, d_output_value);
        }
        else if (kernel == 2){
            //printf("Using Sputnik \n");
            //DTypeVec* d_value_vec = reinterpret_cast<DTypeVec *>(d_value);
            //DTypeVec* d_rhs_matrix_vec = reinterpret_cast<DTypeVec *>(d_rhs_matrix);
            //DTypeVec* d_output_value_vec = reinterpret_cast<DTypeVec *>(d_output_value);
            //ITypeVec* d_col_indices_sputnik_vec = reinterpret_cast<ITypeVec *>(d_col_indices_sputnik);
            //sputnik::CudaSpmm(m, n, k, nonzeros, d_row_indices, d_value_vec, d_row_offsets, d_col_indices_sputnik_vec, d_rhs_matrix_vec, d_output_value_vec, 0);
        }
        else if (kernel == 3){
            printf("Using CuSPARSE \n");
            cusparseHandle_t handle;
            cusparseDnMatDescr_t rhs_dense, output_dense;
            cusparseSpMatDescr_t lhs_sparse;

            cusparseCreate(&handle);

            // create lhs sparse matrix
            cusparseCreateCsr(
                &lhs_sparse, m, n, nonzeros_vec, d_row_offsets, d_col_indices, d_value,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, DCuSPARSE
            );
            
            // create rhs dense matrix
            cusparseCreateDnMat(
                &rhs_dense, n, k, k, d_rhs_matrix, DCuSPARSE, CUSPARSE_ORDER_ROW
            );

            // create output dense matrix
            cusparseCreateDnMat(
                &output_dense, m, k, k, d_output_value, DCuSPARSE, CUSPARSE_ORDER_ROW
            );

            InType alpha = 1.0;
            InType beta  = 0.0;
            size_t buffer_size;
            void* dBuffer = NULL;

            // get buffer
            cusparseSpMM_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, lhs_sparse, rhs_dense, &beta, output_dense, DCuSPARSE, CUSPARSE_SPMM_CSR_ALG2, &buffer_size
            );

            checkCuda(cudaMalloc(&dBuffer, buffer_size));
            
            /*
            // preprocess to get additional speedup
            cusparseSpMM_preprocess(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, lhs_sparse, rhs_dense, &beta, output_dense, CUDA_R_16F, CUSPARSE_SPMM_CSR_ALG2,
                dBuffer
            );
            */

            // execute SpMM
            cusparseSpMM(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, lhs_sparse, rhs_dense, &beta, output_dense, DCuSPARSE, CUSPARSE_SPMM_CSR_ALG2,
                dBuffer
            );

            checkCuda(cudaFree(dBuffer));
            cusparseDestroyDnMat(rhs_dense);
            cusparseDestroyDnMat(output_dense);
            cusparseDestroySpMat(lhs_sparse);
            cusparseDestroy(handle);
        }
        else{
            printf("Unsupported Kernel \n");
        }
        cudaProfilerStop();
	std::cout << "vectorSparse SpMM runtime " << spmm_ms_avg/(float)NUM_PROFILES << " ms" << "\n";


        if (func){
            OutType *output_value_cuda = new OutType[m * k];
            checkCuda(cudaMemcpy(output_value_cuda, d_output_value, m * k * sizeof(OutType), cudaMemcpyDeviceToHost));

            // Verify the result
            int errors = 0;
            for (int j=0; j < m * k; j++){
                // if (j < 256) printf("item %d, expect %.4f, got %.4f\n", j, (float)output_value_host[j], (float)output_value_cuda[j]);
                if (abs((float)output_value_cuda[j] - (float)output_value_host[j]) > 0.5){
                    // if (j < 2560) printf("item %d, expect %.4f, got %.4f\n", j, (float)output_value_host[j], (float)output_value_cuda[j]);
                    errors ++;
                }
            }
            if (errors > 0) {
                printf( "SPMM does not agree with SEQUENTIAL! %d errors!\n",errors);
            }else {
                printf("Results verified: they agree.\n");
            }
            delete output_value_cuda;
        }


        // Free the memory
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        cudaFree(d_row_indices);
        cudaFree(d_col_indices_sputnik);
        cudaFree(d_value);
        cudaFree(d_rhs_matrix);
        cudaFree(d_output_value);

        delete row_offsets;
        delete col_indices;
        delete col_indices_sputnik;
        delete row_indices;
        delete values;
        delete rhs_matrix;
        delete output_value_host;
    }
    // Blocked Ell Algorithm
    else if (sparse == 2){
        // Define the Blocked Ell Size
        float sparsity = (float)nonzeros / (m * n);
        int A_num_rows = m;
        int A_ell_blocksize = vec_length;
        int A_num_col_block = (n + vec_length - 1) / vec_length;
        int A_num_col = A_num_col_block * vec_length;
        int A_num_col_block_nz = A_num_col_block * sparsity + 1;
        int A_ell_cols = A_num_col_block_nz * vec_length;

        printf("A: %d x %d. There are %d nonzero blocks, Each row has %d blocks. Block size is %d x %d\n", 
                A_num_rows, A_num_col, A_num_col_block_nz * m_vec, A_num_col_block_nz, A_ell_blocksize, A_ell_blocksize);

        // Create the matrix A
        int *A_columns = new int[A_num_col_block_nz * m_vec];
        InType *A_values = new InType[A_ell_cols * A_num_rows];

        GenerateUniformBlockedELLIndex(A_num_col_block, A_num_col_block_nz, m_vec, A_columns, generator);
        MakeDenseMatrix<InType>(A_num_rows, A_ell_cols, A_values, generator);

        // Create the matrix B
        InType* rhs_matrix = new InType[A_num_col * k];
        MakeDenseMatrix<InType>(A_num_col, k, rhs_matrix, generator);

        // Device
        int *dA_columns;
        InType * dA_value, * d_rhs_matrix;
        OutType *d_output_value;

        checkCuda(cudaMalloc(&dA_columns, (A_num_col_block_nz * m_vec) * sizeof(int)));
        checkCuda(cudaMalloc(&dA_value, (A_ell_cols * A_num_rows) * sizeof(InType)));
        checkCuda(cudaMalloc(&d_rhs_matrix, (A_num_col * k) * sizeof(InType)));
        checkCuda(cudaMalloc(&d_output_value, (A_num_rows * k) * sizeof(OutType)));

        checkCuda(cudaMemcpy(dA_columns, A_columns, (A_num_col_block_nz * m_vec) * sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(dA_value, A_values, (A_ell_cols * A_num_rows) * sizeof(InType), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix, (A_num_col * k) * sizeof(InType), cudaMemcpyHostToDevice));

        cusparseHandle_t handle_ell;
        cusparseCreate(&handle_ell);

        cusparseSpMatDescr_t lhs_sparse;
        cusparseDnMatDescr_t rhs_dense, output_dense;

        // Create sparse matrix A in blocked ELL format
        cusparseCreateBlockedEll(
            &lhs_sparse, A_num_rows, A_num_col, A_ell_blocksize,
            A_ell_cols, dA_columns, dA_value,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, DCuSPARSE
        );

        // Create dense matrix B
        cusparseCreateDnMat(&rhs_dense, A_num_col, k, k, d_rhs_matrix, DCuSPARSE, CUSPARSE_ORDER_ROW);

        // Create output dense matrix
        cusparseCreateDnMat(&output_dense, A_num_rows, k, k, d_output_value, DCuSPARSE, CUSPARSE_ORDER_ROW);

        InType alpha = 1.0;
        InType beta  = 0.0;
        size_t buffer_size;
        void* dBuffer = NULL;

        cudaProfilerStart();
        printf("Blocked ELL based SpMM\n");
        // get buffer
        cusparseSpMM_bufferSize(
            handle_ell, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, lhs_sparse, rhs_dense, &beta, output_dense, DCuSPARSE, CUSPARSE_SPMM_BLOCKED_ELL_ALG1, &buffer_size
        );

        checkCuda(cudaMalloc(&dBuffer, buffer_size));
	float dense_ms_avg = 0.0f;
	int NUM_PROFILES = 512;
	for(int iter=0; iter<16; ++iter){
            cusparseSpMM(
                handle_ell, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, lhs_sparse, rhs_dense, &beta, output_dense, DCuSPARSE, CUSPARSE_SPMM_BLOCKED_ELL_ALG1, dBuffer
            );
	}
	for(int iter=0; iter<NUM_PROFILES; ++iter){
	    float dense_ms = 0.0f;
	    cudaEvent_t dense_start;
	    cudaEvent_t dense_end;
	    cudaEventCreate(&dense_start);
	    cudaEventCreate(&dense_end);
	    cudaEventRecord(dense_start);
            cusparseSpMM(
                handle_ell, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, lhs_sparse, rhs_dense, &beta, output_dense, DCuSPARSE, CUSPARSE_SPMM_BLOCKED_ELL_ALG1, dBuffer
            );
	    cudaEventRecord(dense_end);
	    cudaEventSynchronize(dense_end);
	    cudaEventElapsedTime(&dense_ms, dense_start, dense_end);
            cudaEventDestroy(dense_start);
            cudaEventDestroy(dense_end);
            dense_ms_avg += dense_ms;
	}
        cudaProfilerStop();
	std::cout << "cusparse fp16 block ell runtime " << dense_ms_avg/(float)NUM_PROFILES << " ms" << "\n";

        // Functional verification
        if (func){
            float *output_value_host = new float[A_num_rows * k];
            // traverse all the vector rows
            for (int i=0; i < m_vec; i++){
                // traverse all the rows within the row vector group
                for (int v=0; v < A_ell_blocksize; v++){
                    int row_idx = i * A_ell_blocksize + v;
                    // travers all the output column s
                    for (int j=0; j < k; j++){
                        float psum = 0;
                        // traverse all the column blocks
                        for (int c=0; c < A_num_col_block_nz; c++){
                            int col_idx_base = A_columns[i * A_num_col_block_nz + c] * A_ell_blocksize;
                            // traverse the column block size
                            for (int cv=0; cv < A_ell_blocksize; cv++){
                                int col_idx = col_idx_base + cv;
                                // psum += (float)A_values[row_idx * A_ell_cols + c * A_ell_blocksize + cv] * (float)rhs_matrix[col_idx * k + j];
                                psum += (float)A_values[row_idx * A_ell_cols + c * A_ell_blocksize + cv] * (float)rhs_matrix[col_idx * k + j];
                            }
                        }
                        output_value_host[row_idx * k + j] = psum;
                    }
                }
            }
            

            OutType *output_value_cuda = new OutType[A_num_rows * k];
            checkCuda(cudaMemcpy(output_value_cuda, d_output_value, A_num_rows * k * sizeof(OutType), cudaMemcpyDeviceToHost));

            // Verify the result
            int errors = 0;
            for (int j=0; j < A_num_rows * k; j++){
                // if (j < 256) printf("item %d, expect %.4f, got %.4f\n", j, (float)output_value_host[j], (float)output_value_cuda[j]);
                if (abs((float)output_value_cuda[j] - (float)output_value_host[j]) > 0.5){
                    // if (j < 2560) printf("item %d, expect %.4f, got %.4f\n", j, (float)output_value_host[j], (float)output_value_cuda[j]);
                    errors ++;
                }
            }
            if (errors > 0) {
                printf( "SPMM Blocked ELL does not agree with SEQUENTIAL! %d errors!\n",errors);
            }else {
                printf("Results verified: they agree.\n");
            }
            delete output_value_cuda;
            delete output_value_host;

        }
        

        checkCuda(cudaFree(dBuffer));
        cusparseDestroyDnMat(rhs_dense);
        cusparseDestroyDnMat(output_dense);
        cusparseDestroySpMat(lhs_sparse);
        cusparseDestroy(handle_ell);
        
        cudaFree(dA_columns);
        cudaFree(dA_value);
        cudaFree(d_rhs_matrix);
        cudaFree(d_output_value);

        delete A_columns;
        delete A_values;
        delete rhs_matrix;
	
    }
    // Blocked Ell Algorithm
    else if (sparse == 3){
        // Define the Blocked Ell Size
        float sparsity = (float)nonzeros / (m * n);
        int A_num_rows = m;
        int A_ell_blocksize = vec_length;
        int A_num_col_block = (n + vec_length - 1) / vec_length;
        int A_num_col = A_num_col_block * vec_length;
        int A_num_col_block_nz = A_num_col_block * sparsity + 1;
        int A_ell_cols = A_num_col_block_nz * vec_length;

        printf("A: %d x %d. There are %d nonzero blocks, Each row has %d blocks. Block size is %d x %d\n", 
                A_num_rows, A_num_col, A_num_col_block_nz * m_vec, A_num_col_block_nz, A_ell_blocksize, A_ell_blocksize);

        // Create the matrix A
        int *A_columns = new int[A_num_col_block_nz * m_vec];
        char *A_values = new char[A_ell_cols * A_num_rows];

        GenerateUniformBlockedELLIndex(A_num_col_block, A_num_col_block_nz, m_vec, A_columns, generator);
        MakeDenseMatrix<char>(A_num_rows, A_ell_cols, A_values, generator);

        // Create the matrix B
        char* rhs_matrix = new char[A_num_col * k];
        MakeDenseMatrix<char>(A_num_col, k, rhs_matrix, generator);

        // Device
        int *dA_columns;
        char * dA_value, * d_rhs_matrix;
        int *d_output_value;

        checkCuda(cudaMalloc(&dA_columns, (A_num_col_block_nz * m_vec) * sizeof(int)));
        checkCuda(cudaMalloc(&dA_value, (A_ell_cols * A_num_rows) * sizeof(char)));
        checkCuda(cudaMalloc(&d_rhs_matrix, (A_num_col * k) * sizeof(char)));
        checkCuda(cudaMalloc(&d_output_value, (A_num_rows * k) * sizeof(OutType)));

        checkCuda(cudaMemcpy(dA_columns, A_columns, (A_num_col_block_nz * m_vec) * sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(dA_value, A_values, (A_ell_cols * A_num_rows) * sizeof(char), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix, (A_num_col * k) * sizeof(char), cudaMemcpyHostToDevice));

        cusparseHandle_t handle_ell;
        cusparseCreate(&handle_ell);

        cusparseSpMatDescr_t lhs_sparse;
        cusparseDnMatDescr_t rhs_dense, output_dense;

        // Create sparse matrix A in blocked ELL format
        cusparseCreateBlockedEll(
            &lhs_sparse, A_num_rows, A_num_col, A_ell_blocksize,
            A_ell_cols, dA_columns, dA_value,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, DCuSPARSE
        );

        // Create dense matrix B
        cusparseCreateDnMat(&rhs_dense, A_num_col, k, k, d_rhs_matrix, DCuSPARSE, CUSPARSE_ORDER_ROW);

        // Create output dense matrix
        cusparseCreateDnMat(&output_dense, A_num_rows, k, k, d_output_value, DCuSPARSE, CUSPARSE_ORDER_ROW);

        int alpha = 1.0;
        int beta  = 0.0;
        size_t buffer_size;
        void* dBuffer = NULL;

        cudaProfilerStart();
        printf("Blocked ELL based SpMM\n");
        // get buffer
        cusparseSpMM_bufferSize(
            handle_ell, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, lhs_sparse, rhs_dense, &beta, output_dense, DCuSPARSE, CUSPARSE_SPMM_BLOCKED_ELL_ALG1, &buffer_size
        );

        checkCuda(cudaMalloc(&dBuffer, buffer_size));
	float dense_ms_avg = 0.0f;
	int NUM_PROFILES = 512;
	for(int iter=0; iter<16; ++iter){
            cusparseSpMM(
                handle_ell, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, lhs_sparse, rhs_dense, &beta, output_dense, DCuSPARSE, CUSPARSE_SPMM_BLOCKED_ELL_ALG1, dBuffer
            );
	}
	for(int iter=0; iter<NUM_PROFILES; ++iter){
	    float dense_ms = 0.0f;
	    cudaEvent_t dense_start;
	    cudaEvent_t dense_end;
	    cudaEventCreate(&dense_start);
	    cudaEventCreate(&dense_end);
	    cudaEventRecord(dense_start);
            cusparseSpMM(
                handle_ell, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, lhs_sparse, rhs_dense, &beta, output_dense, DCuSPARSE, CUSPARSE_SPMM_BLOCKED_ELL_ALG1, dBuffer
            );
	    cudaEventRecord(dense_end);
	    cudaEventSynchronize(dense_end);
	    cudaEventElapsedTime(&dense_ms, dense_start, dense_end);
            cudaEventDestroy(dense_start);
            cudaEventDestroy(dense_end);
            dense_ms_avg += dense_ms;
	}
        cudaProfilerStop();
	std::cout << "cusparse int8 block ell runtime " << dense_ms_avg/(float)NUM_PROFILES << " ms" << "\n";

        // Functional verification
        if (func){
            float *output_value_host = new float[A_num_rows * k];
            // traverse all the vector rows
            for (int i=0; i < m_vec; i++){
                // traverse all the rows within the row vector group
                for (int v=0; v < A_ell_blocksize; v++){
                    int row_idx = i * A_ell_blocksize + v;
                    // travers all the output column s
                    for (int j=0; j < k; j++){
                        float psum = 0;
                        // traverse all the column blocks
                        for (int c=0; c < A_num_col_block_nz; c++){
                            int col_idx_base = A_columns[i * A_num_col_block_nz + c] * A_ell_blocksize;
                            // traverse the column block size
                            for (int cv=0; cv < A_ell_blocksize; cv++){
                                int col_idx = col_idx_base + cv;
                                // psum += (float)A_values[row_idx * A_ell_cols + c * A_ell_blocksize + cv] * (float)rhs_matrix[col_idx * k + j];
                                psum += (float)A_values[row_idx * A_ell_cols + c * A_ell_blocksize + cv] * (float)rhs_matrix[col_idx * k + j];
                            }
                        }
                        output_value_host[row_idx * k + j] = psum;
                    }
                }
            }
            

            OutType *output_value_cuda = new OutType[A_num_rows * k];
            checkCuda(cudaMemcpy(output_value_cuda, d_output_value, A_num_rows * k * sizeof(OutType), cudaMemcpyDeviceToHost));

            // Verify the result
            int errors = 0;
            for (int j=0; j < A_num_rows * k; j++){
                // if (j < 256) printf("item %d, expect %.4f, got %.4f\n", j, (float)output_value_host[j], (float)output_value_cuda[j]);
                if (abs((float)output_value_cuda[j] - (float)output_value_host[j]) > 0.5){
                    // if (j < 2560) printf("item %d, expect %.4f, got %.4f\n", j, (float)output_value_host[j], (float)output_value_cuda[j]);
                    errors ++;
                }
            }
            if (errors > 0) {
                printf( "SPMM Blocked ELL does not agree with SEQUENTIAL! %d errors!\n",errors);
            }else {
                printf("Results verified: they agree.\n");
            }
            delete output_value_cuda;
            delete output_value_host;

        }
        

        checkCuda(cudaFree(dBuffer));
        cusparseDestroyDnMat(rhs_dense);
        cusparseDestroyDnMat(output_dense);
        cusparseDestroySpMat(lhs_sparse);
        cusparseDestroy(handle_ell);
        
        cudaFree(dA_columns);
        cudaFree(dA_value);
        cudaFree(d_rhs_matrix);
        cudaFree(d_output_value);

        delete A_columns;
        delete A_values;
        delete rhs_matrix;

    }
    else if (sparse == -1){
       //cublasGemmEx(handle, 
       //             CUBLAS_OP_N,       
       //             CUBLAS_OP_N,      
       //             k,                   
       //             m,                   
       //             n,               
       //             1,            
       //             d_rhs_matrix,   
       //             CUDA_R_8I,    
       //             k,                   
       //             d_lhs_matrix, 
       //             CUDA_R_8I,   
       //             n,                   
       //             1,           
       //             d_output_values,
       //             CUDA_R_32I,   
       //             k,                 
       //             CUDA_R_32I, 
       //             CUBLAS_GEMM_ALGO0   
       //             )

        // Create cublas handles
        cublasHandle_t handle;
        checkCublas(cublasCreate(&handle));
        

        // Initialize the input operands
        char *lhs_matrix = new char[m * n];
        char *rhs_matrix = new char[n * k];
        MakeDenseMatrix<char>(m, n, lhs_matrix, generator);
        MakeDenseMatrix<char>(n, k, rhs_matrix, generator);

        
        // Allocate and initialize device memory
        char *d_lhs_matrix, *d_rhs_matrix;
        int *d_output_values;

	int alpha = 1;
	int beta = 0;
        checkCuda(cudaMalloc(&d_lhs_matrix, (m * n) * sizeof(char)));
        checkCuda(cudaMalloc(&d_rhs_matrix, (n * k) * sizeof(char)));
        checkCuda(cudaMalloc(&d_output_values, (m * k) * sizeof(int)));
        
        checkCuda(cudaMemcpy(d_lhs_matrix, lhs_matrix, (m * n) * sizeof(char), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix, (n * k) * sizeof(char), cudaMemcpyHostToDevice));


        cudaProfilerStart();
        printf("8-bit dense Baseline\n");
	float dense_ms_avg = 0.0f;
	int NUM_PROFILES = 512;
	for(int iter=0; iter<16; ++iter){
	    //cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, &alpha, d_rhs_matrix, CUDA_R_8I, k, d_lhs_matrix, CUDA_R_8I, n, &beta, d_output_values, CUDA_R_32I, k, CUDA_R_32I, CUBLAS_GEMM_ALGO0_TENSOR_OP);
	    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, &alpha, d_rhs_matrix, CUDA_R_8I, k, d_lhs_matrix, CUDA_R_8I, n, &beta, d_output_values, CUDA_R_32I, k, CUDA_R_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	    //cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, &alpha, d_rhs_matrix, CUDA_R_8I, k, d_lhs_matrix, CUDA_R_8I, n, &beta, d_output_values, CUDA_R_32I, k, CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
	}
	for(int iter=0; iter<NUM_PROFILES; ++iter){
	    float dense_ms = 0.0f;
	    cudaEvent_t dense_start;
	    cudaEvent_t dense_end;
	    cudaEventCreate(&dense_start);
	    cudaEventCreate(&dense_end);
	    cudaEventRecord(dense_start);
	    //cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, &alpha, d_rhs_matrix, CUDA_R_8I, k, d_lhs_matrix, CUDA_R_8I, n, &beta, d_output_values, CUDA_R_32I, k, CUDA_R_32I, CUBLAS_GEMM_ALGO0_TENSOR_OP);
	    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, &alpha, d_rhs_matrix, CUDA_R_8I, k, d_lhs_matrix, CUDA_R_8I, n, &beta, d_output_values, CUDA_R_32I, k, CUDA_R_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	    //cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, &alpha, d_rhs_matrix, CUDA_R_8I, k, d_lhs_matrix, CUDA_R_8I, n, &beta, d_output_values, CUDA_R_32I, k, CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
	    cudaEventRecord(dense_end);
	    cudaEventSynchronize(dense_end);
	    cudaEventElapsedTime(&dense_ms, dense_start, dense_end);
            cudaEventDestroy(dense_start);
            cudaEventDestroy(dense_end);
            dense_ms_avg += dense_ms;
	}
        cudaProfilerStop();
	std::cout << "cublase int8 dense for spmm runtime " << dense_ms_avg/(float)NUM_PROFILES << " ms" << "\n";
    }
    else{
        // Create cublas handles
        cublasHandle_t handle;
        checkCublas(cublasCreate(&handle));
        

        // Initialize the input operands
        InType *lhs_matrix = new InType[m * n];
        InType *rhs_matrix = new InType[n * k];
        MakeDenseMatrix<InType>(m, n, lhs_matrix, generator);
        MakeDenseMatrix<InType>(n, k, rhs_matrix, generator);

        
        // Allocate and initialize device memory
        InType *d_lhs_matrix, *d_rhs_matrix;
        InType *d_output_values;

        checkCuda(cudaMalloc(&d_lhs_matrix, (m * n) * sizeof(InType)));
        checkCuda(cudaMalloc(&d_rhs_matrix, (n * k) * sizeof(InType)));
        checkCuda(cudaMalloc(&d_output_values, (m * k) * sizeof(InType)));
        
        checkCuda(cudaMemcpy(d_lhs_matrix, lhs_matrix, (m * n) * sizeof(InType), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix, (n * k) * sizeof(InType), cudaMemcpyHostToDevice));


        cudaProfilerStart();
        printf("Dense Baseline\n");
	float dense_ms_avg = 0.0f;
	int NUM_PROFILES = 512;
	for(int iter=0; iter<16; ++iter){
            cublasGeMM(handle, m, n, k, d_rhs_matrix, d_lhs_matrix, d_output_values);
	}
	for(int iter=0; iter<NUM_PROFILES; ++iter){
	    float dense_ms = 0.0f;
	    cudaEvent_t dense_start;
	    cudaEvent_t dense_end;
	    cudaEventCreate(&dense_start);
	    cudaEventCreate(&dense_end);
	    cudaEventRecord(dense_start);
            cublasGeMM(handle, m, n, k, d_rhs_matrix, d_lhs_matrix, d_output_values);
	    cudaEventRecord(dense_end);
	    cudaEventSynchronize(dense_end);
	    cudaEventElapsedTime(&dense_ms, dense_start, dense_end);
            cudaEventDestroy(dense_start);
            cudaEventDestroy(dense_end);
            dense_ms_avg += dense_ms;
	}
        cudaProfilerStop();
	std::cout << "cublase fp16 dense for spmm runtime " << dense_ms_avg/(float)NUM_PROFILES << " ms" << "\n";

        InType * output_value_host = new InType[m * k];

        if (func){
            // All the rows in the output matrix
            for (int i=0; i < m; i++){
                // All the columns in the output matrix
                for (int j=0; j < k; j++){
                    // the inner product dimension
                    float out_temp = 0;
                    for (int v=0; v < n; v++){
                        out_temp += (float)lhs_matrix[i * n + v] * (float)rhs_matrix[v * k + j];
                    }
                    output_value_host[i * k + j] = (InType)out_temp;
                }
            }

            InType *output_value_cuda = new InType[m * k];
            checkCuda(cudaMemcpy(output_value_cuda, d_output_values, m * k * sizeof(InType), cudaMemcpyDeviceToHost));

            // Verify the result
            int errors = 0;
            for (int j=0; j < m * k; j++){
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
        printf("This script does a A_mxn * B_nxk = C_mxk matrix multiplication.\n");
        printf("The A_mxn can be a sparse matrix in CSR format loaded from the benchmark [bm], or a row-major dense matrix.\n");
        printf("The B_nxk and C_mxk are row-major dense matrices.\n");
        printf("\n");
        printf("usage: ./spmm_benchmark [bm] [k] [v] [kernel] [sort] [function] [sparse] [mixed]\n");
        printf("arguments\n");
        printf("bm      :   path to the sparse matrix benchmark.\n");
        printf("            e.g.: /raid/datasets/dlmc/rn50/random_pruning/0.5/bottleneck_2_block_group3_5_1.smtx\n");
        printf("k       :   the length of dimension k.\n");
        printf("v       :   the vector length of the column vector sparsity, can be {1, 2, 4, 8}. \n");
        printf("kernel  :   kernel = 0 & v=2, 4, 8,    the wmmaSpMM is used. \n");
        printf("            kernel = 1 & v=1, 2, 4, 8, the cudaSpMM is used. \n");
        printf("            kernel = 2 & v=1, the sputnik is used. \n");
        printf("            kernel = 3 & v=1, the cusparse is used. \n");
        printf("sort    :   sort = 1, the rows are sorted to balance the workload; \n");
        printf("            sort = 0, the rows are processed in order; \n");
        printf("function:   function = 1, the result of the kernel will be verified.\n");
        printf("            function = 0, the result verification is skipped\n");
        printf("sparse  :   sparse = 0, the dense version is executed as a baseline;\n");
        printf("            sparse = 1, the SpMM is executed;\n");
        printf("        :   sparse = 2, the Blocked Ell based SpMM is executed");
        printf("mixed   :   mixed = 0, use single precision; \n");
        printf("            mixed = 1, use half precision; \n");
    }
    // Run the benchmark
    else{
        std::string benchmark(argv[1]);
        int dimK = std::atoi(argv[2]);
        int vec_length = std::atoi(argv[3]);
        int kernel = std::atoi(argv[4]);
        int sorted = std::atoi(argv[5]);
        int func = std::atoi(argv[6]);
        int sparse = std::atoi(argv[7]);
        int mixed = std::atoi(argv[8]);
        std::cout << "Sparse matrix: " << benchmark << "\n" ;

        if (mixed) BmFN<half, half, short, half2, short2, CUDA_R_16F>(benchmark, dimK, vec_length, kernel, sorted, func, sparse);
        else BmFN<float, float, int, float, int, CUDA_R_32F>(benchmark, dimK, vec_length, kernel, sorted, func, sparse); 
    }
    printf("\n");
}
