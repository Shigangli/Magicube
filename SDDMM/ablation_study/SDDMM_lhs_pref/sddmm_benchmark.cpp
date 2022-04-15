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
//#include "sputnik/sputnik.h"
#include <cusparse.h>
#include <iostream>

int power2n(int n){
    int exp=1;
    for(int i=0; i < n; i++)
        exp *= 2;
    return exp;
}

double Host_sddmm_integers(int *lhs_matrix, int *rhs_matrix, int *ref_C, int M_GLOBAL, int K_GLOBAL, int N_GLOBAL, int preA, int preB, int vec_length, int *row_offsets, int *col_indices, int m_vec, int alignment){

    int maskA = (int)(power2n(preA)-1); //0b0000000011111111 for 8 bits
    int maskB = (int)(power2n(preB)-1); //0b0000000011111111 for 8 bits
    int a_tiles = 32 / preA;
    int b_tiles = 32 / preB;

    double flops = 0;
    // Loop over all the rows
    for (int i = 0; i < m_vec; i++){
        // Loop over all the nonzero columns of the column
        for (int j = row_offsets[i*2]; j < row_offsets[i*2+1]; j++){
            // Loop over all the values in the vector
            for (int v = 0; v < vec_length; v++){
                int accumulator = 0;
                int idx_m = i * vec_length + v;
                int idx_n = col_indices[j];
  
	        assert(a_tiles == b_tiles);
                for (int l=0; l<K_GLOBAL; l+=a_tiles){
		    int a_tile = lhs_matrix[idx_m*K_GLOBAL/a_tiles + l/a_tiles];
		    int b_tile = rhs_matrix[idx_n*K_GLOBAL/b_tiles + l/b_tiles];
                    for(int at=0; at < a_tiles; at++){
	            	int shift = at*preA;
                        int a_val = ((maskA << shift) & a_tile) >> shift;
                        int b_val = ((maskB << shift) & b_tile) >> shift;
			accumulator += (a_val*b_val); 
                        flops += 2.0;
	            }
                }
                // Write the output
                ref_C[(j/alignment)*alignment*vec_length + alignment*v + j%alignment] = accumulator;
            }
        }
    }
    return flops;
}

// For benchmarking, as a set of sparse matrices are provided
// The Dim M, N, and number of nonzeros are determined by the benchmark
void BmFN(std::string benchmark, int dimK, int vec_length, bool sorted, bool func, int sparse, int preA, int preB){
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

    int alignment = 8;

    printf("PreA: %d, PreB: %d, vec_len: %d, M: %d, M_vec: %d, N: %d, nnz: %d, K: %d\n", preA, preB, vec_length, m, m_vec, n, nonzeros, k);

    std::default_random_engine generator;

    if (sparse){
        // Host
        // Step 1: fetch the sparse matrix from benchmark file
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

        int *aligned_row_offsets = new int[m_vec*2];
	int aligned_num_item = 0;
	aligned_row_offsets[0] = aligned_num_item;
	for(int i = 1; i < m_vec + 1; i++){
	    int num_item = row_offsets[i] - row_offsets[i-1];
            //ceiling
	    aligned_num_item += (num_item + alignment - 1) / alignment * alignment;
	    if(i != m_vec)
	        aligned_row_offsets[i*2] = aligned_num_item;
	    aligned_row_offsets[i*2-1] = aligned_row_offsets[i*2-2] + num_item;
	}

	std::cout << " nonzero_vec: " << nonzeros_vec << " aligned_ nonzero_vec: " << aligned_num_item  << "\n" ;
        int *aligned_col_indices = new int[aligned_num_item];
	for(int i = 0; i < aligned_num_item; i++){
	    aligned_col_indices[i] = -1;
	}

	for(int i = 1; i < m_vec + 1; i++){
	    int offset_begin = row_offsets[i-1];
	    int offset_end = row_offsets[i];
	    for(int j = offset_begin; j < offset_end; j++)
	        aligned_col_indices[aligned_row_offsets[(i-1)*2] + j - offset_begin] = col_indices[j];
	}

	int *lhs_matrix;
	int *rhs_matrix;
        lhs_matrix = new int[m*k/(32/preA)];
        rhs_matrix = new int[n*k/(32/preB)];

        MakeDenseMatrix<int>(m, k/(32/preA), lhs_matrix, generator);
        MakeDenseMatrix<int>(n, k/(32/preB), rhs_matrix, generator);

        // Step 3: generate the output matrix
        int *h_output_values = new int[aligned_num_item*vec_length];
        int *output_values = new int[aligned_num_item*vec_length];
	for(int i = 0; i < aligned_num_item*vec_length; i++){
	    h_output_values[i] = 0;
	    output_values[i] = 0;
	}

        double flops = 0.0;
        if (func){
            // Step 4: Do the SDDMM on host
            flops = Host_sddmm_integers(lhs_matrix, rhs_matrix, h_output_values, m, k, n, preA, preB, vec_length, aligned_row_offsets, aligned_col_indices, m_vec, alignment);
	}

        // Device
        int *d_row_offsets, *d_col_indices, *d_row_indices;
        int *d_lhs_matrix, *d_rhs_matrix;
        int *d_output_values;

        checkCuda(cudaMalloc(&d_row_offsets, (m_vec*2)*sizeof(int)));
        checkCuda(cudaMalloc(&d_col_indices, aligned_num_item*sizeof(int)));
        checkCuda(cudaMalloc(&d_lhs_matrix, m*k*preA/8));
        checkCuda(cudaMalloc(&d_rhs_matrix, n*k*preB/8));
        checkCuda(cudaMalloc(&d_output_values, aligned_num_item*vec_length*sizeof(int)));
        checkCuda(cudaMalloc(&d_row_indices, m_vec * sizeof(int)));

        checkCuda(cudaMemcpy(d_row_offsets, aligned_row_offsets, (m_vec*2)*sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_col_indices, aligned_col_indices, aligned_num_item*sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_lhs_matrix, lhs_matrix, m*k*preA/8, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix, n*k*preB/8, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_output_values, output_values, aligned_num_item*vec_length*sizeof(int), cudaMemcpyHostToDevice));

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
        int NUM_PROFILES = 512;
        float sddmm_ms_avg = 0.0f;
        // TODO: Launch kernel
        if (preA == 4 && preB == 4){
            printf("Using WMMA \n");
            for(int iter=0; iter<32; ++iter){
                sddmm::wmmaSddmm_4b(m_vec, k, n, d_row_indices, d_row_offsets, d_col_indices, d_lhs_matrix, d_rhs_matrix, d_output_values, vec_length);
            }
            for(int iter=0; iter<NUM_PROFILES; ++iter){
                float sddmm_ms = 0.0f;
                cudaEvent_t sddmm_start;
                cudaEvent_t sddmm_end;
                cudaEventCreate(&sddmm_start);
                cudaEventCreate(&sddmm_end);
                cudaEventRecord(sddmm_start);
                sddmm::wmmaSddmm_4b(m_vec, k, n, d_row_indices, d_row_offsets, d_col_indices, d_lhs_matrix, d_rhs_matrix, d_output_values, vec_length);
                cudaEventRecord(sddmm_end);
                cudaEventSynchronize(sddmm_end);
                cudaEventElapsedTime(&sddmm_ms, sddmm_start, sddmm_end);
                cudaEventDestroy(sddmm_start);
                cudaEventDestroy(sddmm_end);
                sddmm_ms_avg += sddmm_ms;
            }
        }
	else if (preA == 8 && preB == 8){
            for(int iter=0; iter<32; ++iter){
                sddmm::wmmaSddmm_8b(m_vec, k, n, d_row_indices, d_row_offsets, d_col_indices, d_lhs_matrix, d_rhs_matrix, d_output_values, vec_length);
            }
            for(int iter=0; iter<NUM_PROFILES; ++iter){
                float sddmm_ms = 0.0f;
                cudaEvent_t sddmm_start;
                cudaEvent_t sddmm_end;
                cudaEventCreate(&sddmm_start);
                cudaEventCreate(&sddmm_end);
                cudaEventRecord(sddmm_start);
                sddmm::wmmaSddmm_8b(m_vec, k, n, d_row_indices, d_row_offsets, d_col_indices, d_lhs_matrix, d_rhs_matrix, d_output_values, vec_length);
                cudaEventRecord(sddmm_end);
                cudaEventSynchronize(sddmm_end);
                cudaEventElapsedTime(&sddmm_ms, sddmm_start, sddmm_end);
                cudaEventDestroy(sddmm_start);
                cudaEventDestroy(sddmm_end);
                sddmm_ms_avg += sddmm_ms;
            }
        }
	else if (preA == 16 && preB == 16){
            for(int iter=0; iter<32; ++iter){
                sddmm::wmmaSddmm_16b(m_vec, k, n, d_row_indices, d_row_offsets, d_col_indices, d_lhs_matrix, d_rhs_matrix, d_output_values, vec_length);
            }
            for(int iter=0; iter<NUM_PROFILES; ++iter){
                float sddmm_ms = 0.0f;
                cudaEvent_t sddmm_start;
                cudaEvent_t sddmm_end;
                cudaEventCreate(&sddmm_start);
                cudaEventCreate(&sddmm_end);
                cudaEventRecord(sddmm_start);
                sddmm::wmmaSddmm_16b(m_vec, k, n, d_row_indices, d_row_offsets, d_col_indices, d_lhs_matrix, d_rhs_matrix, d_output_values, vec_length);
                cudaEventRecord(sddmm_end);
                cudaEventSynchronize(sddmm_end);
                cudaEventElapsedTime(&sddmm_ms, sddmm_start, sddmm_end);
                cudaEventDestroy(sddmm_start);
                cudaEventDestroy(sddmm_end);
                sddmm_ms_avg += sddmm_ms;
            }
        }
        else{
            printf("unsupported kernel\n");
            // TODO: sputnik
        }

	flops = flops/1000.0/1000.0/1000.0;
        std::cout << "Runtime: " << sddmm_ms_avg/(float)NUM_PROFILES << " ms" << "\n";
        sddmm_ms_avg = sddmm_ms_avg/(float)NUM_PROFILES/1000.0;
        if (func){
            std::cout << "SDDMM lhs pref TOPS: " << flops/1000.0 << "  performance TOP/s: " << flops/sddmm_ms_avg/1000.0 << "\n";
	}

        cudaProfilerStop();

        if (func){
            // Copy the result back to host
            int *output_value_cuda = new int[aligned_num_item*vec_length];
            checkCuda(cudaMemcpy(output_value_cuda, d_output_values, aligned_num_item*vec_length*sizeof(int), cudaMemcpyDeviceToHost)); 
            
            // Verify the result
            int errors = 0;
            for (int j=0; j < aligned_num_item*vec_length; j++){
                if ((output_value_cuda[j] - h_output_values[j]) != 0){
		    //if(j<256)
                    //    printf("item %d, expect %d, got %d\n", j, h_output_values[j], output_value_cuda[j]);
                    errors ++;
                }
            }
            if (errors > 0) {
                printf("SDDMM does not agree with SEQUENTIAL! Total %d, %d errors!\n", aligned_num_item*vec_length, errors);
            }else {
                printf("SDDMM results verification: PASS\n");
            }
            delete output_value_cuda;
        }

        // Free the memory
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        cudaFree(d_row_indices);
        cudaFree(d_lhs_matrix);
        cudaFree(d_rhs_matrix);
        cudaFree(d_output_values);
        delete row_offsets;
        delete col_indices;
        delete aligned_row_offsets;
        delete aligned_col_indices;
        delete lhs_matrix;
        delete rhs_matrix;
        delete output_values;
        delete h_output_values;
        delete row_indices;
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
        printf("sort    :   sort = 1, the rows are sorted to balance the workload; \n");
        printf("            sort = 0, the rows are processed in order; \n");
        printf("function:   function = 1, the result of the kernel will be verified.\n");
        printf("            function = 0, the result verification is skipped\n");
        printf("sparse  :   sparse = 0, the dense version is executed as a baseline;\n");
    }
    else{
        std::string benchmark(argv[1]);
        int dimK = std::atoi(argv[2]);
        int vec_length = std::atoi(argv[3]);
        int sorted = std::atoi(argv[4]);
        int func = std::atoi(argv[5]);
        int sparse = std::atoi(argv[6]);
        int preA = std::atoi(argv[7]);
        int preB = std::atoi(argv[8]);

        std::cout << "Sparse matrix: " << benchmark << "\n" ;

	switch (vec_length){
            case 2:
                //printf("Vec_length: %d \n", vec_length);
                break;
            case 4:
                //printf("Vec_length: %d \n", vec_length);
                break;
            case 8:
                //printf("Vec_length: %d \n", vec_length);
                break;
            default:
                printf("Unsupported vec_length!\n");
        }

	if ((preA == 4) && (preB == 4)) BmFN(benchmark, dimK, vec_length, sorted, func, sparse, preA, preB);
	else if ((preA == 8) && (preB == 8)) BmFN(benchmark, dimK, vec_length, sorted, func, sparse, preA, preB);
	else if ((preA == 16) && (preB == 16)) BmFN(benchmark, dimK, vec_length, sorted, func, sparse, preA, preB);
	else printf("Unsupported precision!\n");
    }
    printf("\n");
}
