#ifndef BM_TEST_UTILS_H
#define BM_TEST_UTILS_H
#include <algorithm>
#include <cstring>
#include <random>
#include <stdio.h>
#include "cuda_fp16.h"
#include <assert.h>
#include <cublas_v2.h>

inline
cudaError_t checkCuda(cudaError_t result){
    if (result != cudaSuccess){
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

inline 
cublasStatus_t checkCublas(cublasStatus_t result){
    if (result != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cublasGetErrorString(result));
        assert(result == CUBLAS_STATUS_SUCCESS);
    }
    return result;
}


// Helper function that generates random column indices for Blocked ELL format
void GenerateUniformBlockedELLIndex(int num_block_col, 
    int num_block_nnz_col, int num_block_row, 
    int* ellColInd, std::default_random_engine generator)
{
    std::uniform_int_distribution<int> uni(0, num_block_col - 1);
    std::vector<int> indices(num_block_col);
    std::iota(indices.begin(), indices.end(), 0);
    // Loop through all the rows of the sparse matrix
    for (int r = 0; r < num_block_row; r ++){
        int* ellColInd_r = reinterpret_cast<int *>(ellColInd + r * num_block_nnz_col);
        // fill the vector with 0 ~ num_block_col-1 in sequence
        std::random_shuffle(indices.begin(), indices.end());
        int offset = 0;
        for (int c = 0; c < num_block_col; c++){
            if (indices[c] < num_block_nnz_col){
                ellColInd_r[offset] = c;
                offset ++;
            }
        }
    }
}


template <typename ValueType, typename IndexType>
void MakeSparseMatrixRandomUniform(int rows, int columns, int nonzeros,
                                   ValueType *values, IndexType *row_offsets,
                                   IndexType *column_indices,
                                   std::default_random_engine generator,
                                   int row_padding) 
{
    int64_t num_elements = static_cast<int64_t>(rows) * columns;
    assert (nonzeros <= num_elements);
    assert (nonzeros > 0);
    assert (row_padding >= 0);

    std::uniform_real_distribution<ValueType> distribution(-1.0, 1.0);
    
    // generate random values for the matrix
    std::vector<ValueType> nonzero_values(nonzeros);
    for (auto &v : nonzero_values){
        v = distribution(generator);
    }

    // Create a uniformly distributed random sparsity mask
    std::vector<int64_t> indices(num_elements);
    // fill the vector with 0 ~ num_elements-1 in sequence
    std::iota(indices.begin(), indices.end(), 0);
    // shuffle the vector
    std::random_shuffle(indices.begin(), indices.end());

    // Create the compressed sparse row indices and offsets
    int64_t offset = 0;
    row_offsets[0] = 0;
    for (int64_t i = 0; i < rows; ++i){
        for (int64_t j = 0; j < columns; ++j){
            int64_t idx = i * columns + j;
            if (indices[idx] < nonzeros){
                values[offset] = nonzero_values[indices[idx]];
                column_indices[offset] = j;
                ++offset;
            }
        }

        if (row_padding > 0){
            int residue = (offset - row_offsets[i]) % row_padding;
            int to_add = (row_padding - residue) % row_padding;
            for (; to_add > 0; --to_add){
                values[offset] = 0.0;
                // NOTE: When we pad with zeros the column index that we assign
                // the phantom zero needs to be a valid column index s.t. we
                // don't index out-of-range into the dense rhs matrix when
                // computing spmm. Here we set all padding column-offsets to
                // the same column as the final non-padding weight in the row.
                column_indices[offset] = column_indices[offset - 1];
                ++offset;
            }
            assert((offset - row_offsets[i]) % row_padding == 0);
        }
        row_offsets[i + 1] = offset;
    }
}


template <typename ValueType>
void MakeDenseMatrix(int rows, int columns, ValueType *matrix,
                     std::default_random_engine generator)
{
    std::uniform_real_distribution<float> distribution(-1.0, 1.0);
    
    for (int64_t i = 0; i < static_cast<int64_t>(rows) * columns; ++i){
        float temp = distribution(generator);
        matrix[i] = ValueType(temp);
        // int temp = (i / columns) % 8;
        // matrix[i] = half(temp * 0.01);
    }
}

/*
// For CSC
template <typename ValueType, typename OutType>
void Host_sddmm(int m_vec, int k, int VecLength, const int* col_offsets, const int* row_indices, const ValueType* lhs_matrix,
                const ValueType* rhs_matrix, OutType* output_values){
    // Loop over all the columns
    for (int i = 0; i < m_vec; ++i){
        // Loop over all the nonzero rows of the column
        for (int j = col_offsets[i]; j < col_offsets[i+1]; ++j){
            // Loop over all the values in the vector
            for (int v = 0; v < VecLength; v ++){
                // set the accumulator
                OutType accumulator = 0.0;
                // compute the index to the real m and n
                int idx_n = row_indices[j];
                int idx_m = i * VecLength + v;
                for (int l=0; l < k; ++l){
                    accumulator += (OutType)lhs_matrix[idx_n * k + l] * (OutType)rhs_matrix[idx_m * k + l];
                }
                // Write the output
                output_values[j * VecLength + v] = accumulator;
            }
        }
    }
}
*/

template <typename ValueType, typename OutType>
void Host_sddmm(int m_vec, int k, int VecLength, const int* row_offsets, const int* col_indices, const ValueType* lhs_matrix,
                const ValueType* rhs_matrix, OutType* output_values){
    // Loop over all the rows
    for (int i = 0; i < m_vec; ++i){
        // Loop over all the nonzero columns of the column
        for (int j = row_offsets[i]; j < row_offsets[i+1]; ++j){
            // Loop over all the values in the vector
            for (int v = 0; v < VecLength; v ++){
                // set the accumulator
                float accumulator = 0.0;
                // compute the index to the real m and n
                int idx_n = col_indices[j];
                int idx_m = i * VecLength + v;
                for (int l=0; l < k; ++l){
                    accumulator += (float)lhs_matrix[idx_m * k + l] * (float)rhs_matrix[idx_n * k + l];
                }
                // Write the output
                output_values[j * VecLength + v] = (OutType)accumulator;
            }
        }
    }
}


void IdentityRowSwizzle(int rows, int *row_indices){
    std::iota(row_indices, row_indices + rows, 0);
}

void SortedRowSwizzle(int rows, const int *row_offsets, int *row_indices){
    // Create the unsorted row indices
    std::vector<int> swizzle_staging(rows);
    std::iota(swizzle_staging.begin(), swizzle_staging.end(), 0);

    // Argsort the row indices based on their length
    std::sort(swizzle_staging.begin(), swizzle_staging.end(),
        [&row_offsets](int idx_a, int idx_b){
            int length_a = row_offsets[idx_a + 1] - row_offsets[idx_a];
            int length_b = row_offsets[idx_b + 1] - row_offsets[idx_b];
            return length_a > length_b;
        });
    
    // Copy the ordered row indices to the output.
    std::memcpy(row_indices, swizzle_staging.data(), sizeof(int) * rows);
}

#endif