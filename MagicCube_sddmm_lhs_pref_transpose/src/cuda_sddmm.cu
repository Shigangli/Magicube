#include "cuda_fp16.h"
#include "../include/cuda_sddmm.cuh"
#include <cstdint>
#include <cmath>
#include <stdio.h>


namespace sddmm{

__device__ void print_val(int blockid, int threadid, float value){
    if (blockid == 0 && threadid == 0) printf("tid: %d, value is: %.8f\n", threadid, float(value));
}
    

constexpr __host__ __device__ __forceinline__ int Min(int a, int b) 
{
    return a < b ? a : b;
}

__device__ __forceinline__ void Swap(int i, int j, float* x) {
    float t = x[i];
    x[i] = x[j];
    x[j] = t;
}

__device__ __forceinline__ void Swap(int i, int j, half* x) {
    half t = x[i];
    x[i] = x[j];
    x[j] = t;
}

__host__ __device__ __forceinline__ int Log2(int x) {
    if (x >>= 1) return Log2(x) + 1;
    return 0;
}

/*
// The OutType is the output vector type. It should match the VecLenghth
template <typename LoadType, typename OutType, int Residual = false, int VecLength=4, int Tile_K=128, int Tile_X=32>
__global__ void cudaSddmmKernel(int m_vec, int k, int n, 
                    const int* __restrict__ row_indices,
                    const int* __restrict__ row_offsets,
                    const int* __restrict__ column_indices,
                    const half* __restrict__ lhs_matrix,
                    const half* __restrict__ rhs_matrix,
                    float* __restrict__ output_values)
{
    constexpr int kValuesPerLoad = sizeof(LoadType) / sizeof(half);
    constexpr int kValuesPerStore = sizeof(OutType) / sizeof(float);
    constexpr int VecLength_ = VecLength / kValuesPerStore;
    // Tile_Y = 1
    int m_index_vec = blockIdx.x;
    // Each thread block handles Tile_X entries
    int n_index = blockIdx.y * Tile_X;

    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset;

    // If this thread block has no nonzeros in the row to process, exit early
    if (n_index >= nonzeros) return;

    // Calculate the number of nonzeros that this thread block processes and
    // substract the x-dim thread index to simplify loop bounds checks
    nonzeros = Min(nonzeros - n_index, Tile_X);

    // register tile for the lhs dense matrix values
    // In each step, VecLength row 
    half lhs_fragment[Tile_K * VecLength / 32];

    // Shared memory tile for the output column indices
    __shared__ int column_indices_tile_array[Tile_X];

    int* column_indices_tile = column_indices_tile_array;

    // register file fragment for the rhs dense matrix values
    half rhs_fragment[Tile_X * Tile_K / 32];

    float accumulator_fragment[Tile_X * VecLength] = {};

    OutType output_fragment[Tile_X * VecLength_ / 32];
    float *output_fragment_ = reinterpret_cast<float *>(output_fragment);

    // float output_fragment[Tile_X * VecLength / 32];



    //
    // Begin kernel main loop
    //

    // Load the column indices for this n-dimension tile
    int nonzeros_ = nonzeros - threadIdx.x;
    const int* column_indices_ = column_indices + row_offset + n_index + threadIdx.x;
    int* column_indices_tile_ = column_indices_tile + threadIdx.x;
    constexpr int kColabItemsX = Tile_X / 32;
    #pragma unroll
    for (int x_item_idx = 0; x_item_idx < kColabItemsX; x_item_idx ++){
        if (nonzeros_ > 0) *column_indices_tile_ = k / kValuesPerLoad * __ldg(column_indices_);
        else *column_indices_tile_ = 0;
        nonzeros_ -= 32;
        column_indices_ += 32;
        column_indices_tile_ += 32;
    }

    int m_index = m_index_vec * VecLength;

    __syncthreads();

    constexpr int kThreadItemsK_ = Tile_K / 32 / kValuesPerLoad;
    const LoadType *lhs_matrix_ = reinterpret_cast<const LoadType *>(lhs_matrix + m_index * k) + threadIdx.x;
    LoadType *lhs_fragment_ = reinterpret_cast<LoadType *>(lhs_fragment);
    const LoadType *rhs_matrix_base = reinterpret_cast<const LoadType *>(rhs_matrix) + threadIdx.x;
    LoadType *rhs_fragment_ = reinterpret_cast<LoadType *>(rhs_fragment);

    const int k_ = k;
    #pragma nounroll
    for(; k >= Tile_K; k -= Tile_K){
        // Load a tile from the dense lhs matrix into register
        #pragma unroll
        for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
            #pragma unroll
            for (int v_idx = 0; v_idx < VecLength; v_idx ++){
                *(lhs_fragment_ + k_item_idx + v_idx * kThreadItemsK_) = __ldg(lhs_matrix_ + v_idx * k_ / kValuesPerLoad);
            }
            lhs_matrix_ += 32;
        }

        column_indices_tile_ = column_indices_tile;

        // Load a tile from the dense rhs matrix into register
        #pragma unroll
        for (int x_item_idx = 0; x_item_idx < Tile_X; x_item_idx ++){
            const LoadType *rhs_matrix_ = rhs_matrix_base + *column_indices_tile_;
            #pragma unroll
            for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
                int fragment_offset = x_item_idx * kThreadItemsK_ + k_item_idx;
                *(rhs_fragment_ + fragment_offset) = __ldg(rhs_matrix_);
                rhs_matrix_ += 32;
            }
            column_indices_tile_ ++;
        }
        rhs_matrix_base += Tile_K / kValuesPerLoad;

        // Multiply the tiles and accumulate the results
        #pragma unroll
        for (int k_item_idx = 0; k_item_idx < Tile_K / 32; k_item_idx++){
            // Load the lhs values
            half lhs_value[VecLength];
            #pragma unroll
            for (int v_idx = 0; v_idx < VecLength; v_idx ++){
                lhs_value[v_idx] = lhs_fragment[v_idx * Tile_K / 32 + k_item_idx];
            }
            // Do the computation
            #pragma unroll
            for (int x_item_idx = 0; x_item_idx < Tile_X; x_item_idx ++){
                const half rhs_value = rhs_fragment[k_item_idx + x_item_idx * Tile_K / 32];
                #pragma unroll
                for (int v_idx = 0; v_idx < VecLength; v_idx ++){
                    accumulator_fragment[x_item_idx + v_idx * Tile_X] += __half2float(lhs_value[v_idx] * rhs_value);
                }
            }
        }
    }

    // todo Residual
    if (Residual && k > 0){
        int residual = k - threadIdx.x * kValuesPerLoad;
        // Load a tile from the dense lhs matrix into register
        #pragma unroll
        for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
            if (residual > 0){
                *(lhs_fragment_ + k_item_idx) = __ldg(lhs_matrix_);
            }
            lhs_matrix_ += 32;
            residual -= 32 * kValuesPerLoad;
        }

        column_indices_tile_ = column_indices_tile;

        // Load a tile from the rhs matrix and compute immediately
        #pragma unroll
        for (int x_item_idx = 0; x_item_idx < Tile_X; x_item_idx ++){
            const LoadType *rhs_matrix_ = rhs_matrix_base + *column_indices_tile_;
            residual = k - threadIdx.x * kValuesPerLoad;
            #pragma unroll
            for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
                int fragment_offset = x_item_idx * kThreadItemsK_ + k_item_idx;
                if (residual > 0){
                    *(rhs_fragment_ + fragment_offset) = __ldg(rhs_matrix_);
                    for (int inner = 0; inner < kValuesPerLoad; inner ++){
                        half lhs_value = lhs_fragment[k_item_idx * kValuesPerLoad + inner];
                        half rhs_value = rhs_fragment[k_item_idx * kValuesPerLoad + x_item_idx * Tile_K / 32 + inner];
                        accumulator_fragment[x_item_idx] += __half2float(lhs_value * rhs_value);
                    }
                }
                rhs_matrix_ += 32;
                residual -= 32 * kValuesPerLoad;
            }
            column_indices_tile_ ++;
        }

    }

    // All reduce
    // Generate the thread mask
    constexpr int items_per_block = Tile_X / 32;
    uint32_t thread_mask = 0xffffffff;

    #pragma unroll
    for (int base_idx = 0; base_idx < Tile_X / 32; ++base_idx){
        #pragma unroll
        for (int k_item_idx = 1; k_item_idx < 32; k_item_idx *= 2){
            const int kBoundX = 32 / (k_item_idx * 2);
            #pragma unroll
            for (int x_item_idx = 0; x_item_idx < kBoundX; ++x_item_idx){
                // const int idx_a = x_item_idx * 2 * vec_length * k_item_idx;
                // const int idx_b = (x_item_idx * 2 + 1) * vec_length * k_item_idx;
                const int idx_a = x_item_idx * 2 * items_per_block * k_item_idx;
                const int idx_b = (x_item_idx * 2 + 1) * items_per_block * k_item_idx;
                const int kStep = Log2(k_item_idx);
                #pragma unroll
                for (int v_idx = 0; v_idx < VecLength; v_idx ++){
                    if ((threadIdx.x >> kStep) & 1) Swap(base_idx + idx_a + v_idx * Tile_X, base_idx + idx_b + v_idx * Tile_X, accumulator_fragment);
                    accumulator_fragment[base_idx + idx_a + v_idx * Tile_X] += __shfl_xor_sync(thread_mask, accumulator_fragment[base_idx + idx_b + v_idx * Tile_X], k_item_idx, 32);
                }
            }
        }
    }
    
    #pragma unroll
    for (int out_idx = 0; out_idx < Tile_X / 32; ++out_idx){
        #pragma unroll
        for (int v_idx = 0; v_idx < VecLength; v_idx ++){
            output_fragment_[out_idx * VecLength + v_idx] = accumulator_fragment[out_idx + v_idx * Tile_X];
        }
    }

    OutType *output_values_ = reinterpret_cast<OutType *>(output_values + (row_offset + n_index) * VecLength);
    
    #pragma unroll
    for (int i = 0; i < items_per_block; i ++){
        int i_ = i + threadIdx.x * items_per_block;
        if (i_ < nonzeros){
            #pragma unroll
            for (int v_idx = 0; v_idx < VecLength_; v_idx ++){
                output_values_[i_ * VecLength_ + v_idx] = output_fragment[i * VecLength_ + v_idx];
            }
        }
    }
}
*/

// The OutType is the output vector type. It should match the VecLenghth
template <typename LoadType, typename OutType, typename StoreType, int Residual = false, int VecLength=4, int Tile_K=128, int Tile_X=32>
__global__ void cudaSddmmKernel(int m_vec, int k, int n, 
                    const int* __restrict__ row_indices,
                    const int* __restrict__ row_offsets,
                    const int* __restrict__ column_indices,
                    const half* __restrict__ lhs_matrix,
                    const half* __restrict__ rhs_matrix,
                    OutType* __restrict__ output_values)
{
    constexpr int kValuesPerLoad = sizeof(LoadType) / sizeof(half);
    constexpr int kValuesPerStore = sizeof(StoreType) / sizeof(OutType);
    constexpr int VecLength_ = VecLength / kValuesPerStore;
    // Tile_Y = 1
    int m_index_vec = blockIdx.x;
    // Each thread block handles Tile_X entries
    int n_index = blockIdx.y * Tile_X;

    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset;

    // If this thread block has no nonzeros in the row to process, exit early
    if (n_index >= nonzeros) return;

    // Calculate the number of nonzeros that this thread block processes and
    // substract the x-dim thread index to simplify loop bounds checks
    nonzeros = Min(nonzeros - n_index, Tile_X);

    // register tile for the lhs dense matrix values
    // In each step, VecLength row 
    half lhs_fragment[Tile_K * VecLength / 32];

    // Shared memory tile for the output column indices
    __shared__ int column_indices_tile_array[Tile_X];

    int* column_indices_tile = column_indices_tile_array;

    // register file fragment for the rhs dense matrix values
    half rhs_fragment[Tile_X * Tile_K / 32];

    float accumulator_fragment[Tile_X * VecLength] = {};

    StoreType output_fragment[Tile_X * VecLength_ / 32];
    OutType *output_fragment_ = reinterpret_cast<OutType *>(output_fragment);

    // float output_fragment[Tile_X * VecLength / 32];



    //
    // Begin kernel main loop
    //

    // Load the column indices for this n-dimension tile
    int nonzeros_ = nonzeros - threadIdx.x;
    const int* column_indices_ = column_indices + row_offset + n_index + threadIdx.x;
    int* column_indices_tile_ = column_indices_tile + threadIdx.x;
    constexpr int kColabItemsX = Tile_X / 32;
    #pragma unroll
    for (int x_item_idx = 0; x_item_idx < kColabItemsX; x_item_idx ++){
        if (nonzeros_ > 0) *column_indices_tile_ = k / kValuesPerLoad * __ldg(column_indices_);
        else *column_indices_tile_ = 0;
        nonzeros_ -= 32;
        column_indices_ += 32;
        column_indices_tile_ += 32;
    }

    int m_index = m_index_vec * VecLength;

    __syncthreads();

    constexpr int kThreadItemsK_ = Tile_K / 32 / kValuesPerLoad;
    const LoadType *lhs_matrix_ = reinterpret_cast<const LoadType *>(lhs_matrix + m_index * k) + threadIdx.x;
    LoadType *lhs_fragment_ = reinterpret_cast<LoadType *>(lhs_fragment);
    const LoadType *rhs_matrix_base = reinterpret_cast<const LoadType *>(rhs_matrix) + threadIdx.x;
    LoadType *rhs_fragment_ = reinterpret_cast<LoadType *>(rhs_fragment);

    const int k_ = k;
    #pragma nounroll
    for(; k >= Tile_K; k -= Tile_K){
        // Load a tile from the dense lhs matrix into register
        #pragma unroll
        for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
            #pragma unroll
            for (int v_idx = 0; v_idx < VecLength; v_idx ++){
                *(lhs_fragment_ + k_item_idx + v_idx * kThreadItemsK_) = __ldg(lhs_matrix_ + v_idx * k_ / kValuesPerLoad);
            }
            lhs_matrix_ += 32;
        }

        column_indices_tile_ = column_indices_tile;

        // Load a tile from the dense rhs matrix into register
        #pragma unroll
        for (int x_item_idx = 0; x_item_idx < Tile_X; x_item_idx ++){
            const LoadType *rhs_matrix_ = rhs_matrix_base + *column_indices_tile_;
            #pragma unroll
            for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
                int fragment_offset = x_item_idx * kThreadItemsK_ + k_item_idx;
                *(rhs_fragment_ + fragment_offset) = __ldg(rhs_matrix_);
                rhs_matrix_ += 32;
            }
            column_indices_tile_ ++;
        }
        rhs_matrix_base += Tile_K / kValuesPerLoad;

        // Multiply the tiles and accumulate the results
        #pragma unroll
        for (int k_item_idx = 0; k_item_idx < Tile_K / 32; k_item_idx++){
            // Load the lhs values
            half lhs_value[VecLength];
            #pragma unroll
            for (int v_idx = 0; v_idx < VecLength; v_idx ++){
                lhs_value[v_idx] = lhs_fragment[v_idx * Tile_K / 32 + k_item_idx];
            }
            // Do the computation
            #pragma unroll
            for (int x_item_idx = 0; x_item_idx < Tile_X; x_item_idx ++){
                const half rhs_value = rhs_fragment[k_item_idx + x_item_idx * Tile_K / 32];
                #pragma unroll
                for (int v_idx = 0; v_idx < VecLength; v_idx ++){
                    accumulator_fragment[x_item_idx + v_idx * Tile_X] += __half2float(lhs_value[v_idx] * rhs_value);
                }
            }
        }
    }

    // todo Residual
    if (Residual && k > 0){
        int residual = k - threadIdx.x * kValuesPerLoad;
        // Load a tile from the dense lhs matrix into register
        #pragma unroll
        for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
            if (residual > 0){
                *(lhs_fragment_ + k_item_idx) = __ldg(lhs_matrix_);
            }
            lhs_matrix_ += 32;
            residual -= 32 * kValuesPerLoad;
        }

        column_indices_tile_ = column_indices_tile;

        // Load a tile from the rhs matrix and compute immediately
        #pragma unroll
        for (int x_item_idx = 0; x_item_idx < Tile_X; x_item_idx ++){
            const LoadType *rhs_matrix_ = rhs_matrix_base + *column_indices_tile_;
            residual = k - threadIdx.x * kValuesPerLoad;
            #pragma unroll
            for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
                int fragment_offset = x_item_idx * kThreadItemsK_ + k_item_idx;
                if (residual > 0){
                    *(rhs_fragment_ + fragment_offset) = __ldg(rhs_matrix_);
                    for (int inner = 0; inner < kValuesPerLoad; inner ++){
                        half lhs_value = lhs_fragment[k_item_idx * kValuesPerLoad + inner];
                        half rhs_value = rhs_fragment[k_item_idx * kValuesPerLoad + x_item_idx * Tile_K / 32 + inner];
                        accumulator_fragment[x_item_idx] += __half2float(lhs_value * rhs_value);
                    }
                }
                rhs_matrix_ += 32;
                residual -= 32 * kValuesPerLoad;
            }
            column_indices_tile_ ++;
        }

    }

    // All reduce
    // Generate the thread mask
    constexpr int items_per_block = Tile_X / 32;
    uint32_t thread_mask = 0xffffffff;

    #pragma unroll
    for (int base_idx = 0; base_idx < Tile_X / 32; ++base_idx){
        #pragma unroll
        for (int k_item_idx = 1; k_item_idx < 32; k_item_idx *= 2){
            const int kBoundX = 32 / (k_item_idx * 2);
            #pragma unroll
            for (int x_item_idx = 0; x_item_idx < kBoundX; ++x_item_idx){
                // const int idx_a = x_item_idx * 2 * vec_length * k_item_idx;
                // const int idx_b = (x_item_idx * 2 + 1) * vec_length * k_item_idx;
                const int idx_a = x_item_idx * 2 * items_per_block * k_item_idx;
                const int idx_b = (x_item_idx * 2 + 1) * items_per_block * k_item_idx;
                const int kStep = Log2(k_item_idx);
                #pragma unroll
                for (int v_idx = 0; v_idx < VecLength; v_idx ++){
                    if ((threadIdx.x >> kStep) & 1) Swap(base_idx + idx_a + v_idx * Tile_X, base_idx + idx_b + v_idx * Tile_X, accumulator_fragment);
                    accumulator_fragment[base_idx + idx_a + v_idx * Tile_X] += __shfl_xor_sync(thread_mask, accumulator_fragment[base_idx + idx_b + v_idx * Tile_X], k_item_idx, 32);
                }
            }
        }
    }
    
    #pragma unroll
    for (int out_idx = 0; out_idx < Tile_X / 32; ++out_idx){
        #pragma unroll
        for (int v_idx = 0; v_idx < VecLength; v_idx ++){
            output_fragment_[out_idx * VecLength + v_idx] = (OutType)accumulator_fragment[out_idx + v_idx * Tile_X];
        }
    }

    StoreType *output_values_ = reinterpret_cast<StoreType *>(output_values + (row_offset + n_index) * VecLength);
    
    #pragma unroll
    for (int i = 0; i < items_per_block; i ++){
        int i_ = i + threadIdx.x * items_per_block;
        if (i_ < nonzeros){
            #pragma unroll
            for (int v_idx = 0; v_idx < VecLength_; v_idx ++){
                output_values_[i_ * VecLength_ + v_idx] = output_fragment[i * VecLength_ + v_idx];
            }
        }
    }
}


// We implement a special kernel for Vector = 8 by having Tile_X=16 to avoid register spliling
template <typename LoadType, typename OutType, typename StoreType, int Residual = false, int VecLength=4, int Tile_K=128, int Tile_X=16>
__global__ void cudaSddmmKernel8(int m_vec, int k, int n, 
                    const int* __restrict__ row_indices,
                    const int* __restrict__ row_offsets,
                    const int* __restrict__ column_indices,
                    const half* __restrict__ lhs_matrix,
                    const half* __restrict__ rhs_matrix,
                    OutType* __restrict__ output_values)
{
    constexpr int kValuesPerLoad = sizeof(LoadType) / sizeof(half);
    constexpr int kValuesPerStore = sizeof(StoreType) / sizeof(OutType);
    constexpr int VecLength_ = VecLength / kValuesPerStore;
    // Tile_Y = 1
    int m_index_vec = blockIdx.x;
    // Each thread block handles Tile_X entries
    int n_index = blockIdx.y * Tile_X;

    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset;

    // If this thread block has no nonzeros in the row to process, exit early
    if (n_index >= nonzeros) return;

    // Calculate the number of nonzeros that this thread block processes and
    // substract the x-dim thread index to simplify loop bounds checks
    nonzeros = Min(nonzeros - n_index, Tile_X);

    // register tile for the lhs dense matrix values
    // In each step, VecLength row 
    half lhs_fragment[Tile_K * VecLength / 32];

    // Shared memory tile for the output column indices
    __shared__ int column_indices_tile_array[Tile_X];

    int* column_indices_tile = column_indices_tile_array;

    // register file fragment for the rhs dense matrix values
    half rhs_fragment[Tile_X * Tile_K / 32];

    float accumulator_fragment[Tile_X * VecLength] = {};

    StoreType output_fragment[VecLength_];
    OutType *output_fragment_ = reinterpret_cast<OutType *>(output_fragment);

    // float output_fragment[Tile_X * VecLength / 32];



    //
    // Begin kernel main loop
    //

    // Load the column indices for this n-dimension tile
    int nonzeros_ = nonzeros - threadIdx.x;
    const int* column_indices_ = column_indices + row_offset + n_index + threadIdx.x;
    int* column_indices_tile_ = column_indices_tile + threadIdx.x;
    constexpr int kColabItemsX = 1;//Tile_X / 32;
    #pragma unroll
    for (int x_item_idx = 0; x_item_idx < kColabItemsX; x_item_idx ++){
        if (nonzeros_ > 0) *column_indices_tile_ = k / kValuesPerLoad * __ldg(column_indices_);
        else *column_indices_tile_ = 0;
        nonzeros_ -= 32;
        column_indices_ += 32;
        column_indices_tile_ += 32;
    }

    int m_index = m_index_vec * VecLength;

    __syncthreads();

    constexpr int kThreadItemsK_ = Tile_K / 32 / kValuesPerLoad;
    const LoadType *lhs_matrix_ = reinterpret_cast<const LoadType *>(lhs_matrix + m_index * k) + threadIdx.x;
    LoadType *lhs_fragment_ = reinterpret_cast<LoadType *>(lhs_fragment);
    const LoadType *rhs_matrix_base = reinterpret_cast<const LoadType *>(rhs_matrix) + threadIdx.x;
    LoadType *rhs_fragment_ = reinterpret_cast<LoadType *>(rhs_fragment);

    const int k_ = k;
    #pragma nounroll
    for(; k >= Tile_K; k -= Tile_K){
        // Load a tile from the dense lhs matrix into register
        #pragma unroll
        for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
            #pragma unroll
            for (int v_idx = 0; v_idx < VecLength; v_idx ++){
                *(lhs_fragment_ + k_item_idx + v_idx * kThreadItemsK_) = __ldg(lhs_matrix_ + v_idx * k_ / kValuesPerLoad);
            }
            lhs_matrix_ += 32;
        }

        column_indices_tile_ = column_indices_tile;

        // Load a tile from the dense rhs matrix into register
        #pragma unroll
        for (int x_item_idx = 0; x_item_idx < Tile_X; x_item_idx ++){
            const LoadType *rhs_matrix_ = rhs_matrix_base + *column_indices_tile_;
            #pragma unroll
            for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
                int fragment_offset = x_item_idx * kThreadItemsK_ + k_item_idx;
                *(rhs_fragment_ + fragment_offset) = __ldg(rhs_matrix_);
                rhs_matrix_ += 32;
            }
            column_indices_tile_ ++;
        }
        rhs_matrix_base += Tile_K / kValuesPerLoad;

        // Multiply the tiles and accumulate the results
        #pragma unroll
        for (int k_item_idx = 0; k_item_idx < Tile_K / 32; k_item_idx++){
            // Load the lhs values
            half lhs_value[VecLength];
            #pragma unroll
            for (int v_idx = 0; v_idx < VecLength; v_idx ++){
                lhs_value[v_idx] = lhs_fragment[v_idx * Tile_K / 32 + k_item_idx];
            }
            // Do the computation
            #pragma unroll
            for (int x_item_idx = 0; x_item_idx < Tile_X; x_item_idx ++){
                const half rhs_value = rhs_fragment[k_item_idx + x_item_idx * Tile_K / 32];
                #pragma unroll
                for (int v_idx = 0; v_idx < VecLength; v_idx ++){
                    accumulator_fragment[x_item_idx + v_idx * Tile_X] += __half2float(lhs_value[v_idx] * rhs_value);
                }
            }
        }
    }

    // todo Residual
    if (Residual && k > 0){
        int residual = k - threadIdx.x * kValuesPerLoad;
        // Load a tile from the dense lhs matrix into register
        #pragma unroll
        for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
            if (residual > 0){
                *(lhs_fragment_ + k_item_idx) = __ldg(lhs_matrix_);
            }
            lhs_matrix_ += 32;
            residual -= 32 * kValuesPerLoad;
        }

        column_indices_tile_ = column_indices_tile;

        // Load a tile from the rhs matrix and compute immediately
        #pragma unroll
        for (int x_item_idx = 0; x_item_idx < Tile_X; x_item_idx ++){
            const LoadType *rhs_matrix_ = rhs_matrix_base + *column_indices_tile_;
            residual = k - threadIdx.x * kValuesPerLoad;
            #pragma unroll
            for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
                int fragment_offset = x_item_idx * kThreadItemsK_ + k_item_idx;
                if (residual > 0){
                    *(rhs_fragment_ + fragment_offset) = __ldg(rhs_matrix_);
                    for (int inner = 0; inner < kValuesPerLoad; inner ++){
                        half lhs_value = lhs_fragment[k_item_idx * kValuesPerLoad + inner];
                        half rhs_value = rhs_fragment[k_item_idx * kValuesPerLoad + x_item_idx * Tile_K / 32 + inner];
                        accumulator_fragment[x_item_idx] += __half2float(lhs_value * rhs_value);
                    }
                }
                rhs_matrix_ += 32;
                residual -= 32 * kValuesPerLoad;
            }
            column_indices_tile_ ++;
        }

    }

    // All reduce
    constexpr int Tile_X_d = Tile_X * 2;
    constexpr int VecLength_d = VecLength / 2;
    // Generate the thread mask
    constexpr int items_per_block = Tile_X_d / 32;
    uint32_t thread_mask = 0xffffffff;

    #pragma unroll
    for (int base_idx = 0; base_idx < Tile_X_d / 32; ++base_idx){
        #pragma unroll
        for (int k_item_idx = 1; k_item_idx < 32; k_item_idx *= 2){
            const int kBoundX = 32 / (k_item_idx * 2);
            #pragma unroll
            for (int x_item_idx = 0; x_item_idx < kBoundX; ++x_item_idx){
                // const int idx_a = x_item_idx * 2 * vec_length * k_item_idx;
                // const int idx_b = (x_item_idx * 2 + 1) * vec_length * k_item_idx;
                const int idx_a = x_item_idx * 2 * items_per_block * k_item_idx;
                const int idx_b = (x_item_idx * 2 + 1) * items_per_block * k_item_idx;
                const int kStep = Log2(k_item_idx);
                #pragma unroll
                for (int v_idx = 0; v_idx < VecLength_d; v_idx ++){
                    if ((threadIdx.x >> kStep) & 1) Swap(base_idx + idx_a + v_idx * Tile_X_d, base_idx + idx_b + v_idx * Tile_X_d, accumulator_fragment);
                    accumulator_fragment[base_idx + idx_a + v_idx * Tile_X_d] += __shfl_xor_sync(thread_mask, accumulator_fragment[base_idx + idx_b + v_idx * Tile_X_d], k_item_idx, 32);
                }
            }
        }
    }
    
    #pragma unroll
    for (int out_idx = 0; out_idx < Tile_X_d / 32; ++out_idx){
        #pragma unroll
        for (int v_idx = 0; v_idx < VecLength_d; v_idx ++){
            output_fragment_[out_idx * VecLength_d + v_idx * 2] = (OutType)(accumulator_fragment[out_idx + v_idx * Tile_X_d]);
        }
    }

    #pragma unroll
    for (int v_idx = 0; v_idx < VecLength_d; v_idx ++){
        output_fragment_[v_idx * 2 + 1] = __shfl_down_sync(0xffffffff, output_fragment_[v_idx * 2], 16);
    }

    StoreType *output_values_ = reinterpret_cast<StoreType *>(output_values + (row_offset + n_index) * VecLength);
    
    #pragma unroll
    for (int i = 0; i < items_per_block; i ++){
        int i_ = i + threadIdx.x * items_per_block;
        if (i_ < nonzeros){
            #pragma unroll
            for (int v_idx = 0; v_idx < VecLength_; v_idx ++){
                output_values_[i_ * VecLength_ + v_idx] = output_fragment[i * VecLength_ + v_idx];
            }
        }
    }
}


// We implement a special kernel for Vector = 4 by having Tile_X=16 to avoid register spliling
template <typename LoadType, typename OutType, typename StoreType, int Residual = false, int VecLength=4, int Tile_K=128, int Tile_X=16>
__global__ void cudaSddmmKernel4(int m_vec, int k, int n, 
                    const int* __restrict__ row_indices,
                    const int* __restrict__ row_offsets,
                    const int* __restrict__ column_indices,
                    const half* __restrict__ lhs_matrix,
                    const half* __restrict__ rhs_matrix,
                    OutType* __restrict__ output_values)
{
    constexpr int kValuesPerLoad = sizeof(LoadType) / sizeof(half);
    constexpr int kValuesPerStore = sizeof(StoreType) / sizeof(OutType);
    constexpr int VecLength_ = VecLength / kValuesPerStore;
    // Tile_Y = 1
    int m_index_vec = blockIdx.x;
    // Each thread block handles Tile_X entries
    int n_index = blockIdx.y * Tile_X;

    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset;

    // If this thread block has no nonzeros in the row to process, exit early
    if (n_index >= nonzeros) return;

    // Calculate the number of nonzeros that this thread block processes and
    // substract the x-dim thread index to simplify loop bounds checks
    nonzeros = Min(nonzeros - n_index, Tile_X);

    // register tile for the lhs dense matrix values
    // In each step, VecLength row 
    half lhs_fragment[Tile_K * VecLength / 32];

    // Shared memory tile for the output column indices
    __shared__ int column_indices_tile_array[Tile_X];

    int* column_indices_tile = column_indices_tile_array;

    // register file fragment for the rhs dense matrix values
    half rhs_fragment[Tile_X * Tile_K / 32];

    float accumulator_fragment[Tile_X * VecLength] = {};

    StoreType output_fragment[VecLength_];
    OutType *output_fragment_ = reinterpret_cast<OutType *>(output_fragment);

    // float output_fragment[Tile_X * VecLength / 32];



    //
    // Begin kernel main loop
    //

    // Load the column indices for this n-dimension tile
    int nonzeros_ = nonzeros - threadIdx.x;
    const int* column_indices_ = column_indices + row_offset + n_index + threadIdx.x;
    int* column_indices_tile_ = column_indices_tile + threadIdx.x;
    constexpr int kColabItemsX = 1;//Tile_X / 32;
    #pragma unroll
    for (int x_item_idx = 0; x_item_idx < kColabItemsX; x_item_idx ++){
        if (nonzeros_ > 0) *column_indices_tile_ = k / kValuesPerLoad * __ldg(column_indices_);
        else *column_indices_tile_ = 0;
        nonzeros_ -= 32;
        column_indices_ += 32;
        column_indices_tile_ += 32;
    }

    int m_index = m_index_vec * VecLength;

    __syncthreads();

    constexpr int kThreadItemsK_ = Tile_K / 32 / kValuesPerLoad;
    const LoadType *lhs_matrix_ = reinterpret_cast<const LoadType *>(lhs_matrix + m_index * k) + threadIdx.x;
    LoadType *lhs_fragment_ = reinterpret_cast<LoadType *>(lhs_fragment);
    const LoadType *rhs_matrix_base = reinterpret_cast<const LoadType *>(rhs_matrix) + threadIdx.x;
    LoadType *rhs_fragment_ = reinterpret_cast<LoadType *>(rhs_fragment);

    const int k_ = k;
    #pragma nounroll
    for(; k >= Tile_K; k -= Tile_K){
        // Load a tile from the dense lhs matrix into register
        #pragma unroll
        for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
            #pragma unroll
            for (int v_idx = 0; v_idx < VecLength; v_idx ++){
                *(lhs_fragment_ + k_item_idx + v_idx * kThreadItemsK_) = __ldg(lhs_matrix_ + v_idx * k_ / kValuesPerLoad);
            }
            lhs_matrix_ += 32;
        }

        column_indices_tile_ = column_indices_tile;

        // Load a tile from the dense rhs matrix into register
        #pragma unroll
        for (int x_item_idx = 0; x_item_idx < Tile_X; x_item_idx ++){
            const LoadType *rhs_matrix_ = rhs_matrix_base + *column_indices_tile_;
            #pragma unroll
            for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
                int fragment_offset = x_item_idx * kThreadItemsK_ + k_item_idx;
                *(rhs_fragment_ + fragment_offset) = __ldg(rhs_matrix_);
                rhs_matrix_ += 32;
            }
            column_indices_tile_ ++;
        }
        rhs_matrix_base += Tile_K / kValuesPerLoad;

        // Multiply the tiles and accumulate the results
        #pragma unroll
        for (int k_item_idx = 0; k_item_idx < Tile_K / 32; k_item_idx++){
            // Load the lhs values
            half lhs_value[VecLength];
            #pragma unroll
            for (int v_idx = 0; v_idx < VecLength; v_idx ++){
                lhs_value[v_idx] = lhs_fragment[v_idx * Tile_K / 32 + k_item_idx];
            }
            // Do the computation
            #pragma unroll
            for (int x_item_idx = 0; x_item_idx < Tile_X; x_item_idx ++){
                const half rhs_value = rhs_fragment[k_item_idx + x_item_idx * Tile_K / 32];
                #pragma unroll
                for (int v_idx = 0; v_idx < VecLength; v_idx ++){
                    accumulator_fragment[x_item_idx + v_idx * Tile_X] += __half2float(lhs_value[v_idx] * rhs_value);
                }
            }
        }
    }

    // todo Residual
    if (Residual && k > 0){
        int residual = k - threadIdx.x * kValuesPerLoad;
        // Load a tile from the dense lhs matrix into register
        #pragma unroll
        for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
            if (residual > 0){
                *(lhs_fragment_ + k_item_idx) = __ldg(lhs_matrix_);
            }
            lhs_matrix_ += 32;
            residual -= 32 * kValuesPerLoad;
        }

        column_indices_tile_ = column_indices_tile;

        // Load a tile from the rhs matrix and compute immediately
        #pragma unroll
        for (int x_item_idx = 0; x_item_idx < Tile_X; x_item_idx ++){
            const LoadType *rhs_matrix_ = rhs_matrix_base + *column_indices_tile_;
            residual = k - threadIdx.x * kValuesPerLoad;
            #pragma unroll
            for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
                int fragment_offset = x_item_idx * kThreadItemsK_ + k_item_idx;
                if (residual > 0){
                    *(rhs_fragment_ + fragment_offset) = __ldg(rhs_matrix_);
                    for (int inner = 0; inner < kValuesPerLoad; inner ++){
                        half lhs_value = lhs_fragment[k_item_idx * kValuesPerLoad + inner];
                        half rhs_value = rhs_fragment[k_item_idx * kValuesPerLoad + x_item_idx * Tile_K / 32 + inner];
                        accumulator_fragment[x_item_idx] += __half2float(lhs_value * rhs_value);
                    }
                }
                rhs_matrix_ += 32;
                residual -= 32 * kValuesPerLoad;
            }
            column_indices_tile_ ++;
        }

    }

    // All reduce
    constexpr int Tile_X_d = Tile_X * 2;
    constexpr int VecLength_d = VecLength / 2;
    // Generate the thread mask
    constexpr int items_per_block = Tile_X_d / 32;
    uint32_t thread_mask = 0xffffffff;

    #pragma unroll
    for (int base_idx = 0; base_idx < Tile_X_d / 32; ++base_idx){
        #pragma unroll
        for (int k_item_idx = 1; k_item_idx < 32; k_item_idx *= 2){
            const int kBoundX = 32 / (k_item_idx * 2);
            #pragma unroll
            for (int x_item_idx = 0; x_item_idx < kBoundX; ++x_item_idx){
                // const int idx_a = x_item_idx * 2 * vec_length * k_item_idx;
                // const int idx_b = (x_item_idx * 2 + 1) * vec_length * k_item_idx;
                const int idx_a = x_item_idx * 2 * items_per_block * k_item_idx;
                const int idx_b = (x_item_idx * 2 + 1) * items_per_block * k_item_idx;
                const int kStep = Log2(k_item_idx);
                #pragma unroll
                for (int v_idx = 0; v_idx < VecLength_d; v_idx ++){
                    if ((threadIdx.x >> kStep) & 1) Swap(base_idx + idx_a + v_idx * Tile_X_d, base_idx + idx_b + v_idx * Tile_X_d, accumulator_fragment);
                    accumulator_fragment[base_idx + idx_a + v_idx * Tile_X_d] += __shfl_xor_sync(thread_mask, accumulator_fragment[base_idx + idx_b + v_idx * Tile_X_d], k_item_idx, 32);
                }
            }
        }
    }
    
    #pragma unroll
    for (int out_idx = 0; out_idx < Tile_X_d / 32; ++out_idx){
        #pragma unroll
        for (int v_idx = 0; v_idx < VecLength_d; v_idx ++){
            output_fragment_[out_idx * VecLength_d + v_idx * 2] = (OutType)(accumulator_fragment[out_idx + v_idx * Tile_X_d]);
        }
    }

    #pragma unroll
    for (int v_idx = 0; v_idx < VecLength_d; v_idx ++){
        output_fragment_[v_idx * 2 + 1] = __shfl_down_sync(0xffffffff, output_fragment_[v_idx * 2], 16);
    }

    StoreType *output_values_ = reinterpret_cast<StoreType *>(output_values + (row_offset + n_index) * VecLength);
    
    #pragma unroll
    for (int i = 0; i < items_per_block; i ++){
        int i_ = i + threadIdx.x * items_per_block;
        if (i_ < nonzeros){
            #pragma unroll
            for (int v_idx = 0; v_idx < VecLength_; v_idx ++){
                output_values_[i_ * VecLength_ + v_idx] = output_fragment[i * VecLength_ + v_idx];
            }
        }
    }
}


/* This kernel is archived, as it is slower
// We implement a special kernel for Vector = 8 to avoid register spliling (With Tile_X = 8)
template <typename LoadType, typename OutType, typename StoreType, int Residual = false, int VecLength=4, int Tile_K=128, int Tile_X=8>
__global__ void cudaSddmmKernel8v2(int m_vec, int k, int n, 
                    const int* __restrict__ row_indices,
                    const int* __restrict__ row_offsets,
                    const int* __restrict__ column_indices,
                    const half* __restrict__ lhs_matrix,
                    const half* __restrict__ rhs_matrix,
                    OutType* __restrict__ output_values)
{
    constexpr int kValuesPerLoad = sizeof(LoadType) / sizeof(half);
    constexpr int kValuesPerStore = sizeof(StoreType) / sizeof(OutType);
    constexpr int VecLength_ = VecLength / kValuesPerStore;
    // Tile_Y = 1
    int m_index_vec = blockIdx.x;
    // Each thread block handles Tile_X entries
    int n_index = blockIdx.y * Tile_X;

    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset;

    // If this thread block has no nonzeros in the row to process, exit early
    if (n_index >= nonzeros) return;

    // Calculate the number of nonzeros that this thread block processes and
    // substract the x-dim thread index to simplify loop bounds checks
    nonzeros = Min(nonzeros - n_index, Tile_X);

    // register tile for the lhs dense matrix values
    // In each step, VecLength row 
    half lhs_fragment[Tile_K * VecLength / 32];

    // Shared memory tile for the output column indices
    __shared__ int column_indices_tile_array[Tile_X];

    int* column_indices_tile = column_indices_tile_array;

    // register file fragment for the rhs dense matrix values
    half rhs_fragment[Tile_X * Tile_K / 32];

    float accumulator_fragment[Tile_X * VecLength] = {};

    StoreType output_fragment[VecLength_];
    OutType *output_fragment_ = reinterpret_cast<OutType *>(output_fragment);

    // float output_fragment[Tile_X * VecLength / 32];



    //
    // Begin kernel main loop
    //

    // Load the column indices for this n-dimension tile
    int nonzeros_ = nonzeros - threadIdx.x;
    const int* column_indices_ = column_indices + row_offset + n_index + threadIdx.x;
    int* column_indices_tile_ = column_indices_tile + threadIdx.x;
    constexpr int kColabItemsX = 1;//Tile_X / 32;
    #pragma unroll
    for (int x_item_idx = 0; x_item_idx < kColabItemsX; x_item_idx ++){
        if (nonzeros_ > 0) *column_indices_tile_ = k / kValuesPerLoad * __ldg(column_indices_);
        else *column_indices_tile_ = 0;
        nonzeros_ -= 32;
        column_indices_ += 32;
        column_indices_tile_ += 32;
    }

    int m_index = m_index_vec * VecLength;

    __syncthreads();

    constexpr int kThreadItemsK_ = Tile_K / 32 / kValuesPerLoad;
    const LoadType *lhs_matrix_ = reinterpret_cast<const LoadType *>(lhs_matrix + m_index * k) + threadIdx.x;
    LoadType *lhs_fragment_ = reinterpret_cast<LoadType *>(lhs_fragment);
    const LoadType *rhs_matrix_base = reinterpret_cast<const LoadType *>(rhs_matrix) + threadIdx.x;
    LoadType *rhs_fragment_ = reinterpret_cast<LoadType *>(rhs_fragment);

    const int k_ = k;
    #pragma nounroll
    for(; k >= Tile_K; k -= Tile_K){
        // Load a tile from the dense lhs matrix into register
        #pragma unroll
        for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
            #pragma unroll
            for (int v_idx = 0; v_idx < VecLength; v_idx ++){
                *(lhs_fragment_ + k_item_idx + v_idx * kThreadItemsK_) = __ldg(lhs_matrix_ + v_idx * k_ / kValuesPerLoad);
            }
            lhs_matrix_ += 32;
        }

        column_indices_tile_ = column_indices_tile;

        // Load a tile from the dense rhs matrix into register
        #pragma unroll
        for (int x_item_idx = 0; x_item_idx < Tile_X; x_item_idx ++){
            const LoadType *rhs_matrix_ = rhs_matrix_base + *column_indices_tile_;
            #pragma unroll
            for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
                int fragment_offset = x_item_idx * kThreadItemsK_ + k_item_idx;
                *(rhs_fragment_ + fragment_offset) = __ldg(rhs_matrix_);
                rhs_matrix_ += 32;
            }
            column_indices_tile_ ++;
        }
        rhs_matrix_base += Tile_K / kValuesPerLoad;

        // Multiply the tiles and accumulate the results
        #pragma unroll
        for (int k_item_idx = 0; k_item_idx < Tile_K / 32; k_item_idx++){
            // Load the lhs values
            half lhs_value[VecLength];
            #pragma unroll
            for (int v_idx = 0; v_idx < VecLength; v_idx ++){
                lhs_value[v_idx] = lhs_fragment[v_idx * Tile_K / 32 + k_item_idx];
            }
            // Do the computation
            #pragma unroll
            for (int x_item_idx = 0; x_item_idx < Tile_X; x_item_idx ++){
                const half rhs_value = rhs_fragment[k_item_idx + x_item_idx * Tile_K / 32];
                #pragma unroll
                for (int v_idx = 0; v_idx < VecLength; v_idx ++){
                    accumulator_fragment[x_item_idx + v_idx * Tile_X] += __half2float(lhs_value[v_idx] * rhs_value);
                }
            }
        }
    }

    // todo Residual
    if (Residual && k > 0){
        int residual = k - threadIdx.x * kValuesPerLoad;
        // Load a tile from the dense lhs matrix into register
        #pragma unroll
        for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
            if (residual > 0){
                *(lhs_fragment_ + k_item_idx) = __ldg(lhs_matrix_);
            }
            lhs_matrix_ += 32;
            residual -= 32 * kValuesPerLoad;
        }

        column_indices_tile_ = column_indices_tile;

        // Load a tile from the rhs matrix and compute immediately
        #pragma unroll
        for (int x_item_idx = 0; x_item_idx < Tile_X; x_item_idx ++){
            const LoadType *rhs_matrix_ = rhs_matrix_base + *column_indices_tile_;
            residual = k - threadIdx.x * kValuesPerLoad;
            #pragma unroll
            for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
                int fragment_offset = x_item_idx * kThreadItemsK_ + k_item_idx;
                if (residual > 0){
                    *(rhs_fragment_ + fragment_offset) = __ldg(rhs_matrix_);
                    for (int inner = 0; inner < kValuesPerLoad; inner ++){
                        half lhs_value = lhs_fragment[k_item_idx * kValuesPerLoad + inner];
                        half rhs_value = rhs_fragment[k_item_idx * kValuesPerLoad + x_item_idx * Tile_K / 32 + inner];
                        accumulator_fragment[x_item_idx] += __half2float(lhs_value * rhs_value);
                    }
                }
                rhs_matrix_ += 32;
                residual -= 32 * kValuesPerLoad;
            }
            column_indices_tile_ ++;
        }

    }

    // All reduce
    constexpr int Tile_X_d = Tile_X * 4;
    constexpr int VecLength_d = VecLength / 4;
    // Generate the thread mask
    constexpr int items_per_block = Tile_X_d / 32;
    uint32_t thread_mask = 0xffffffff;

    #pragma unroll
    for (int base_idx = 0; base_idx < Tile_X_d / 32; ++base_idx){
        #pragma unroll
        for (int k_item_idx = 1; k_item_idx < 32; k_item_idx *= 2){
            const int kBoundX = 32 / (k_item_idx * 2);
            #pragma unroll
            for (int x_item_idx = 0; x_item_idx < kBoundX; ++x_item_idx){
                // const int idx_a = x_item_idx * 2 * vec_length * k_item_idx;
                // const int idx_b = (x_item_idx * 2 + 1) * vec_length * k_item_idx;
                const int idx_a = x_item_idx * 2 * items_per_block * k_item_idx;
                const int idx_b = (x_item_idx * 2 + 1) * items_per_block * k_item_idx;
                const int kStep = Log2(k_item_idx);
                #pragma unroll
                for (int v_idx = 0; v_idx < VecLength_d; v_idx ++){
                    if ((threadIdx.x >> kStep) & 1) Swap(base_idx + idx_a + v_idx * Tile_X_d, base_idx + idx_b + v_idx * Tile_X_d, accumulator_fragment);
                    accumulator_fragment[base_idx + idx_a + v_idx * Tile_X_d] += __shfl_xor_sync(thread_mask, accumulator_fragment[base_idx + idx_b + v_idx * Tile_X_d], k_item_idx, 32);
                }
            }
        }
    }
    
    #pragma unroll
    for (int out_idx = 0; out_idx < Tile_X_d / 32; ++out_idx){
        #pragma unroll
        for (int v_idx = 0; v_idx < VecLength_d; v_idx ++){
            output_fragment_[out_idx * VecLength_d + v_idx * 4] = (OutType)(accumulator_fragment[out_idx + v_idx * Tile_X_d]);
        }
    }

    #pragma unroll
    for (int v_idx = 0; v_idx < VecLength_d; v_idx ++){
        output_fragment_[v_idx * 4 + 1] = __shfl_down_sync(0xffffffff, output_fragment_[v_idx * 4], 8);
        output_fragment_[v_idx * 4 + 2] = __shfl_down_sync(0xffffffff, output_fragment_[v_idx * 4], 16);
        output_fragment_[v_idx * 4 + 3] = __shfl_down_sync(0xffffffff, output_fragment_[v_idx * 4], 24);
    }

    StoreType *output_values_ = reinterpret_cast<StoreType *>(output_values + (row_offset + n_index) * VecLength);
    
    #pragma unroll
    for (int i = 0; i < items_per_block; i ++){
        int i_ = i + threadIdx.x * items_per_block;
        if (i_ < nonzeros){
            #pragma unroll
            for (int v_idx = 0; v_idx < VecLength_; v_idx ++){
                output_values_[i_ * VecLength_ + v_idx] = output_fragment[i * VecLength_ + v_idx];
            }
        }
    }
}
*/


template <typename LoadType, int Residual, int kBlockItemsY=1, int kBlockItemsK=128,
          int kBlockItemsX=32, int kBlockWidth=32>
cudaError_t cudaSddmmEx(
    int m, int k, int n, int nonzeros, const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets, const int* __restrict__ column_indices,
    const half* __restrict__ lhs_matrix, const half* __restrict__ rhs_matrix,
    float* __restrict__ output_values, int vec_length, cudaStream_t stream) 
{
    dim3 grid_dim(std::ceil(static_cast<float>(m) / kBlockItemsY), std::ceil(static_cast<float>(n) / kBlockItemsX), 1);
    dim3 block_dim(kBlockWidth, kBlockItemsY, 1);
    
    switch(vec_length){
        case 1: 
            cudaSddmmKernel<LoadType, float, float, Residual, 1, 128><<<grid_dim, block_dim>>>(
                m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
            break;
        case 2:
            cudaSddmmKernel<LoadType, float, float2, Residual, 2, 128><<<grid_dim, block_dim>>>(
                m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
            break;
        case 4:
            cudaSddmmKernel<LoadType, float, float4, Residual, 4, 128><<<grid_dim, block_dim>>>(
                m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
            break;
        case 8:
            cudaSddmmKernel<LoadType, float, float4, Residual, 8, 128><<<grid_dim, block_dim>>>(
                m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
            break;
        default:
            printf("Unsupported Vector Length!\n");

    }
  return cudaGetLastError();
}


template <typename LoadType, int Residual, int kBlockItemsY=1, int kBlockItemsK=128,
          int kBlockItemsX=32, int kBlockWidth=32>
cudaError_t cudaSddmmEx(
    int m, int k, int n, int nonzeros, const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets, const int* __restrict__ column_indices,
    const half* __restrict__ lhs_matrix, const half* __restrict__ rhs_matrix,
    half* __restrict__ output_values, int vec_length, cudaStream_t stream) 
{   

    dim3* grid_dim;
    dim3* block_dim;
    switch(vec_length){
        case 1: 
            grid_dim = new dim3(std::ceil(static_cast<float>(m) / kBlockItemsY), std::ceil(static_cast<float>(n) / kBlockItemsX), 1);
            block_dim = new dim3(kBlockWidth, kBlockItemsY, 1);
            cudaSddmmKernel<LoadType, half, half, Residual, 1, 64><<<*grid_dim, *block_dim>>>(
                m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
            break;
        case 2:
            grid_dim = new dim3(std::ceil(static_cast<float>(m) / kBlockItemsY), std::ceil(static_cast<float>(n) / kBlockItemsX), 1);
            block_dim = new dim3(kBlockWidth, kBlockItemsY, 1);
            cudaSddmmKernel<LoadType, half, half2, Residual, 2, 64><<<*grid_dim, *block_dim>>>(
                m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
            break;
        case 4:
            grid_dim = new dim3(std::ceil(static_cast<float>(m) / kBlockItemsY), std::ceil(static_cast<float>(n) / kBlockItemsX), 1);
            block_dim = new dim3(kBlockWidth, kBlockItemsY, 1);
            cudaSddmmKernel<LoadType, half, float2, Residual, 4, 64><<<*grid_dim, *block_dim>>>(
                m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
            
            // With Tile_X = 16. It is slower due to the low data reuse rate
            // grid_dim = new dim3(std::ceil(static_cast<float>(m) / kBlockItemsY), std::ceil(static_cast<float>(n) / 16), 1);
            // block_dim = new dim3(kBlockWidth, kBlockItemsY, 1);
            // cudaSddmmKernel4<LoadType, half, float2, Residual, 4, 128, 16><<<*grid_dim, *block_dim>>>(
            //     m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
            break;
        case 8:
            // With Tile_X = 32: 1 ms, very slow, due to the register split
            // grid_dim = new dim3(std::ceil(static_cast<float>(m) / 1), std::ceil(static_cast<float>(n) / 32), 1);
            // block_dim = new dim3(kBlockWidth, kBlockItemsY, 1);
            // cudaSddmmKernel<LoadType, half, float4, Residual, 8, 128, 32><<<*grid_dim, *block_dim>>>(
            //    m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
            
            // With Tile_X = 16, 0.2252 ms, the best we have for now
            grid_dim = new dim3(std::ceil(static_cast<float>(m) / 1), std::ceil(static_cast<float>(n) / 16), 1);
            block_dim = new dim3(kBlockWidth, kBlockItemsY, 1);
            cudaSddmmKernel8<float, half, float4, Residual, 8, 64, 16><<<*grid_dim, *block_dim>>>(
                m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
            
            // With Tile_X = 8, 0.2393 ms, more memory access, slower
            // grid_dim = new dim3(std::ceil(static_cast<float>(m) / 1), std::ceil(static_cast<float>(n) / 8), 1);
            // block_dim = new dim3(kBlockWidth, kBlockItemsY, 1);
            // cudaSddmmKernel8v2<float2, half, float4, Residual, 8, 128, 8><<<*grid_dim, *block_dim>>>(
            //    m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
            break;
        default:
            printf("Unsupported Vector Length!\n");

    }
  return cudaGetLastError();
}


cudaError_t cudaSddmm(int m_vec, int k, int n, int nonzeros_vec,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const half* __restrict__ lhs_matrix,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_values, 
    int vec_length, cudaStream_t stream) {
    
    if ((k % 64) == 0){
        return cudaSddmmEx<float, false>(
            m_vec, k, n, nonzeros_vec, row_indices, row_offsets, col_indices,
            lhs_matrix, rhs_matrix, output_values, vec_length, stream);
    }
    else{
        if ((k % 8 == 0)){
            return cudaSddmmEx<float2, true>(
                m_vec, k, n, nonzeros_vec, row_indices, row_offsets, col_indices,
                lhs_matrix, rhs_matrix, output_values, vec_length, stream);
        }
        else if ((k % 4 == 0)){
            return cudaSddmmEx<float2, true>(
                m_vec, k, n, nonzeros_vec, row_indices, row_offsets, col_indices,
                lhs_matrix, rhs_matrix, output_values, vec_length, stream);
        }
        else if ((k % 2 == 0)){
            return cudaSddmmEx<half2, true>(
                m_vec, k, n, nonzeros_vec, row_indices, row_offsets, col_indices,
                rhs_matrix, lhs_matrix, output_values, vec_length, stream);
        }
        else {
            return cudaSddmmEx<half, true>(
                m_vec, k, n, nonzeros_vec, row_indices, row_offsets, col_indices,
                rhs_matrix, lhs_matrix, output_values, vec_length, stream);
        }
    }
}

cudaError_t cudaSddmm(int m_vec, int k, int n, int nonzeros_vec,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const half* __restrict__ lhs_matrix,
    const half* __restrict__ rhs_matrix,
    half* __restrict__ output_values, 
    int vec_length, cudaStream_t stream) {
    
    if ((k % 64) == 0){
        return cudaSddmmEx<float, false>(
            m_vec, k, n, nonzeros_vec, row_indices, row_offsets, col_indices,
            lhs_matrix, rhs_matrix, output_values, vec_length, stream);
    }
    else{
        if ((k % 8 == 0)){
            return cudaSddmmEx<float2, true>(
                m_vec, k, n, nonzeros_vec, row_indices, row_offsets, col_indices,
                lhs_matrix, rhs_matrix, output_values, vec_length, stream);
        }
        else if ((k % 4 == 0)){
            return cudaSddmmEx<float2, true>(
                m_vec, k, n, nonzeros_vec, row_indices, row_offsets, col_indices,
                lhs_matrix, rhs_matrix, output_values, vec_length, stream);
        }
        else if ((k % 2 == 0)){
            return cudaSddmmEx<half2, true>(
                m_vec, k, n, nonzeros_vec, row_indices, row_offsets, col_indices,
                rhs_matrix, lhs_matrix, output_values, vec_length, stream);
        }
        else {
            return cudaSddmmEx<half, true>(
                m_vec, k, n, nonzeros_vec, row_indices, row_offsets, col_indices,
                rhs_matrix, lhs_matrix, output_values, vec_length, stream);
        }
    }
}

cudaError_t cudaSddmm(int m_vec, int k, int n, int nonzeros_vec,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const float* __restrict__ lhs_matrix,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_values, 
    int vec_length, cudaStream_t stream)
{
    printf("Doesn't support single precision\n");
    return cudaSuccess;
}

}