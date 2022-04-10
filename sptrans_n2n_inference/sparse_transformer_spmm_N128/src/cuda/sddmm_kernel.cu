#include <cuda.h>
#include "cuda_fp16.h"
#include <mma.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

using namespace nvcuda;


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



// In this version, we try to use the mma to avoid shared memory access and sync barriers
// The inverted thread group is solved by using additional registers to hold the inversed operands

template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=8, int Tile_X=32, int Tile_K=64>
__device__ void mmaSddmmKernel8reg_(int m_vec, int k, int n, 
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ lhs_matrix,
    const half* __restrict__ rhs_matrix,
    OutType* __restrict__ output_values)
{
    // const int tid = threadIdx.x;
    // const int bid = blockIdx.x + gridDim.x * blockIdx.y;

    // 
    // Constant values computed during compilation
    // 

    // The length of the vector type
    constexpr int kValuesPerLoad = 8;
    // The number of column indices loaded by each thread
    constexpr int kColabItemsX = Tile_X / 32;

    constexpr int mixed = sizeof(OutType) / sizeof(half);

    //
    // Storage
    //

    // Shared memory that stores the column indices
    __shared__ int column_indices_tile_array[Tile_X];
    // Registers that stores the lhs_fragment
    // There are Tile_K * VecLenghth elements in the lhs tile
    float4 lhs_fragment[2];
    float4 rhs_fragment[8];

    float acc_frag[4][2][8] = {0};

    //
    // Step 1: Determine the workload of the current thread block
    //

    int m_index_vec = blockIdx.x;
    int n_index = blockIdx.y * Tile_X;

    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);
    int m_index = m_index_vec * VecLength;

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset;

    // If this thread block has no nonzeros in the row to process, exit early
    if (n_index >= nonzeros) return;

    // Calculate the number of nonzeros that this thread block processes
    nonzeros = min(nonzeros - n_index, Tile_X);

    //
    // Step 2: Load the column indices to the shared memory
    //
    

    int* column_indices_tile = column_indices_tile_array;
    int nonzeros_ = nonzeros - threadIdx.x;
    // The pointer that points to the input column_indices array
    const int* column_indices_ = column_indices + row_offset + n_index + threadIdx.x;
    // The pointer that points to the shared memory <column_indices_tile_array>
    int* column_indices_tile_ = column_indices_tile + threadIdx.x;

    // Load the column indices
    #pragma unroll
    for (int x_item_idx = 0; x_item_idx < kColabItemsX; x_item_idx ++){
        if (nonzeros_ > 0) *column_indices_tile_ = k / kValuesPerLoad * __ldg(column_indices_);
        else *column_indices_tile_ = 0;
        nonzeros_ -= 32;
        column_indices_ += 32;
        column_indices_tile_ += 32;
    }

    __syncthreads();

    //
    // Step 3: Begin kernel main loop
    //

    const int thread_group_id = threadIdx.x / 4;
    const int lane_id = threadIdx.x % 4;
    const int high_group = threadIdx.x / 16;
    const int octet_id = thread_group_id % 4;

    // Declare the pointers to the lhs and rhs matrices
    const float4 *lhs_matrix_low = reinterpret_cast<const float4 *>(lhs_matrix + (m_index + lane_id) * k) + thread_group_id;
    const float4 *lhs_matrix_high = reinterpret_cast<const float4 *>(lhs_matrix + (m_index + lane_id + 4) * k) + (thread_group_id + 4) % 8;

    const float4 *rhs_matrix_base_low = reinterpret_cast<const float4 *>(rhs_matrix) + thread_group_id;
    const float4 *rhs_matrix_base_high = reinterpret_cast<const float4 *>(rhs_matrix) + (thread_group_id + 4) % 8;

    // Declare the pointers to the lhs and rhs fragments in int.
    int *lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
    int *rhs_fragment_int = reinterpret_cast<int *>(rhs_fragment);

    // half *lhs_fragment_half = reinterpret_cast<half *>(lhs_fragment);
    // half *rhs_fragment_half = reinterpret_cast<half *>(rhs_fragment);

    #pragma nounroll
    for (; k >= Tile_K; k -= Tile_K){
        // Load a tile from the dense lhs matrix into the register
        lhs_fragment[0] = __ldg(lhs_matrix_low);
        lhs_fragment[1] = __ldg(lhs_matrix_high);

        if (high_group){
            float4 temp = lhs_fragment[1];
            lhs_fragment[1] = lhs_fragment[0];
            lhs_fragment[0] = temp;
        }

        lhs_matrix_low += 8;
        lhs_matrix_high += 8;

        column_indices_tile_ = column_indices_tile + lane_id;

        // Load a tile from the dense rhs matrix into the register
        #pragma unroll
        for (int x_item_idx = 0; x_item_idx < 4; x_item_idx ++){
            const float4 *rhs_matrix_ = rhs_matrix_base_low + *column_indices_tile_;
            rhs_fragment[x_item_idx * 2] = __ldg(rhs_matrix_);
            column_indices_tile_ += 4;

            rhs_matrix_ = rhs_matrix_base_high + *column_indices_tile_;
            rhs_fragment[x_item_idx * 2 + 1] = __ldg(rhs_matrix_);
            column_indices_tile_ += 4;

            if (high_group){
                float4 temp = rhs_fragment[x_item_idx * 2 + 1];
                rhs_fragment[x_item_idx * 2 + 1] = rhs_fragment[x_item_idx * 2];
                rhs_fragment[x_item_idx * 2] = temp;
            }

            asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%10, %11}, \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                "+f"(acc_frag[x_item_idx][0][0]), "+f"(acc_frag[x_item_idx][0][1]), 
                "+f"(acc_frag[x_item_idx][0][2]), "+f"(acc_frag[x_item_idx][0][3]), 
                "+f"(acc_frag[x_item_idx][0][4]), "+f"(acc_frag[x_item_idx][0][5]), 
                "+f"(acc_frag[x_item_idx][0][6]), "+f"(acc_frag[x_item_idx][0][7]):
                "r"(rhs_fragment_int[x_item_idx * 8 + 0]), "r"(rhs_fragment_int[x_item_idx * 8 + 1]),
                "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
            );

            asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%10, %11}, \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                "+f"(acc_frag[x_item_idx][0][0]), "+f"(acc_frag[x_item_idx][0][1]), 
                "+f"(acc_frag[x_item_idx][0][2]), "+f"(acc_frag[x_item_idx][0][3]), 
                "+f"(acc_frag[x_item_idx][0][4]), "+f"(acc_frag[x_item_idx][0][5]), 
                "+f"(acc_frag[x_item_idx][0][6]), "+f"(acc_frag[x_item_idx][0][7]):
                "r"(rhs_fragment_int[x_item_idx * 8 + 2]), "r"(rhs_fragment_int[x_item_idx * 8 + 3]),
                "r"(lhs_fragment_int[2]), "r"(lhs_fragment_int[3])
            );

            asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%10, %11}, \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                "+f"(acc_frag[x_item_idx][1][0]), "+f"(acc_frag[x_item_idx][1][1]), 
                "+f"(acc_frag[x_item_idx][1][2]), "+f"(acc_frag[x_item_idx][1][3]), 
                "+f"(acc_frag[x_item_idx][1][4]), "+f"(acc_frag[x_item_idx][1][5]), 
                "+f"(acc_frag[x_item_idx][1][6]), "+f"(acc_frag[x_item_idx][1][7]):
                "r"(rhs_fragment_int[x_item_idx * 8 + 4]), "r"(rhs_fragment_int[x_item_idx * 8 + 5]),
                "r"(lhs_fragment_int[4]), "r"(lhs_fragment_int[5])
            );

            asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%10, %11}, \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                "+f"(acc_frag[x_item_idx][1][0]), "+f"(acc_frag[x_item_idx][1][1]), 
                "+f"(acc_frag[x_item_idx][1][2]), "+f"(acc_frag[x_item_idx][1][3]), 
                "+f"(acc_frag[x_item_idx][1][4]), "+f"(acc_frag[x_item_idx][1][5]), 
                "+f"(acc_frag[x_item_idx][1][6]), "+f"(acc_frag[x_item_idx][1][7]):
                "r"(rhs_fragment_int[x_item_idx * 8 + 6]), "r"(rhs_fragment_int[x_item_idx * 8 + 7]),
                "r"(lhs_fragment_int[6]), "r"(lhs_fragment_int[7])
            );  
        }

        rhs_matrix_base_low += 8;
        rhs_matrix_base_high += 8;
    }

    // TODO Residual

    // Accumulate partial sum

    int src_line = (threadIdx.x + 16) % 32;
    #pragma unroll
    for (int x_item_idx = 0; x_item_idx < 4; x_item_idx ++){
        #pragma unroll
        for (int acc_idx = 0; acc_idx < 8; acc_idx ++){
            acc_frag[x_item_idx][0][acc_idx] += __shfl_sync(0xffffffff, acc_frag[x_item_idx][1][(acc_idx + 4) % 8], src_line, 32);
        }
    }

    #pragma unroll
    for (int acc_idx = 0; acc_idx < 8; acc_idx ++){
        acc_frag[0][0][acc_idx] += __shfl_down_sync(0xffffffff, acc_frag[0][0][acc_idx], 8);
        acc_frag[1][0][acc_idx] += __shfl_down_sync(0xffffffff, acc_frag[1][0][acc_idx], 8);
        acc_frag[2][0][acc_idx] += __shfl_up_sync(0xffffffff, acc_frag[2][0][acc_idx], 8);
        acc_frag[3][0][acc_idx] += __shfl_up_sync(0xffffffff, acc_frag[3][0][acc_idx], 8);

        acc_frag[0][0][acc_idx] += __shfl_down_sync(0xffffffff, acc_frag[0][0][acc_idx], 4);
        acc_frag[1][0][acc_idx] += __shfl_up_sync(0xffffffff, acc_frag[1][0][acc_idx], 4);
        acc_frag[2][0][acc_idx] += __shfl_down_sync(0xffffffff, acc_frag[2][0][acc_idx], 4);
        acc_frag[3][0][acc_idx] += __shfl_up_sync(0xffffffff, acc_frag[3][0][acc_idx], 4);
    }

    #pragma unroll
    for (int acc_idx = 0; acc_idx < 8; acc_idx ++){
        switch(octet_id){
            case 0: break;
            case 1: acc_frag[0][0][acc_idx] = acc_frag[1][0][acc_idx]; break;
            case 2: acc_frag[0][0][acc_idx] = acc_frag[2][0][acc_idx]; break;
            case 3: acc_frag[0][0][acc_idx] = acc_frag[3][0][acc_idx]; break;
        }
    }

    // Convert the output to half if necessary
    if (mixed == 1){
        OutType *acc_frag_half = reinterpret_cast<OutType *>(acc_frag);
        #pragma unroll
        for (int i = 0; i < 8; i++){
            acc_frag_half[i] = __float2half(acc_frag[0][0][i]);
        }
    }

    StoreType *acc_frag_float2 = reinterpret_cast<StoreType *>(acc_frag);
    
    int out_col_id = octet_id * 8 + high_group * 4 + lane_id % 2;

    StoreType *output_values_ = reinterpret_cast<StoreType *>(output_values + (row_offset + n_index + out_col_id) * VecLength) + lane_id / 2;

    if (out_col_id < nonzeros){
        *(output_values_) = *(acc_frag_float2);
        *(output_values_ + 2) = *(acc_frag_float2 + 2);
    }
    
    if (out_col_id + 2 < nonzeros){
        *(output_values_ + 8) = *(acc_frag_float2 + 1);
        *(output_values_ + 10) = *(acc_frag_float2 + 3);
    }
}


template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=8, int Tile_X=32, int Tile_K=64>
__global__ void mmaSddmmKernel8reg(int m_vec, int k, int n,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ lhs_matrix,
    const half* __restrict__ rhs_matrix,
    OutType* __restrict__ output_values)
{
    mmaSddmmKernel8reg_<LoadType, OutType, StoreType, Residual, VecLength, Tile_X, Tile_K>(
        m_vec, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values
    );
}


template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=8, int Tile_X=32, int Tile_K=64>
__global__ void batchedMmaSddmmKernel8reg(int m_vec, int k, int n,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ lhs_matrix_b,
    int lhs_stride,
    const half* __restrict__ rhs_matrix_b,
    int rhs_stride,
    OutType* __restrict__ output_values_b,
    int output_stride)
{
    // Get the entry index
    int entry_idx = blockIdx.z;
    const half* lhs_matrix = lhs_matrix_b + entry_idx * lhs_stride;
    const half* rhs_matrix = rhs_matrix_b + entry_idx * rhs_stride;
    OutType* output_values = output_values_b + output_stride;

    mmaSddmmKernel8reg_<LoadType, OutType, StoreType, Residual, VecLength, Tile_X, Tile_K>(
        m_vec, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values
    );
}


template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=4, int Tile_X=32, int Tile_K=64>
__device__ void mmaSddmmKernel4reg_(int m_vec, int k, int n, 
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ lhs_matrix,
    const half* __restrict__ rhs_matrix,
    OutType* __restrict__ output_values)
{
    //
    // Constant values computed during compilation
    //
    constexpr int kValuesPerLoad = 8;
    constexpr int kColabItemsX = Tile_X / 32;

    constexpr int mixed = sizeof(OutType) / sizeof(half);
    
    //
    // Storage
    //

    // Shared memory tile for the output column indices
    __shared__ int column_indices_tile_array[Tile_X];
    // Registers that stores the lhs_fragment
    float4 lhs_fragment[1];
    float4 rhs_fragment[8];

    float acc_frag[4][8] = {0};
    float acc_waste[4];

    //
    // Step 1: Determine the workload of the current thread block
    //

    int m_index_vec = blockIdx.x;
    int n_index = blockIdx.y * Tile_X;

    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);
    int m_index = m_index_vec * VecLength;

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset;

    // If this thread block has no nonzeros in the row to process, exit early
    if (n_index >= nonzeros) return;

    // Calculate the number of nonzeros that this thread block processes
    nonzeros = min(nonzeros - n_index, Tile_X);

    //
    // Step 2: Load the column indices to the shared memory
    //
    

    int* column_indices_tile = column_indices_tile_array;
    int nonzeros_ = nonzeros - threadIdx.x;
    const int* column_indices_ = column_indices + row_offset + n_index + threadIdx.x;
    int* column_indices_tile_ = column_indices_tile + threadIdx.x;
    #pragma unroll
    for (int x_item_idx = 0; x_item_idx < kColabItemsX; x_item_idx ++){
        if (nonzeros_ > 0) *column_indices_tile_ = k / kValuesPerLoad * __ldg(column_indices_);
        else *column_indices_tile_ = 0;
        nonzeros_ -= 32;
        column_indices_ += 32;
        column_indices_tile_ += 32;
    }

    __syncthreads();

    //
    // Step 3: Begin kernel main loop
    //

    const int thread_group_id = threadIdx.x / 4;
    const int lane_id = threadIdx.x % 4;
    const int high_group = threadIdx.x / 16;
    const int octet_id = thread_group_id % 4;

    // Declare the pointers to the lhs and rhs matrices
    const float4 *lhs_matrix_low = reinterpret_cast<const float4 *>(lhs_matrix + (m_index + lane_id) * k) + thread_group_id;

    const float4 *rhs_matrix_base_low = reinterpret_cast<const float4 *>(rhs_matrix) + thread_group_id;
    const float4 *rhs_matrix_base_high = reinterpret_cast<const float4 *>(rhs_matrix) + (thread_group_id + 4) % 8;

    // Declare the pointers to the lhs and rhs fragments in int
    int *lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
    int *rhs_fragment_int = reinterpret_cast<int *>(rhs_fragment);

    #pragma nounroll
    for(; k >= Tile_K; k-= Tile_K){
        // Load a tile from the dense lhs matrix into the register
        lhs_fragment[0] = __ldg(lhs_matrix_low);

        lhs_matrix_low += 8;

        column_indices_tile_ = column_indices_tile + lane_id;

        // Load a tile from the dense rhs matrix into the register
        #pragma unroll
        for (int x_item_idx = 0; x_item_idx < 4; x_item_idx ++){
            const float4 *rhs_matrix_ = rhs_matrix_base_low + *column_indices_tile_;
            rhs_fragment[x_item_idx * 2] = __ldg(rhs_matrix_);
            column_indices_tile_ += 4;

            rhs_matrix_ = rhs_matrix_base_high + *column_indices_tile_;
            rhs_fragment[x_item_idx * 2 + 1] = __ldg(rhs_matrix_);
            column_indices_tile_ += 4;

            if (high_group){
                float4 temp = rhs_fragment[x_item_idx * 2 + 1];
                rhs_fragment[x_item_idx * 2 + 1] = rhs_fragment[x_item_idx * 2];
                rhs_fragment[x_item_idx * 2] = temp;
            }

            asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%10, %11}, \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                "+f"(acc_frag[x_item_idx][0]), "+f"(acc_frag[x_item_idx][1]), 
                "+f"(acc_frag[x_item_idx][2]), "+f"(acc_frag[x_item_idx][3]), 
                "+f"(acc_waste[0]), "+f"(acc_waste[1]), 
                "+f"(acc_waste[2]), "+f"(acc_waste[3]):
                "r"(rhs_fragment_int[x_item_idx * 8 + 0]), "r"(rhs_fragment_int[x_item_idx * 8 + 1]),
                "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
            );

            asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \t"
                "{%4, %5, %6, %7, %0, %1, %2, %3}, \t"
                "{%8, %9}, \t"
                "{%10, %11}, \t"
                "{%4, %5, %6, %7, %0, %1, %2, %3}; ":
                "+f"(acc_frag[x_item_idx][4]), "+f"(acc_frag[x_item_idx][5]), 
                "+f"(acc_frag[x_item_idx][6]), "+f"(acc_frag[x_item_idx][7]), 
                "+f"(acc_waste[0]), "+f"(acc_waste[1]), 
                "+f"(acc_waste[2]), "+f"(acc_waste[3]):
                "r"(rhs_fragment_int[x_item_idx * 8 + 4]), "r"(rhs_fragment_int[x_item_idx * 8 + 5]),
                "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
            );

            asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%10, %11}, \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                "+f"(acc_frag[x_item_idx][0]), "+f"(acc_frag[x_item_idx][1]), 
                "+f"(acc_frag[x_item_idx][2]), "+f"(acc_frag[x_item_idx][3]), 
                "+f"(acc_waste[0]), "+f"(acc_waste[1]), 
                "+f"(acc_waste[2]), "+f"(acc_waste[3]):
                "r"(rhs_fragment_int[x_item_idx * 8 + 2]), "r"(rhs_fragment_int[x_item_idx * 8 + 3]),
                "r"(lhs_fragment_int[2]), "r"(lhs_fragment_int[3])
            );

            asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \t"
                "{%4, %5, %6, %7, %0, %1, %2, %3}, \t"
                "{%8, %9}, \t"
                "{%10, %11}, \t"
                "{%4, %5, %6, %7, %0, %1, %2, %3}; ":
                "+f"(acc_frag[x_item_idx][4]), "+f"(acc_frag[x_item_idx][5]), 
                "+f"(acc_frag[x_item_idx][6]), "+f"(acc_frag[x_item_idx][7]), 
                "+f"(acc_waste[0]), "+f"(acc_waste[1]), 
                "+f"(acc_waste[2]), "+f"(acc_waste[3]):
                "r"(rhs_fragment_int[x_item_idx * 8 + 6]), "r"(rhs_fragment_int[x_item_idx * 8 + 7]),
                "r"(lhs_fragment_int[2]), "r"(lhs_fragment_int[3])
            );
        }

        rhs_matrix_base_low += 8;
        rhs_matrix_base_high += 8;
    }

    // TODO Residual

    // Accumulate partial sum
    int src_line = (threadIdx.x + 16) % 32;
    #pragma unroll
    for (int x_item_idx = 0; x_item_idx < 4; x_item_idx ++){
        #pragma unroll
        for (int acc_idx = 0; acc_idx < 4; acc_idx ++){
            acc_frag[x_item_idx][acc_idx] += __shfl_sync(0xffffffff, acc_frag[x_item_idx][(acc_idx + 4) % 8], src_line, 32);
        }
    }

    #pragma unroll
    for (int acc_idx = 0; acc_idx < 4; acc_idx ++){
        acc_frag[0][acc_idx] += __shfl_down_sync(0xffffffff, acc_frag[0][acc_idx], 8);
        acc_frag[1][acc_idx] += __shfl_down_sync(0xffffffff, acc_frag[1][acc_idx], 8);
        acc_frag[2][acc_idx] += __shfl_up_sync(0xffffffff, acc_frag[2][acc_idx], 8);
        acc_frag[3][acc_idx] += __shfl_up_sync(0xffffffff, acc_frag[3][acc_idx], 8);

        acc_frag[0][acc_idx] += __shfl_down_sync(0xffffffff, acc_frag[0][acc_idx], 4);
        acc_frag[1][acc_idx] += __shfl_up_sync(0xffffffff, acc_frag[1][acc_idx], 4);
        acc_frag[2][acc_idx] += __shfl_down_sync(0xffffffff, acc_frag[2][acc_idx], 4);
        acc_frag[3][acc_idx] += __shfl_up_sync(0xffffffff, acc_frag[3][acc_idx], 4);
    }

    #pragma unroll
    for (int acc_idx = 0; acc_idx < 4; acc_idx ++){
        switch(octet_id){
            case 0: break;
            case 1: acc_frag[0][acc_idx] = acc_frag[1][acc_idx]; break;
            case 2: acc_frag[0][acc_idx] = acc_frag[2][acc_idx]; break;
            case 3: acc_frag[0][acc_idx] = acc_frag[3][acc_idx]; break;
        }
    }

    // Switch the registers between threads to achieve coalesced write back

    int src_lane = octet_id * 4 + high_group * 16 + (lane_id + 2) % 4;

    #pragma unroll
    for (int acc_idx = 0; acc_idx < 2; acc_idx ++){
        float reg_share;
        if (lane_id < 2) reg_share = acc_frag[0][acc_idx + 2];
        else reg_share = acc_frag[0][acc_idx];
        float temp = __shfl_sync(0xffffffff, reg_share, src_lane, 32);
        if (lane_id < 2) acc_frag[0][acc_idx + 2] = temp;
        else acc_frag[0][acc_idx] = temp;
    }

    // Convert the output to half if necessary
    if (mixed == 1){
        OutType *acc_frag_half = reinterpret_cast<OutType *>(acc_frag);
        #pragma unroll
        for (int i = 0; i < 4; i++){
            acc_frag_half[i] = __float2half(acc_frag[0][i]);
        }
    }

    StoreType *acc_frag_float4 = reinterpret_cast<StoreType *>(acc_frag);

    int out_col_id = octet_id * 8 + high_group * 4 + lane_id;

    StoreType *output_values_ = reinterpret_cast<StoreType *>(output_values + (row_offset + n_index + out_col_id) * VecLength);

    if (out_col_id < nonzeros){
        *(output_values_) = *(acc_frag_float4);
    }
}


template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=4, int Tile_X=32, int Tile_K=64>
__global__ void mmaSddmmKernel4reg(int m_vec, int k, int n, 
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ lhs_matrix,
    const half* __restrict__ rhs_matrix,
    OutType* __restrict__ output_values)
{
    mmaSddmmKernel4reg_<LoadType, OutType, StoreType, Residual, VecLength, Tile_X, Tile_K>(
        m_vec, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values
    );
}


template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=4, int Tile_X=32, int Tile_K=64>
__global__ void batchedMmaSddmmKernel4reg(int m_vec, int k, int n, 
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ lhs_matrix_b,
    int lhs_stride,
    const half* __restrict__ rhs_matrix_b,
    int rhs_stride,
    OutType* __restrict__ output_values_b,
    int output_stride)
{
    // Get the entry index
    int entry_idx = blockIdx.z;
    const half* lhs_matrix = lhs_matrix_b + entry_idx * lhs_stride;
    const half* rhs_matrix = rhs_matrix_b + entry_idx * rhs_stride;
    OutType* output_values = output_values_b + entry_idx * output_stride;
    mmaSddmmKernel4reg_<LoadType, OutType, StoreType, Residual, VecLength, Tile_X, Tile_K>(
        m_vec, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values
    );
}


template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=2, int Tile_X=32, int Tile_K=64>
__device__ void mmaSddmmKernel2reg_(int m_vec, int k, int n, 
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ lhs_matrix,
    const half* __restrict__ rhs_matrix,
    OutType* __restrict__ output_values)
{
    //
    // Constant values computed during compilation
    //
    constexpr int kValuesPerLoad = 8;
    constexpr int kColabItemsX = Tile_X / 32;
    constexpr int mixed = sizeof(OutType) / sizeof(half);
    
    //
    // Storage
    //

    // Shared memory tile for the output column indices
    __shared__ int column_indices_tile_array[Tile_X];
    // Registers that stores the lhs_fragment
    float4 lhs_fragment[1];
    float4 rhs_fragment[8];

    float acc_frag[4][8] = {0};
    float acc_waste[4];

    //
    // Step 1: Determine the workload of the current thread block
    //

    int m_index_vec = blockIdx.x;
    int n_index = blockIdx.y * Tile_X;

    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);
    int m_index = m_index_vec * VecLength;

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset;

    // If this thread block has no nonzeros in the row to process, exit early
    if (n_index >= nonzeros) return;

    // Calculate the number of nonzeros that this thread block processes
    nonzeros = min(nonzeros - n_index, Tile_X);

    //
    // Step 2: Load the column indices to the shared memory
    //
    

    int* column_indices_tile = column_indices_tile_array;
    int nonzeros_ = nonzeros - threadIdx.x;
    const int* column_indices_ = column_indices + row_offset + n_index + threadIdx.x;
    int* column_indices_tile_ = column_indices_tile + threadIdx.x;
    #pragma unroll
    for (int x_item_idx = 0; x_item_idx < kColabItemsX; x_item_idx ++){
        if (nonzeros_ > 0) *column_indices_tile_ = k / kValuesPerLoad * __ldg(column_indices_);
        else *column_indices_tile_ = 0;
        nonzeros_ -= 32;
        column_indices_ += 32;
        column_indices_tile_ += 32;
    }

    __syncthreads();

    //
    // Step 3: Begin kernel main loop
    //

    const int thread_group_id = threadIdx.x / 4;
    const int lane_id = threadIdx.x % 4;
    const int high_group = threadIdx.x / 16;
    const int octet_id = thread_group_id % 4;

    // Declare the pointers to the lhs and rhs matrices
    const float4 *lhs_matrix_low = reinterpret_cast<const float4 *>(lhs_matrix + (m_index + lane_id) * k) + thread_group_id;

    const float4 *rhs_matrix_base_low = reinterpret_cast<const float4 *>(rhs_matrix) + thread_group_id;
    const float4 *rhs_matrix_base_high = reinterpret_cast<const float4 *>(rhs_matrix) + (thread_group_id + 4) % 8;

    // Declare the pointers to the lhs and rhs fragments in int
    int *lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
    int *rhs_fragment_int = reinterpret_cast<int *>(rhs_fragment);

    #pragma nounroll
    for(; k >= Tile_K; k-= Tile_K){
        // Load a tile from the dense lhs matrix into the register
        if (lane_id < 2) lhs_fragment[0] = __ldg(lhs_matrix_low);

        lhs_matrix_low += 8;

        column_indices_tile_ = column_indices_tile + lane_id;

        // Load a tile from the dense rhs matrix into the register
        #pragma unroll
        for (int x_item_idx = 0; x_item_idx < 4; x_item_idx ++){
            const float4 *rhs_matrix_ = rhs_matrix_base_low + *column_indices_tile_;
            rhs_fragment[x_item_idx * 2] = __ldg(rhs_matrix_);
            column_indices_tile_ += 4;

            rhs_matrix_ = rhs_matrix_base_high + *column_indices_tile_;
            rhs_fragment[x_item_idx * 2 + 1] = __ldg(rhs_matrix_);
            column_indices_tile_ += 4;

            if (high_group){
                float4 temp = rhs_fragment[x_item_idx * 2 + 1];
                rhs_fragment[x_item_idx * 2 + 1] = rhs_fragment[x_item_idx * 2];
                rhs_fragment[x_item_idx * 2] = temp;
            }

            asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%10, %11}, \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                "+f"(acc_frag[x_item_idx][0]), "+f"(acc_frag[x_item_idx][1]), 
                "+f"(acc_frag[x_item_idx][2]), "+f"(acc_frag[x_item_idx][3]),
                "+f"(acc_waste[0]), "+f"(acc_waste[1]), 
                "+f"(acc_waste[2]), "+f"(acc_waste[3]):
                "r"(rhs_fragment_int[x_item_idx * 8 + 0]), "r"(rhs_fragment_int[x_item_idx * 8 + 1]),
                "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
            );

            asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \t"
                "{%4, %5, %6, %7, %0, %1, %2, %3}, \t"
                "{%8, %9}, \t"
                "{%10, %11}, \t"
                "{%4, %5, %6, %7, %0, %1, %2, %3}; ":
                "+f"(acc_frag[x_item_idx][4]), "+f"(acc_frag[x_item_idx][5]), 
                "+f"(acc_frag[x_item_idx][6]), "+f"(acc_frag[x_item_idx][7]), 
                "+f"(acc_waste[0]), "+f"(acc_waste[1]), 
                "+f"(acc_waste[2]), "+f"(acc_waste[3]):
                "r"(rhs_fragment_int[x_item_idx * 8 + 4]), "r"(rhs_fragment_int[x_item_idx * 8 + 5]),
                "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
            );

            asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%10, %11}, \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                "+f"(acc_frag[x_item_idx][0]), "+f"(acc_frag[x_item_idx][1]), 
                "+f"(acc_frag[x_item_idx][2]), "+f"(acc_frag[x_item_idx][3]), 
                "+f"(acc_waste[0]), "+f"(acc_waste[1]), 
                "+f"(acc_waste[2]), "+f"(acc_waste[3]):
                "r"(rhs_fragment_int[x_item_idx * 8 + 2]), "r"(rhs_fragment_int[x_item_idx * 8 + 3]),
                "r"(lhs_fragment_int[2]), "r"(lhs_fragment_int[3])
            );

            asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \t"
                "{%4, %5, %6, %7, %0, %1, %2, %3}, \t"
                "{%8, %9}, \t"
                "{%10, %11}, \t"
                "{%4, %5, %6, %7, %0, %1, %2, %3}; ":
                "+f"(acc_frag[x_item_idx][4]), "+f"(acc_frag[x_item_idx][5]), 
                "+f"(acc_frag[x_item_idx][6]), "+f"(acc_frag[x_item_idx][7]), 
                "+f"(acc_waste[0]), "+f"(acc_waste[1]), 
                "+f"(acc_waste[2]), "+f"(acc_waste[3]):
                "r"(rhs_fragment_int[x_item_idx * 8 + 6]), "r"(rhs_fragment_int[x_item_idx * 8 + 7]),
                "r"(lhs_fragment_int[2]), "r"(lhs_fragment_int[3])
            );
            
        }

        rhs_matrix_base_low += 8;
        rhs_matrix_base_high += 8;
    }

    // TODO Residual

    unsigned mask = __ballot_sync(0xffffffff, lane_id < 2);

    // Accumulate partial sum

    #pragma unroll
    for (int x_item_idx = 0; x_item_idx < 4; x_item_idx ++){
        #pragma unroll
        for (int acc_idx = 0; acc_idx < 2; acc_idx ++){
            float temp1 = __shfl_up_sync(mask, acc_frag[x_item_idx][acc_idx + 2], 2);
            float temp2 = __shfl_up_sync(mask, acc_frag[x_item_idx][acc_idx + 6], 2);
            if (lane_id > 1){
                acc_frag[x_item_idx][acc_idx] = temp1;
                acc_frag[x_item_idx][acc_idx + 4] = temp2;
            }
        }
    }

    __syncwarp();

    int src_line = (threadIdx.x + 16) % 32;
    #pragma unroll
    for (int x_item_idx = 0; x_item_idx < 4; x_item_idx ++){
        #pragma unroll
        for (int acc_idx = 0; acc_idx < 2; acc_idx ++){
            acc_frag[x_item_idx][acc_idx] += __shfl_sync(0xffffffff, acc_frag[x_item_idx][(acc_idx + 4) % 8], src_line, 32);
        }
    }

    #pragma unroll
    for (int acc_idx = 0; acc_idx < 2; acc_idx ++){
        acc_frag[0][acc_idx] += __shfl_down_sync(0xffffffff, acc_frag[0][acc_idx], 8);
        acc_frag[1][acc_idx] += __shfl_down_sync(0xffffffff, acc_frag[1][acc_idx], 8);
        acc_frag[2][acc_idx] += __shfl_up_sync(0xffffffff, acc_frag[2][acc_idx], 8);
        acc_frag[3][acc_idx] += __shfl_up_sync(0xffffffff, acc_frag[3][acc_idx], 8);

        acc_frag[0][acc_idx] += __shfl_down_sync(0xffffffff, acc_frag[0][acc_idx], 4);
        acc_frag[1][acc_idx] += __shfl_up_sync(0xffffffff, acc_frag[1][acc_idx], 4);
        acc_frag[2][acc_idx] += __shfl_down_sync(0xffffffff, acc_frag[2][acc_idx], 4);
        acc_frag[3][acc_idx] += __shfl_up_sync(0xffffffff, acc_frag[3][acc_idx], 4);
    }

    #pragma unroll
    for (int acc_idx = 0; acc_idx < 2; acc_idx ++){
        switch(octet_id){
            case 0: break;
            case 1: acc_frag[0][acc_idx] = acc_frag[1][acc_idx]; break;
            case 2: acc_frag[0][acc_idx] = acc_frag[2][acc_idx]; break;
            case 3: acc_frag[0][acc_idx] = acc_frag[3][acc_idx]; break;
        }
    }

    // Convert the output to half if necessary
    if (mixed == 1){
        OutType *acc_frag_half = reinterpret_cast<OutType *>(acc_frag);
        #pragma unroll
        for (int i = 0; i < 4; i++){
            acc_frag_half[i] = __float2half(acc_frag[0][i]);
        }
    }

    StoreType *acc_frag_float2 = reinterpret_cast<StoreType *>(acc_frag);

    int out_col_id = octet_id * 8 + high_group * 4 + lane_id;

    StoreType *output_values_ = reinterpret_cast<StoreType *>(output_values + (row_offset + n_index + out_col_id) * VecLength);


    if (out_col_id < nonzeros){
        *(output_values_) = *(acc_frag_float2);
    }
}


template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=2, int Tile_X=32, int Tile_K=64>
__global__ void mmaSddmmKernel2reg(int m_vec, int k, int n, 
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ lhs_matrix,
    const half* __restrict__ rhs_matrix,
    OutType* __restrict__ output_values)
{
    mmaSddmmKernel2reg_<LoadType, OutType, StoreType, Residual, VecLength, Tile_X, Tile_K>(
        m_vec, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values
    );
}


template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=2, int Tile_X=32, int Tile_K=64>
__global__ void batchedMmaSddmmKernel2reg(int m_vec, int k, int n, 
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ lhs_matrix_b,
    int lhs_stride,
    const half* __restrict__ rhs_matrix_b,
    int rhs_stride,
    OutType* __restrict__ output_values_b,
    int output_stride)
{
    // Get the entry index
    int entry_idx = blockIdx.z;
    const half* lhs_matrix = lhs_matrix_b + entry_idx * lhs_stride;
    const half* rhs_matrix = rhs_matrix_b + entry_idx * rhs_stride;
    OutType* output_values = output_values_b + entry_idx * output_stride;

    mmaSddmmKernel2reg_<LoadType, OutType, StoreType, Residual, VecLength, Tile_X, Tile_K>(
        m_vec, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values
    );

}


template <typename LoadType, typename OutType, typename StoreType, int Residual = false, int VecLength=4, int Tile_K=128, int Tile_X=32>
__device__ void cudaSddmmKernel_(int m_vec, int k, int n, 
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
                    accumulator_fragment[x_item_idx + v_idx * Tile_X] += __half2float(__hmul(lhs_value[v_idx], rhs_value));
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
                        accumulator_fragment[x_item_idx] += __half2float(__hmul(lhs_value, rhs_value));
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
            output_fragment_[out_idx * VecLength + v_idx] = __float2half(accumulator_fragment[out_idx + v_idx * Tile_X]);
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

template <typename LoadType, typename OutType, typename StoreType, int Residual = false, int VecLength=4, int Tile_K=128, int Tile_X=32>
__global__ void cudaSddmmKernel(int m_vec, int k, int n, 
                    const int* __restrict__ row_indices,
                    const int* __restrict__ row_offsets,
                    const int* __restrict__ column_indices,
                    const half* __restrict__ lhs_matrix,
                    const half* __restrict__ rhs_matrix,
                    OutType* __restrict__ output_values)
{
    cudaSddmmKernel_<LoadType, OutType, StoreType, Residual, VecLength, Tile_K, Tile_X>(
        m_vec, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values
    );
}


template <typename LoadType, typename OutType, typename StoreType, int Residual = false, int VecLength=4, int Tile_K=128, int Tile_X=32>
__global__ void batchedCudaSddmmKernel(int m_vec, int k, int n, 
                    const int* __restrict__ row_indices,
                    const int* __restrict__ row_offsets,
                    const int* __restrict__ column_indices,
                    const half* __restrict__ lhs_matrix_b,
                    int lhs_stride,
                    const half* __restrict__ rhs_matrix_b,
                    int rhs_stride,
                    OutType* __restrict__ output_values_b,
                    int output_stride)
{
    // get entry index
    int entry_idx = blockIdx.z;
    const half* lhs_matrix = lhs_matrix_b + entry_idx * lhs_stride;
    const half* rhs_matrix = rhs_matrix_b + entry_idx * rhs_stride;
    OutType* output_values = output_values_b + entry_idx * output_stride;

    cudaSddmmKernel_<LoadType, OutType, StoreType, Residual, VecLength, Tile_K, Tile_X>(
        m_vec, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values
    );
}


torch::Tensor sddmm_cuda(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int vec_length)
{
    int m = lhs_matrix.size(0);
    int k = lhs_matrix.size(1);
    int n = rhs_matrix.size(0);

    int m_vec = m / vec_length;

    int nnz = column_indices.numel();

    dim3 grid;
    grid.x = m_vec;
    grid.y = n / 32;

    dim3 block;
    block.x = 32;

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(lhs_matrix.device());

    auto output_vals = torch::empty({nnz * vec_length, }, options);

    switch(vec_length){
        case 8:
            mmaSddmmKernel8reg<float4, half, float, false, 8><<<grid, block>>>(
                m_vec, k, n, row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(),
                reinterpret_cast<half *>(lhs_matrix.data<torch::Half>()), 
                reinterpret_cast<half *>(rhs_matrix.data<torch::Half>()), 
                reinterpret_cast<half *>(output_vals.data<torch::Half>())
            );
            break;
        case 4:
            mmaSddmmKernel4reg<float4, half, float2, false, 4><<<grid, block>>>(
                m_vec, k, n, row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(),
                reinterpret_cast<half *>(lhs_matrix.data<torch::Half>()), 
                reinterpret_cast<half *>(rhs_matrix.data<torch::Half>()), 
                reinterpret_cast<half *>(output_vals.data<torch::Half>())
            );
            break;
        case 2:
            mmaSddmmKernel2reg<float4, half, float, false, 2><<<grid, block>>>(
                m_vec, k, n, row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(),
                reinterpret_cast<half *>(lhs_matrix.data<torch::Half>()), 
                reinterpret_cast<half *>(rhs_matrix.data<torch::Half>()), 
                reinterpret_cast<half *>(output_vals.data<torch::Half>())
            );
            break;
        case 1:
            cudaSddmmKernel<float, half, half, false, 1, 64><<<grid, block>>>(
                m_vec, k, n, row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(),
                reinterpret_cast<half *>(lhs_matrix.data<torch::Half>()), 
                reinterpret_cast<half *>(rhs_matrix.data<torch::Half>()), 
                reinterpret_cast<half *>(output_vals.data<torch::Half>())
            );
            break;
    }

    return output_vals;
}



torch::Tensor batched_sddmm_cuda(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int vec_length)
{
    int m = lhs_matrix.size(-2);
    int k = lhs_matrix.size(-1);
    int n = rhs_matrix.size(-2);
    int batch_size = lhs_matrix.numel() / (m * k);

    int m_vec = m / vec_length;
    int nnz = column_indices.numel();

    int lhs_stride = m * k;
    int rhs_stride = n * k;
    int output_stride = nnz * vec_length;

    dim3 grid;
    grid.x = m_vec;
    grid.y = n / 32;
    grid.z = batch_size;

    dim3 block;
    block.x = 32;

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(lhs_matrix.device());

    auto output_vals = torch::empty({batch_size, nnz * vec_length, }, options);

    switch(vec_length){
        case 8:
            batchedMmaSddmmKernel8reg<float4, half, float, false, 8><<<grid, block>>>(
                m_vec, k, n, row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(),
                reinterpret_cast<half *>(lhs_matrix.data<torch::Half>()),
                lhs_stride,
                reinterpret_cast<half *>(rhs_matrix.data<torch::Half>()), 
                rhs_stride,
                reinterpret_cast<half *>(output_vals.data<torch::Half>()),
                output_stride
            );
            break;
        case 4:
            batchedMmaSddmmKernel4reg<float4, half, float2, false, 4><<<grid, block>>>(
                m_vec, k, n, row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(),
                reinterpret_cast<half *>(lhs_matrix.data<torch::Half>()),
                lhs_stride, 
                reinterpret_cast<half *>(rhs_matrix.data<torch::Half>()),
                rhs_stride,
                reinterpret_cast<half *>(output_vals.data<torch::Half>()),
                output_stride
            );
            break;
        case 2:
            batchedMmaSddmmKernel2reg<float4, half, float, false, 2><<<grid, block>>>(
                m_vec, k, n, row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(),
                reinterpret_cast<half *>(lhs_matrix.data<torch::Half>()),
                lhs_stride, 
                reinterpret_cast<half *>(rhs_matrix.data<torch::Half>()),
                rhs_stride,
                reinterpret_cast<half *>(output_vals.data<torch::Half>()),
                output_stride
            );
            break;
        case 1:
            batchedCudaSddmmKernel<float, half, half, false, 1, 64><<<grid, block>>>(
                m_vec, k, n, row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(),
                reinterpret_cast<half *>(lhs_matrix.data<torch::Half>()),
                lhs_stride,
                reinterpret_cast<half *>(rhs_matrix.data<torch::Half>()),
                rhs_stride,
                reinterpret_cast<half *>(output_vals.data<torch::Half>()),
                output_stride
            );
            break;
    }

    return output_vals;
}