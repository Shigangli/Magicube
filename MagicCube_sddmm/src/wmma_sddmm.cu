#include "cuda_fp16.h"
#include "../include/wmma_sddmm.cuh"
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include <mma.h>

using namespace nvcuda;


namespace sddmm{

// In this version, we try to use the mma to avoid shared memory access and sync barriers
// The inverted thread group is solved by using additional registers to hold the inversed operands
template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=8, int Tile_X=32, int Tile_K=64>
__global__ void mmaSddmmKernel8reg(int m_vec, int k, int n, 
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
            acc_frag_half[i] = (OutType)acc_frag[0][0][i];
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



template <typename LoadType, int Residual, int kBlockItemsY=1, int kBlockItemsK=128,
          int kBlockItemsX=32, int kBlockWidth=32>
cudaError_t wmmaSddmmEx(
    int m, int k, int n, int nonzeros, const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets, const int* __restrict__ column_indices,
    const half* __restrict__ lhs_matrix, const half* __restrict__ rhs_matrix,
    float* __restrict__ output_values, int vec_length, cudaStream_t stream, int algorithm) 
{
    dim3 grid_dim(std::ceil(static_cast<float>(m) / kBlockItemsY), std::ceil(static_cast<float>(n) / kBlockItemsX), 1);
    dim3 block_dim(kBlockWidth, kBlockItemsY, 1);
    
    switch(vec_length){
        case 8:
            switch (algorithm){
                case 0:
                    //printf("wmma\n");
                    wmmaSddmmKernel8<LoadType, float, float2, Residual, 8><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
                case 1:
                    //printf("mma_reg\n");
                    mmaSddmmKernel8reg<LoadType, float, float2, Residual, 8><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
                case 2:
                    //printf("mma_shfl\n");
                    mmaSddmmKernel8shfl<LoadType, float, float2, Residual, 8><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
                case 3:
                    //printf("mma_arch\n");
                    mmaSddmmKernel8fake<LoadType, float, float2, Residual, 8><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
            }
            break;
        case 4:
            switch (algorithm){
                case 0:
                    //printf("wmma\n");
                    wmmaSddmmKernel4<LoadType, float, float4, Residual, 4><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
                case 1:
                    //printf("mma_reg\n");
                    mmaSddmmKernel4reg<LoadType, float, float4, Residual, 4><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
                case 2:
                    //printf("mma_shfl\n");
                    mmaSddmmKernel4shfl<LoadType, float, float4, Residual, 4><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
                case 3:
                    //printf("mma_arch\n");
                    mmaSddmmKernel4fake<LoadType, float, float4, Residual, 4><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
            }
            break;
        case 2:
            switch (algorithm){
                case 0:
                    //printf("wmma\n");
                    wmmaSddmmKernel2<LoadType, float, float2, Residual, 2><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
                case 1:
                    //printf("mma_reg\n");
                    mmaSddmmKernel2reg<LoadType, float, float2, Residual, 2><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
                case 2:
                    //printf("mma_shfl\n");
                    mmaSddmmKernel2shfl<LoadType, float, float2, Residual, 2><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
                case 3:
                    //printf("mma_arch\n");
                    mmaSddmmKernel2fake<LoadType, float, float2, Residual, 2><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
            }
            break;
        default:
            printf("Unsupported Vector Length!\n");
    }
  return cudaGetLastError();
}


template <typename LoadType, int Residual, int kBlockItemsY=1, int kBlockItemsK=128,
          int kBlockItemsX=32, int kBlockWidth=32>
cudaError_t wmmaSddmmEx(
    int m, int k, int n, int nonzeros, const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets, const int* __restrict__ column_indices,
    const half* __restrict__ lhs_matrix, const half* __restrict__ rhs_matrix,
    half* __restrict__ output_values, int vec_length, cudaStream_t stream, int algorithm) 
{
    dim3 grid_dim(std::ceil(static_cast<float>(m) / kBlockItemsY), std::ceil(static_cast<float>(n) / kBlockItemsX), 1);
    dim3 block_dim(kBlockWidth, kBlockItemsY, 1);
    
    switch(vec_length){
        case 8:
            switch (algorithm){
                case 0:
                    //printf("wmma\n");
                    wmmaSddmmKernel8<LoadType, half, float, Residual, 8><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
                case 1:
                    //printf("mma_reg\n");
                    mmaSddmmKernel8reg<LoadType, half, float, Residual, 8><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
                case 2:
                    //printf("mma_shfl\n");
                    mmaSddmmKernel8shfl<LoadType, half, float, Residual, 8><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
                case 3:
                    //printf("mma_arch\n");
                    mmaSddmmKernel8fake<LoadType, half, float, Residual, 8><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
            }
            break;
        case 4:
            switch (algorithm){
                case 0:
                    //printf("wmma\n");
                    wmmaSddmmKernel4<LoadType, half, float2, Residual, 4><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
                case 1:
                    //printf("mma_reg\n");
                    mmaSddmmKernel4reg<LoadType, half, float2, Residual, 4><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
                case 2:
                    //printf("mma_shfl\n");
                    mmaSddmmKernel4shfl<LoadType, half, float2, Residual, 4><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
                case 3:
                    //printf("mma_arch\n");
                    mmaSddmmKernel4fake<LoadType, half, float2, Residual, 4><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
            }
            break;
        case 2:
            switch (algorithm){
                case 0:
                    //printf("wmma\n");
                    wmmaSddmmKernel2<LoadType, half, float, Residual, 2><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
                case 1:
                    //printf("mma_reg\n");
                    mmaSddmmKernel2reg<LoadType, half, float, Residual, 2><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
                case 2:
                    //printf("mma_shfl\n");
                    mmaSddmmKernel2shfl<LoadType, half, float, Residual, 2><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
                case 3:
                    //printf("mma_arch\n");
                    mmaSddmmKernel2fake<LoadType, half, float, Residual, 2><<<grid_dim, block_dim>>>(
                        m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
                    break;
            }
            break;
        default:
            printf("Unsupported Vector Length!\n");
    }
    return cudaGetLastError();
}


cudaError_t wmmaSddmm(int m_vec, int k, int n, int nonzeros_vec,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const half* __restrict__ lhs_matrix,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_values, 
    int vec_length, cudaStream_t stream, int algorithm) {
    
    return wmmaSddmmEx<float4, false>(
        m_vec, k, n, nonzeros_vec, row_indices, row_offsets, col_indices,
        lhs_matrix, rhs_matrix, output_values, vec_length, stream, algorithm);
}


cudaError_t wmmaSddmm(int m_vec, int k, int n, int nonzeros_vec,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const half* __restrict__ lhs_matrix,
    const half* __restrict__ rhs_matrix,
    half* __restrict__ output_values, 
    int vec_length, cudaStream_t stream, int algorithm) {
    
    return wmmaSddmmEx<float4, false>(
        m_vec, k, n, nonzeros_vec, row_indices, row_offsets, col_indices,
        lhs_matrix, rhs_matrix, output_values, vec_length, stream, algorithm);
}

cudaError_t wmmaSddmm(int m_vec, int k, int n, int nonzeros_vec,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const float* __restrict__ lhs_matrix,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_values, 
    int vec_length, cudaStream_t stream, int algorithm) {
    
    printf("wmmaSddmm doesn't support single precision\n");
    return cudaSuccess;
}

}
