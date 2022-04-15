#include "cuda_fp16.h"
#include "../include/wmma_sddmm.cuh"
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include <mma.h>

using namespace nvcuda;


namespace sddmm{

__device__ void print_val_f(int blockid, int threadid, int tar_threadid, float value){
    if (blockid == 0 && threadid == tar_threadid) printf("tid: %d, value is: %.4f\n", threadid, float(value));
}

__device__ void print_val_h(int blockid, int threadid, half value){
    if (blockid == 0 && threadid == 16) printf("tid: %d, value is: %.4f\n", threadid, float(value));
}

// This is the shared memory based wmmaSddmmKernel
template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=8, int Tile_X=32, int Tile_K=64>
__global__ void wmmaSddmmKernel8(int m_vec, int k, int n, 
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

    // The length of the vector type
    constexpr int kValuesPerLoad = sizeof(LoadType) / sizeof(half);
    // The Tile_K in shared memory is padded by 8 to reduce bank conflict
    constexpr int Tile_K_pad = Tile_K + 8;
    // The padded Tile_K under vector type (float4)
    constexpr int Tile_K_pad_float4 = Tile_K_pad / 8;
    // The number of column indices loaded by each thread
    constexpr int kColabItemsX = Tile_X / 32;
    // When loading, the warp is splitted to subwarps
    // The threads in the subwarp collaborate on the same Tile_K
    constexpr int SubWarp = (Tile_K / 8) > 32 ? 32 : (Tile_K / 8);
    // The amount of subwarps per warp
    constexpr int NumSubWarp = 32 / SubWarp;
    // The amount of float4 vectors loaded by each subwarp
    constexpr int kThreadItemsK_ = Tile_K / SubWarp / 8;
    // The number of steps required to fill the float4 vector
    constexpr int VecLoadStep = sizeof(float4) / sizeof(LoadType);

    constexpr int mixed = sizeof(OutType) / sizeof(half);

    //
    // Storage
    //

    // Shared memory that stores the column indices
    __shared__ int column_indices_tile_array[Tile_X];
    // Shared memory that stores the lhs_fragment
    __shared__ float4 lhs_fragment[Tile_K_pad_float4 * VecLength];
    // Shared memory that stores the rhs_fragment
    __shared__ float4 rhs_fragment[Tile_X * Tile_K_pad_float4];

    // The fragments for wmma
    wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> rhs_frag;
    wmma::fragment<wmma::matrix_b, 32, 8, 16, half, wmma::col_major> lhs_frag;
    wmma::fragment<wmma::accumulator, 32, 8, 16, float> acc_frag;


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

    // the id of the subwarp that this thread belongs to
    const int subwarp_id = threadIdx.x / SubWarp;
    // the lane id of the subwarp
    const int sublane_id = threadIdx.x % SubWarp;

    // Declare pointers to the lhs and rhs matrices
    const LoadType *lhs_matrix_ = reinterpret_cast<const LoadType *>(lhs_matrix + (m_index + subwarp_id) * k) + sublane_id;
    float4 *lhs_fragment_ = reinterpret_cast<float4 *>(lhs_fragment + subwarp_id * Tile_K_pad_float4) + sublane_id;

    const LoadType *rhs_matrix_base = reinterpret_cast<const LoadType *>(rhs_matrix) + sublane_id;
    float4 *rhs_fragment_ = reinterpret_cast<float4 *>(rhs_fragment + subwarp_id * Tile_K_pad_float4) + sublane_id;

    half *lhs_fragment_half = reinterpret_cast<half *>(lhs_fragment);
    half *rhs_fragment_half = reinterpret_cast<half *>(rhs_fragment);

    // fill the accumulator fragment with 0
    wmma::fill_fragment(acc_frag, 0.0f);

    // k_ is used to store the overall k of the kernel
    const int k_ = k;
    // When loading, we will use half8 (float4) to fill the Tile_K vectors
    
    
    static_assert(VecLength % NumSubWarp == 0, "the veclength should be divisible by the number of subwarps");
    
    #pragma nounroll
    for (; k >= Tile_K; k -= Tile_K){
        // Load a tile from the dense lhs matrix into the shared memory
        #pragma unroll
        for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
            #pragma unroll
            for (int v_idx = 0; v_idx < VecLength; v_idx +=NumSubWarp){
                LoadType lhs_reg[VecLoadStep];
                #pragma unroll
                for (int r_idx = 0; r_idx < VecLoadStep; r_idx ++){
                    lhs_reg[r_idx] = __ldg(lhs_matrix_ + v_idx * k_ / kValuesPerLoad + r_idx * SubWarp);
                }
                *(lhs_fragment_ + v_idx * Tile_K_pad_float4 + k_item_idx) = reinterpret_cast<float4 *>(lhs_reg)[0];
            }
            lhs_matrix_ += SubWarp * VecLoadStep;
        }

        column_indices_tile_ = column_indices_tile + subwarp_id;

        // Load a tile from the dense lhs matrix into shared memory
        #pragma unroll
        for (int x_item_idx = 0; x_item_idx < Tile_X; x_item_idx += NumSubWarp){
            const LoadType *rhs_matrix_ = rhs_matrix_base + *column_indices_tile_;
            #pragma unroll
            for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
                LoadType rhs_reg[VecLoadStep];
                #pragma unroll
                for (int r_idx = 0; r_idx < VecLoadStep; r_idx ++){
                    rhs_reg[r_idx] = __ldg(rhs_matrix_ + r_idx * SubWarp);
                }
                *(rhs_fragment_ + x_item_idx * Tile_K_pad_float4 + k_item_idx) = reinterpret_cast<float4 *>(rhs_reg)[0];
                rhs_matrix_ += SubWarp * VecLoadStep;
            }
            column_indices_tile_ += NumSubWarp;
        }

        rhs_matrix_base += Tile_K / kValuesPerLoad;

        __syncthreads();

        #pragma unroll
        for (int k_item_idx = 0; k_item_idx < Tile_K; k_item_idx += 16){
            // Load the inputs
            wmma::load_matrix_sync(rhs_frag, rhs_fragment_half + k_item_idx, Tile_K_pad);
            wmma::load_matrix_sync(lhs_frag, lhs_fragment_half + k_item_idx, Tile_K_pad);
            wmma::mma_sync(acc_frag, rhs_frag, lhs_frag, acc_frag);
        }
    }

    // wmma::fill_fragment(acc_frag, 0.0f);

    // TODO Residual

    /*

    float2 *output_values_ = reinterpret_cast<float2 *>(output_values + (row_offset + n_index) * VecLength);

    float2 *acc_frag_float2 = reinterpret_cast<float2 *>(acc_frag.x);

    int out_id = (threadIdx.x / 4) * 32 - (threadIdx.x / 16) * 112 + (threadIdx.x % 2) * 4 + (threadIdx.x / 2) % 2;

    if (out_id < nonzeros * 4){
        *(output_values_ + out_id) = *(acc_frag_float2);
        *(output_values_ + out_id + 2) = *(acc_frag_float2 + 2);
    }

    if (out_id + 8 < nonzeros * 4){
        *(output_values_ + out_id + 8) = *(acc_frag_float2 + 1);
        *(output_values_ + out_id + 10) = *(acc_frag_float2 + 3);
    }
    */
    
    
    StoreType *output_values_ = reinterpret_cast<StoreType *>(output_values + (row_offset + n_index) * VecLength);

    // Convert the output to half type
    if (mixed == 1){
        OutType *acc_frag_half = reinterpret_cast<OutType *>(acc_frag.x);
        #pragma unroll
        for (int i = 0; i < 8; i++){
            acc_frag_half[i] = (OutType)acc_frag.x[i];
        }
    }
    StoreType *acc_frag_float2 = reinterpret_cast<StoreType *>(acc_frag.x);

    int out_id = (threadIdx.x / 4) * 32 - (threadIdx.x / 16) * 112 + (threadIdx.x % 2) * 4 + (threadIdx.x / 2) % 2;

    if (out_id < nonzeros * 4){
        *(output_values_ + out_id) = *(acc_frag_float2);
        *(output_values_ + out_id + 2) = *(acc_frag_float2 + 2);
    }

    if (out_id + 8 < nonzeros * 4){
        *(output_values_ + out_id + 8) = *(acc_frag_float2 + 1);
        *(output_values_ + out_id + 10) = *(acc_frag_float2 + 3);
    }
    
}



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


// In this version, we try to solve the inversed group problem with warp shuffle
template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=8, int Tile_X=32, int Tile_K=64>
__global__ void mmaSddmmKernel8shfl(int m_vec, int k, int n, 
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

    float acc_frag[4][8] = {0};

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

    int oprand_src_line = (threadIdx.x + 16) % 32;
    // half *lhs_fragment_half = reinterpret_cast<half *>(lhs_fragment);
    // half *rhs_fragment_half = reinterpret_cast<half *>(rhs_fragment);

    #pragma nounroll
    for (; k >= Tile_K; k -= Tile_K){
        // Load a tile from the dense lhs matrix into the register
        lhs_fragment[0] = __ldg(lhs_matrix_low);
        lhs_fragment[1] = __ldg(lhs_matrix_high);

        // High group Shuffle
        if (high_group){
            float4 temp = lhs_fragment[1];
            lhs_fragment[1] = lhs_fragment[0];
            lhs_fragment[0] = temp;
        }

        #pragma unroll
        for (int i=0; i < 4; i++){
            lhs_fragment_int[4 + i] = __shfl_sync(0xffffffff, lhs_fragment_int[4 + i], oprand_src_line, 32);
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
            #pragma unroll
            for (int i=0; i < 4; i++){
                rhs_fragment_int[(x_item_idx * 2 + 1) * 4 + i] = __shfl_sync(0xffffffff, rhs_fragment_int[(x_item_idx * 2 + 1) * 4 + i], oprand_src_line, 32);
            }

            asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%10, %11}, \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                "+f"(acc_frag[x_item_idx][0]), "+f"(acc_frag[x_item_idx][1]), 
                "+f"(acc_frag[x_item_idx][2]), "+f"(acc_frag[x_item_idx][3]), 
                "+f"(acc_frag[x_item_idx][4]), "+f"(acc_frag[x_item_idx][5]), 
                "+f"(acc_frag[x_item_idx][6]), "+f"(acc_frag[x_item_idx][7]):
                "r"(rhs_fragment_int[x_item_idx * 8 + 0]), "r"(rhs_fragment_int[x_item_idx * 8 + 1]),
                "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
            );

            asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%10, %11}, \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                "+f"(acc_frag[x_item_idx][0]), "+f"(acc_frag[x_item_idx][1]), 
                "+f"(acc_frag[x_item_idx][2]), "+f"(acc_frag[x_item_idx][3]), 
                "+f"(acc_frag[x_item_idx][4]), "+f"(acc_frag[x_item_idx][5]), 
                "+f"(acc_frag[x_item_idx][6]), "+f"(acc_frag[x_item_idx][7]):
                "r"(rhs_fragment_int[x_item_idx * 8 + 2]), "r"(rhs_fragment_int[x_item_idx * 8 + 3]),
                "r"(lhs_fragment_int[2]), "r"(lhs_fragment_int[3])
            );

            asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%10, %11}, \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                "+f"(acc_frag[x_item_idx][0]), "+f"(acc_frag[x_item_idx][1]), 
                "+f"(acc_frag[x_item_idx][2]), "+f"(acc_frag[x_item_idx][3]), 
                "+f"(acc_frag[x_item_idx][4]), "+f"(acc_frag[x_item_idx][5]), 
                "+f"(acc_frag[x_item_idx][6]), "+f"(acc_frag[x_item_idx][7]):
                "r"(rhs_fragment_int[x_item_idx * 8 + 4]), "r"(rhs_fragment_int[x_item_idx * 8 + 5]),
                "r"(lhs_fragment_int[4]), "r"(lhs_fragment_int[5])
            );

            asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%10, %11}, \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                "+f"(acc_frag[x_item_idx][0]), "+f"(acc_frag[x_item_idx][1]), 
                "+f"(acc_frag[x_item_idx][2]), "+f"(acc_frag[x_item_idx][3]), 
                "+f"(acc_frag[x_item_idx][4]), "+f"(acc_frag[x_item_idx][5]), 
                "+f"(acc_frag[x_item_idx][6]), "+f"(acc_frag[x_item_idx][7]):
                "r"(rhs_fragment_int[x_item_idx * 8 + 6]), "r"(rhs_fragment_int[x_item_idx * 8 + 7]),
                "r"(lhs_fragment_int[6]), "r"(lhs_fragment_int[7])
            );  
        }

        rhs_matrix_base_low += 8;
        rhs_matrix_base_high += 8;
    }

    // TODO Residual

    // Accumulate partial sum

    #pragma unroll
    for (int acc_idx = 0; acc_idx < 8; acc_idx ++){
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
    for (int acc_idx = 0; acc_idx < 8; acc_idx ++){
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
        for (int i = 0; i < 8; i++){
            acc_frag_half[i] = (OutType)acc_frag[0][i];
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


// In this version, we simulate the speedup with the hardware optimization
template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=8, int Tile_X=32, int Tile_K=64>
__global__ void mmaSddmmKernel8fake(int m_vec, int k, int n, 
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

    float acc_frag[4][8] = {0};

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
                "+f"(acc_frag[x_item_idx][0]), "+f"(acc_frag[x_item_idx][1]), 
                "+f"(acc_frag[x_item_idx][2]), "+f"(acc_frag[x_item_idx][3]), 
                "+f"(acc_frag[x_item_idx][4]), "+f"(acc_frag[x_item_idx][5]), 
                "+f"(acc_frag[x_item_idx][6]), "+f"(acc_frag[x_item_idx][7]):
                "r"(rhs_fragment_int[x_item_idx * 8 + 0]), "r"(rhs_fragment_int[x_item_idx * 8 + 1]),
                "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
            );

            asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%10, %11}, \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                "+f"(acc_frag[x_item_idx][0]), "+f"(acc_frag[x_item_idx][1]), 
                "+f"(acc_frag[x_item_idx][2]), "+f"(acc_frag[x_item_idx][3]), 
                "+f"(acc_frag[x_item_idx][4]), "+f"(acc_frag[x_item_idx][5]), 
                "+f"(acc_frag[x_item_idx][6]), "+f"(acc_frag[x_item_idx][7]):
                "r"(rhs_fragment_int[x_item_idx * 8 + 2]), "r"(rhs_fragment_int[x_item_idx * 8 + 3]),
                "r"(lhs_fragment_int[2]), "r"(lhs_fragment_int[3])
            );

            asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%10, %11}, \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                "+f"(acc_frag[x_item_idx][0]), "+f"(acc_frag[x_item_idx][1]), 
                "+f"(acc_frag[x_item_idx][2]), "+f"(acc_frag[x_item_idx][3]), 
                "+f"(acc_frag[x_item_idx][4]), "+f"(acc_frag[x_item_idx][5]), 
                "+f"(acc_frag[x_item_idx][6]), "+f"(acc_frag[x_item_idx][7]):
                "r"(rhs_fragment_int[x_item_idx * 8 + 4]), "r"(rhs_fragment_int[x_item_idx * 8 + 5]),
                "r"(lhs_fragment_int[4]), "r"(lhs_fragment_int[5])
            );

            asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                "{%8, %9}, \t"
                "{%10, %11}, \t"
                "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                "+f"(acc_frag[x_item_idx][0]), "+f"(acc_frag[x_item_idx][1]), 
                "+f"(acc_frag[x_item_idx][2]), "+f"(acc_frag[x_item_idx][3]), 
                "+f"(acc_frag[x_item_idx][4]), "+f"(acc_frag[x_item_idx][5]), 
                "+f"(acc_frag[x_item_idx][6]), "+f"(acc_frag[x_item_idx][7]):
                "r"(rhs_fragment_int[x_item_idx * 8 + 6]), "r"(rhs_fragment_int[x_item_idx * 8 + 7]),
                "r"(lhs_fragment_int[6]), "r"(lhs_fragment_int[7])
            );  
        }

        rhs_matrix_base_low += 8;
        rhs_matrix_base_high += 8;
    }

    // TODO Residual

    // Accumulate partial sum

    #pragma unroll
    for (int acc_idx = 0; acc_idx < 8; acc_idx ++){
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
    for (int acc_idx = 0; acc_idx < 8; acc_idx ++){
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
        for (int i = 0; i < 8; i ++){
            acc_frag_half[i] = (OutType)acc_frag[0][i];
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



template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=4, int Tile_X=32, int Tile_K=64>
__global__ void wmmaSddmmKernel4(int m_vec, int k, int n, 
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ lhs_matrix,
    const half* __restrict__ rhs_matrix,
    OutType* __restrict__ output_values)
{
    constexpr int kValuesPerLoad = sizeof(LoadType) / sizeof(half);
    constexpr int Tile_K_pad = Tile_K + 8;
    constexpr int Tile_K_pad_float4 = Tile_K_pad / 8;
    
    // Tile_Y = 1
    int m_index_vec = blockIdx.x;
    int n_index = blockIdx.y * Tile_X;

    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset;

    // If this thread block has no nonzeros in the row to process, exit early
    if (n_index >= nonzeros) return;

    // Calculate the number of nonzeros that this thread block processes
    nonzeros = min(nonzeros - n_index, Tile_X);

    // We use he shared memory here
    __shared__ float4 lhs_fragment[Tile_K_pad_float4 * VecLength];

    // Shared memory tile for the output column indices
    __shared__ int column_indices_tile_array[Tile_X];

    int* column_indices_tile = column_indices_tile_array;

    __shared__ float4 rhs_fragment[Tile_X * Tile_K_pad_float4];


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

    // When loading, we will use half8 (float4) to fill the Tile_K vectors
    constexpr int SubWarp = (Tile_K / 8) > 32 ? 32 : (Tile_K / 8);
    constexpr int NumSubWarp = 32 / SubWarp;
    static_assert(VecLength % NumSubWarp == 0, "the veclength should be divisible by the number of subwarps");
    // The lhs fragment has VecLength rows, and each row has Tile_K entries
    // Each subwarp will load one row into the shared memory
    constexpr int kThreadItemsK_ = Tile_K / SubWarp / 8;
    const int subwarp_id = threadIdx.x / SubWarp;
    const int sublane_id = threadIdx.x % SubWarp;
    const LoadType *lhs_matrix_ = reinterpret_cast<const LoadType *>(lhs_matrix + (m_index + subwarp_id) * k) + sublane_id;
    float4 *lhs_fragment_ = reinterpret_cast<float4 *>(lhs_fragment + subwarp_id * Tile_K_pad_float4) + sublane_id;

    constexpr int VecLoadStep = sizeof(float4) / sizeof(LoadType);

    // Similarly, the when loading the rhs matrix, each subwarp will load one column into the shared memory
    const LoadType *rhs_matrix_base = reinterpret_cast<const LoadType *>(rhs_matrix) + sublane_id;
    float4 *rhs_fragment_ = reinterpret_cast<float4 *>(rhs_fragment + subwarp_id * Tile_K_pad_float4) + sublane_id;

    // The fragments for wmma
    wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> rhs_frag;
    wmma::fragment<wmma::matrix_b, 32, 8, 16, half, wmma::col_major> lhs_frag;
    wmma::fragment<wmma::accumulator, 32, 8, 16, float> acc_frag;

    int *lhs_frag_r = reinterpret_cast<int *>(lhs_frag.x);
    int *rhs_frag_r = reinterpret_cast<int *>(rhs_frag.x);

    // do the computation
    half *rhs_fragment_half = reinterpret_cast<half *>(rhs_fragment);

    float4 *lhs_frag_float4 = reinterpret_cast<float4 *>(lhs_frag.x);

    int lhs_vec_idx = threadIdx.x % 4 + (threadIdx.x / 16) * 4;
    lhs_vec_idx = lhs_vec_idx * (Tile_K / 16)  + lhs_vec_idx / 2;

    // Fill the accumulator fragment
    wmma::fill_fragment(acc_frag, 0.0f);


    const int k_ = k;
    #pragma nounroll
    for (; k >= Tile_K; k -= Tile_K){
        // Loading the matrices should be the same
        // Load a tile from the dense lhs matrix into the shared memory
        #pragma unroll
        for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
            #pragma unroll
            for (int v_idx = 0; v_idx < VecLength; v_idx +=NumSubWarp){
                LoadType lhs_reg[VecLoadStep];
                #pragma unroll
                for (int r_idx = 0; r_idx < VecLoadStep; r_idx ++){
                    lhs_reg[r_idx] = __ldg(lhs_matrix_ + v_idx * k_ / kValuesPerLoad + r_idx * SubWarp);
                }
                *(lhs_fragment_ + v_idx * Tile_K_pad_float4 + k_item_idx) = reinterpret_cast<float4 *>(lhs_reg)[0];
            }
            lhs_matrix_ += SubWarp * VecLoadStep;
        }

        column_indices_tile_ = column_indices_tile + subwarp_id;

        // Load a tile from the dense lhs matrix into shared memory
        #pragma unroll
        for (int x_item_idx = 0; x_item_idx < Tile_X; x_item_idx += NumSubWarp){
            const LoadType *rhs_matrix_ = rhs_matrix_base + *column_indices_tile_;
            #pragma unroll
            for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
                LoadType rhs_reg[VecLoadStep];
                #pragma unroll
                for (int r_idx = 0; r_idx < VecLoadStep; r_idx ++){
                    rhs_reg[r_idx] = __ldg(rhs_matrix_ + r_idx * SubWarp);
                }
                *(rhs_fragment_ + x_item_idx * Tile_K_pad_float4 + k_item_idx) = reinterpret_cast<float4 *>(rhs_reg)[0];
                rhs_matrix_ += SubWarp * VecLoadStep;
            }
            column_indices_tile_ += NumSubWarp;
        }

        rhs_matrix_base += Tile_K / kValuesPerLoad;

        __syncthreads();

        #pragma unroll
        for (int k_item_idx = 0; k_item_idx < Tile_K / 2; k_item_idx += 16){
            lhs_frag_float4[0] = *(lhs_fragment + lhs_vec_idx + k_item_idx/8);
            lhs_frag_float4[1] = *(lhs_fragment + lhs_vec_idx + k_item_idx/8 + 1);
            //wmma::load_matrix_sync(lhs_frag, lhs_fragment_half + k_item_idx, Tile_K / 2);
            // Load the lhs matrix
            #pragma unroll
            for (int step = 0; step < 2; step ++){
                if (step == 0){
                    wmma::load_matrix_sync(rhs_frag, rhs_fragment_half + k_item_idx, Tile_K_pad);
                    asm("wmma.mma.sync.aligned.row.col.m32n8k16.f32.f32 \t"
                        "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                        "{%8, %9, %10, %11, %12, %13, %14, %15}, \t"
                        "{%16, %17, %18, %19, %20, %21, %22, %23}, \t"
                        "{%0, %1, %2, %3, %4, %5, %6, %7}; " 
                        :"+f"(acc_frag.x[0]), "+f"(acc_frag.x[1]), "+f"(acc_frag.x[2]), "+f"(acc_frag.x[3]),
                          "+f"(acc_frag.x[4]), "+f"(acc_frag.x[5]), "+f"(acc_frag.x[6]), "+f"(acc_frag.x[7]):
                          "r"(rhs_frag_r[0]), "r"(rhs_frag_r[1]), "r"(rhs_frag_r[2]), "r"(rhs_frag_r[3]),
                          "r"(rhs_frag_r[4]), "r"(rhs_frag_r[5]), "r"(rhs_frag_r[6]), "r"(rhs_frag_r[7]),
                          "r"(lhs_frag_r[0]), "r"(lhs_frag_r[1]), "r"(lhs_frag_r[2]), "r"(lhs_frag_r[3]),
                          "r"(lhs_frag_r[4]), "r"(lhs_frag_r[5]), "r"(lhs_frag_r[6]), "r"(lhs_frag_r[7])
                        );
                }
                else{
                    wmma::load_matrix_sync(rhs_frag, rhs_fragment_half + k_item_idx + Tile_K / 2, Tile_K_pad);
                    asm("wmma.mma.sync.aligned.row.col.m32n8k16.f32.f32 \t"
                        "{%1, %0, %3, %2, %5, %4, %7, %6}, \t"
                        "{%8, %9, %10, %11, %12, %13, %14, %15}, \t"
                        "{%16, %17, %18, %19, %20, %21, %22, %23}, \t"
                        "{%1, %0, %3, %2, %5, %4, %7, %6}; " 
                        :"+f"(acc_frag.x[0]), "+f"(acc_frag.x[1]), "+f"(acc_frag.x[2]), "+f"(acc_frag.x[3]),
                          "+f"(acc_frag.x[4]), "+f"(acc_frag.x[5]), "+f"(acc_frag.x[6]), "+f"(acc_frag.x[7]):
                          "r"(rhs_frag_r[0]), "r"(rhs_frag_r[1]), "r"(rhs_frag_r[2]), "r"(rhs_frag_r[3]),
                          "r"(rhs_frag_r[4]), "r"(rhs_frag_r[5]), "r"(rhs_frag_r[6]), "r"(rhs_frag_r[7]),
                          "r"(lhs_frag_r[0]), "r"(lhs_frag_r[1]), "r"(lhs_frag_r[2]), "r"(lhs_frag_r[3]),
                          "r"(lhs_frag_r[4]), "r"(lhs_frag_r[5]), "r"(lhs_frag_r[6]), "r"(lhs_frag_r[7])
                        );
                }

            }
        }

        // __syncthreads();
    }

    // wmma::fill_fragment(acc_frag, 0.0f);

    // TODO Residual

    // Now each thread has 4 elements in acc_frag.x[0], [2], [4], [6]
    // We load them into a register.
    OutType output_fragment[4];
    float reg_share[2];
    if ((threadIdx.x / 2) % 2 == 0){
        reg_share[0] = acc_frag.x[2];
        reg_share[1] = acc_frag.x[6];
    }
    else{
        reg_share[0] = acc_frag.x[0];
        reg_share[1] = acc_frag.x[4];
    }
    // We use warp shuffle to exchange data within the registers
    int src_line = threadIdx.x - ((threadIdx.x / 2) % 2) * 4 + 2;

    reg_share[0] = __shfl_sync(0xffffffff, reg_share[0], src_line, 32);
    reg_share[1] = __shfl_sync(0xffffffff, reg_share[1], src_line, 32);

    if ((threadIdx.x / 2) % 2 == 0){
        output_fragment[0] = (OutType)acc_frag.x[0];
        output_fragment[1] = (OutType)reg_share[0];
        output_fragment[2] = (OutType)acc_frag.x[4];
        output_fragment[3] = (OutType)reg_share[1];
    }
    else{
        output_fragment[0] = (OutType)reg_share[0];
        output_fragment[1] = (OutType)acc_frag.x[2];
        output_fragment[2] = (OutType)reg_share[1];
        output_fragment[3] = (OutType)acc_frag.x[6];
    }

    StoreType *output_values_ = reinterpret_cast<StoreType *>(output_values + (row_offset + n_index) * VecLength);

    int out_idx = (threadIdx.x / 4) * 8 + threadIdx.x % 4 - (threadIdx.x / 16) * 28;

    if (out_idx < nonzeros) output_values_[out_idx] = reinterpret_cast<StoreType *>(output_fragment)[0];
}


// In this version, we try to use the mma to avoid shared memory access and sync barriers
template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=4, int Tile_X=32, int Tile_K=64>
__global__ void mmaSddmmKernel4reg(int m_vec, int k, int n, 
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
            acc_frag_half[i] = (OutType)acc_frag[0][i];
        }
    }

    StoreType *acc_frag_float4 = reinterpret_cast<StoreType *>(acc_frag);

    int out_col_id = octet_id * 8 + high_group * 4 + lane_id;

    StoreType *output_values_ = reinterpret_cast<StoreType *>(output_values + (row_offset + n_index + out_col_id) * VecLength);

    if (out_col_id < nonzeros){
        *(output_values_) = *(acc_frag_float4);
    }
}


// In this version, we try to use the mma to avoid shared memory access and sync barriers
template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=4, int Tile_X=32, int Tile_K=64>
__global__ void mmaSddmmKernel4shfl(int m_vec, int k, int n, 
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

    float acc_frag[4][4] = {0};
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

    int oprand_src_line = (threadIdx.x + 16) % 32;

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
            #pragma unroll
            for (int i=0; i < 4; i++){
                rhs_fragment_int[(x_item_idx * 2 + 1) * 4 + i] = __shfl_sync(0xffffffff, rhs_fragment_int[(x_item_idx * 2 + 1) * 4 + i], oprand_src_line, 32);
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
                "+f"(acc_frag[x_item_idx][0]), "+f"(acc_frag[x_item_idx][1]), 
                "+f"(acc_frag[x_item_idx][2]), "+f"(acc_frag[x_item_idx][3]), 
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
                "+f"(acc_frag[x_item_idx][0]), "+f"(acc_frag[x_item_idx][1]), 
                "+f"(acc_frag[x_item_idx][2]), "+f"(acc_frag[x_item_idx][3]), 
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
        for (int i = 0; i < 8; i ++){
            acc_frag_half[i] = (OutType)acc_frag[0][i];
        }
    }


    StoreType *acc_frag_float4 = reinterpret_cast<StoreType *>(acc_frag);

    int out_col_id = octet_id * 8 + high_group * 4 + lane_id;

    StoreType *output_values_ = reinterpret_cast<StoreType *>(output_values + (row_offset + n_index + out_col_id) * VecLength);

    if (out_col_id < nonzeros){
        *(output_values_) = *(acc_frag_float4);
    }
}


// In this version, we try to use the mma to avoid shared memory access and sync barriers
template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=4, int Tile_X=32, int Tile_K=64>
__global__ void mmaSddmmKernel4fake(int m_vec, int k, int n, 
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

    float acc_frag[4][4] = {0};
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
                "+f"(acc_frag[x_item_idx][0]), "+f"(acc_frag[x_item_idx][1]), 
                "+f"(acc_frag[x_item_idx][2]), "+f"(acc_frag[x_item_idx][3]), 
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
                "+f"(acc_frag[x_item_idx][0]), "+f"(acc_frag[x_item_idx][1]), 
                "+f"(acc_frag[x_item_idx][2]), "+f"(acc_frag[x_item_idx][3]), 
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
        for (int i = 0; i < 8; i++){
            acc_frag_half[i] = (OutType)acc_frag[0][i];
        }
    }

    StoreType *acc_frag_float4 = reinterpret_cast<StoreType *>(acc_frag);

    int out_col_id = octet_id * 8 + high_group * 4 + lane_id;

    StoreType *output_values_ = reinterpret_cast<StoreType *>(output_values + (row_offset + n_index + out_col_id) * VecLength);

    if (out_col_id < nonzeros){
        *(output_values_) = *(acc_frag_float4);
    }
}



// In this kernel, we try to handle k=2 efficiently
template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=4, int Tile_X=32, int Tile_K=64>
__global__ void wmmaSddmmKernel2(int m_vec, int k, int n, 
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ lhs_matrix,
    const half* __restrict__ rhs_matrix,
    OutType* __restrict__ output_values)
{
    constexpr int kValuesPerLoad = sizeof(LoadType) / sizeof(half);
    constexpr int Tile_K_pad = Tile_K + 8;
    constexpr int Tile_K_pad_float4 = Tile_K_pad / 8;
    
    // Tile_Y = 1
    int m_index_vec = blockIdx.x;
    int n_index = blockIdx.y * Tile_X;

    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset;

    // If this thread block has no nonzeros in the row to process, exit early
    if (n_index >= nonzeros) return;

    // Calculate the number of nonzeros that this thread block processes
    nonzeros = min(nonzeros - n_index, Tile_X);

    // We use he shared memory here
    __shared__ float4 lhs_fragment[Tile_K_pad_float4 * VecLength];

    // Shared memory tile for the output column indices
    __shared__ int column_indices_tile_array[Tile_X];

    int* column_indices_tile = column_indices_tile_array;

    __shared__ float4 rhs_fragment[Tile_X * Tile_K_pad_float4];


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

    // When loading, we will use half8 (float4) to fill the Tile_K vectors
    constexpr int SubWarp = (Tile_K / 8) > 32 ? 32 : (Tile_K / 8);
    constexpr int NumSubWarp = 32 / SubWarp;
    // static_assert(VecLength % NumSubWarp == 0, "the veclength should be divisible by the number of subwarps");
    // We no longer assert the above argument. When Tile_K = 64, the SubWarp size is 8. And there are 4 Subwarps.
    // However, we only have 2 vectors to load.


    // The lhs fragment has VecLength rows, and each row has Tile_K entries
    // Each subwarp will load one row into the shared memory
    constexpr int kThreadItemsK_ = Tile_K / SubWarp / 8;
    const int subwarp_id = threadIdx.x / SubWarp;
    const int sublane_id = threadIdx.x % SubWarp;
    const LoadType *lhs_matrix_ = reinterpret_cast<const LoadType *>(lhs_matrix + (m_index + subwarp_id) * k) + sublane_id;
    float4 *lhs_fragment_ = reinterpret_cast<float4 *>(lhs_fragment + subwarp_id * Tile_K_pad_float4) + sublane_id;

    constexpr int VecLoadStep = sizeof(float4) / sizeof(LoadType);

    // Similarly, the when loading the rhs matrix, each subwarp will load one column into the shared memory
    const LoadType *rhs_matrix_base = reinterpret_cast<const LoadType *>(rhs_matrix) + sublane_id;
    float4 *rhs_fragment_ = reinterpret_cast<float4 *>(rhs_fragment + subwarp_id * Tile_K_pad_float4) + sublane_id;

    // The fragments for wmma
    wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> rhs_frag;
    wmma::fragment<wmma::matrix_b, 32, 8, 16, half, wmma::col_major> lhs_frag;
    wmma::fragment<wmma::accumulator, 32, 8, 16, float> acc_frag;

    int *lhs_frag_r = reinterpret_cast<int *>(lhs_frag.x);
    int *rhs_frag_r = reinterpret_cast<int *>(rhs_frag.x);

    // do the computation
    half *rhs_fragment_half = reinterpret_cast<half *>(rhs_fragment);

    float4 *lhs_frag_float4 = reinterpret_cast<float4 *>(lhs_frag.x);

    int lhs_vec_idx = threadIdx.x % 4 + (threadIdx.x / 16) * 4;
    lhs_vec_idx = (lhs_vec_idx * (Tile_K / 16)  + lhs_vec_idx / 2) % 16;

    // Fill the accumulator fragment
    wmma::fill_fragment(acc_frag, 0.0f);


    const int k_ = k;
    #pragma nounroll
    for (; k >= Tile_K; k -= Tile_K){
        // Loading the matrices should be the same
        // Load a tile from the dense lhs matrix into the shared memory
        if (subwarp_id < VecLength){
            #pragma unroll
            for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
                #pragma unroll
                for (int v_idx = 0; v_idx < VecLength; v_idx +=NumSubWarp){
                    LoadType lhs_reg[VecLoadStep];
                    #pragma unroll
                    for (int r_idx = 0; r_idx < VecLoadStep; r_idx ++){
                        lhs_reg[r_idx] = __ldg(lhs_matrix_ + v_idx * k_ / kValuesPerLoad + r_idx * SubWarp);
                    }
                    *(lhs_fragment_ + v_idx * Tile_K_pad_float4 + k_item_idx) = reinterpret_cast<float4 *>(lhs_reg)[0];
                }
                lhs_matrix_ += SubWarp * VecLoadStep;
            }
        }

        column_indices_tile_ = column_indices_tile + subwarp_id;

        // Load a tile from the dense lhs matrix into shared memory
        #pragma unroll
        for (int x_item_idx = 0; x_item_idx < Tile_X; x_item_idx += NumSubWarp){
            const LoadType *rhs_matrix_ = rhs_matrix_base + *column_indices_tile_;
            #pragma unroll
            for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
                LoadType rhs_reg[VecLoadStep];
                #pragma unroll
                for (int r_idx = 0; r_idx < VecLoadStep; r_idx ++){
                    rhs_reg[r_idx] = __ldg(rhs_matrix_ + r_idx * SubWarp);
                }
                *(rhs_fragment_ + x_item_idx * Tile_K_pad_float4 + k_item_idx) = reinterpret_cast<float4 *>(rhs_reg)[0];
                rhs_matrix_ += SubWarp * VecLoadStep;
            }
            column_indices_tile_ += NumSubWarp;
        }

        rhs_matrix_base += Tile_K / kValuesPerLoad;

        __syncthreads();

        #pragma unroll
        for (int k_item_idx = 0; k_item_idx < Tile_K / 4; k_item_idx += 16){
            lhs_frag_float4[0] = *(lhs_fragment + lhs_vec_idx + k_item_idx/8);
            lhs_frag_float4[1] = *(lhs_fragment + lhs_vec_idx + k_item_idx/8 + 1);
            //wmma::load_matrix_sync(lhs_frag, lhs_fragment_half + k_item_idx, Tile_K / 2);
            // Load the lhs matrix
            #pragma unroll
            for (int step = 0; step < 4; step ++){
                switch(step){
                    case 0:
                        wmma::load_matrix_sync(rhs_frag, rhs_fragment_half + k_item_idx, Tile_K_pad);
                        asm("wmma.mma.sync.aligned.row.col.m32n8k16.f32.f32 \t"
                            "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                            "{%8, %9, %10, %11, %12, %13, %14, %15}, \t"
                            "{%16, %17, %18, %19, %20, %21, %22, %23}, \t"
                            "{%0, %1, %2, %3, %4, %5, %6, %7}; " 
                            :"+f"(acc_frag.x[0]), "+f"(acc_frag.x[1]), "+f"(acc_frag.x[2]), "+f"(acc_frag.x[3]),
                            "+f"(acc_frag.x[4]), "+f"(acc_frag.x[5]), "+f"(acc_frag.x[6]), "+f"(acc_frag.x[7]):
                            "r"(rhs_frag_r[0]), "r"(rhs_frag_r[1]), "r"(rhs_frag_r[2]), "r"(rhs_frag_r[3]),
                            "r"(rhs_frag_r[4]), "r"(rhs_frag_r[5]), "r"(rhs_frag_r[6]), "r"(rhs_frag_r[7]),
                            "r"(lhs_frag_r[0]), "r"(lhs_frag_r[1]), "r"(lhs_frag_r[2]), "r"(lhs_frag_r[3]),
                            "r"(lhs_frag_r[4]), "r"(lhs_frag_r[5]), "r"(lhs_frag_r[6]), "r"(lhs_frag_r[7])
                        );
                        break;
                    case 1:
                        wmma::load_matrix_sync(rhs_frag, rhs_fragment_half + k_item_idx + Tile_K / 2, Tile_K_pad);
                        asm("wmma.mma.sync.aligned.row.col.m32n8k16.f32.f32 \t"
                            "{%1, %0, %3, %2, %4, %5, %6, %7}, \t"
                            "{%8, %9, %10, %11, %12, %13, %14, %15}, \t"
                            "{%16, %17, %18, %19, %20, %21, %22, %23}, \t"
                            "{%1, %0, %3, %2, %4, %5, %6, %7}; " 
                            :"+f"(acc_frag.x[0]), "+f"(acc_frag.x[1]), "+f"(acc_frag.x[2]), "+f"(acc_frag.x[3]),
                            "+f"(acc_frag.x[4]), "+f"(acc_frag.x[5]), "+f"(acc_frag.x[6]), "+f"(acc_frag.x[7]):
                            "r"(rhs_frag_r[0]), "r"(rhs_frag_r[1]), "r"(rhs_frag_r[2]), "r"(rhs_frag_r[3]),
                            "r"(rhs_frag_r[4]), "r"(rhs_frag_r[5]), "r"(rhs_frag_r[6]), "r"(rhs_frag_r[7]),
                            "r"(lhs_frag_r[0]), "r"(lhs_frag_r[1]), "r"(lhs_frag_r[2]), "r"(lhs_frag_r[3]),
                            "r"(lhs_frag_r[4]), "r"(lhs_frag_r[5]), "r"(lhs_frag_r[6]), "r"(lhs_frag_r[7])
                        );
                        break;
                    case 2:
                        wmma::load_matrix_sync(rhs_frag, rhs_fragment_half + k_item_idx + Tile_K / 4, Tile_K_pad);
                        asm("wmma.mma.sync.aligned.row.col.m32n8k16.f32.f32 \t"
                            "{%4, %1, %6, %3, %0, %5, %2, %7}, \t"
                            "{%8, %9, %10, %11, %12, %13, %14, %15}, \t"
                            "{%16, %17, %18, %19, %20, %21, %22, %23}, \t"
                            "{%4, %1, %6, %3, %0, %5, %2, %7}; " 
                            :"+f"(acc_frag.x[0]), "+f"(acc_frag.x[1]), "+f"(acc_frag.x[2]), "+f"(acc_frag.x[3]),
                            "+f"(acc_frag.x[4]), "+f"(acc_frag.x[5]), "+f"(acc_frag.x[6]), "+f"(acc_frag.x[7]):
                            "r"(rhs_frag_r[0]), "r"(rhs_frag_r[1]), "r"(rhs_frag_r[2]), "r"(rhs_frag_r[3]),
                            "r"(rhs_frag_r[4]), "r"(rhs_frag_r[5]), "r"(rhs_frag_r[6]), "r"(rhs_frag_r[7]),
                            "r"(lhs_frag_r[0]), "r"(lhs_frag_r[1]), "r"(lhs_frag_r[2]), "r"(lhs_frag_r[3]),
                            "r"(lhs_frag_r[4]), "r"(lhs_frag_r[5]), "r"(lhs_frag_r[6]), "r"(lhs_frag_r[7])
                        );
                        break;
                    case 3:
                        wmma::load_matrix_sync(rhs_frag, rhs_fragment_half + k_item_idx + 3 * Tile_K / 4, Tile_K_pad);
                        asm("wmma.mma.sync.aligned.row.col.m32n8k16.f32.f32 \t"
                            "{%5, %1, %7, %3, %4, %0, %6, %2}, \t"
                            "{%8, %9, %10, %11, %12, %13, %14, %15}, \t"
                            "{%16, %17, %18, %19, %20, %21, %22, %23}, \t"
                            "{%5, %1, %7, %3, %4, %0, %6, %2}; " 
                            :"+f"(acc_frag.x[0]), "+f"(acc_frag.x[1]), "+f"(acc_frag.x[2]), "+f"(acc_frag.x[3]),
                            "+f"(acc_frag.x[4]), "+f"(acc_frag.x[5]), "+f"(acc_frag.x[6]), "+f"(acc_frag.x[7]):
                            "r"(rhs_frag_r[0]), "r"(rhs_frag_r[1]), "r"(rhs_frag_r[2]), "r"(rhs_frag_r[3]),
                            "r"(rhs_frag_r[4]), "r"(rhs_frag_r[5]), "r"(rhs_frag_r[6]), "r"(rhs_frag_r[7]),
                            "r"(lhs_frag_r[0]), "r"(lhs_frag_r[1]), "r"(lhs_frag_r[2]), "r"(lhs_frag_r[3]),
                            "r"(lhs_frag_r[4]), "r"(lhs_frag_r[5]), "r"(lhs_frag_r[6]), "r"(lhs_frag_r[7])
                        );
                        break;
                }
            }
        }

        // __syncthreads();
    }

    // wmma::fill_fragment(acc_frag, 0.0f);

    // TODO Residual

    // Now each thread has 4 elements in acc_frag.x[0], [2], [4], [6]
    // We load them into a register.
    OutType output_fragment[2];
    float reg_share;
    if ((threadIdx.x / 2) % 2 == 0){
        reg_share = acc_frag.x[2];
    }
    else{
        reg_share = acc_frag.x[0];
    }
    // We use warp shuffle to exchange data within the registers
    int src_line = threadIdx.x - ((threadIdx.x / 2) % 2) * 4 + 2;

    reg_share = __shfl_sync(0xffffffff, reg_share, src_line, 32);

    if ((threadIdx.x / 2) % 2 == 0){
        output_fragment[0] = (OutType)acc_frag.x[0];
        output_fragment[1] = (OutType)reg_share;
    }
    else{
        output_fragment[0] = (OutType)reg_share;
        output_fragment[1] = (OutType)acc_frag.x[2];
    }

    StoreType *output_values_ = reinterpret_cast<StoreType *>(output_values + (row_offset + n_index) * VecLength);

    int out_idx = (threadIdx.x / 4) * 8 + threadIdx.x % 4 - (threadIdx.x / 16) * 28;

    if (out_idx < nonzeros) output_values_[out_idx] = reinterpret_cast<StoreType *>(output_fragment)[0];
}



// In this version, we try to use the mma to avoid shared memory access and sync barriers
template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=2, int Tile_X=32, int Tile_K=64>
__global__ void mmaSddmmKernel2reg(int m_vec, int k, int n, 
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
            acc_frag_half[i] = (OutType)acc_frag[0][i];
        }
    }

    StoreType *acc_frag_float2 = reinterpret_cast<StoreType *>(acc_frag);

    int out_col_id = octet_id * 8 + high_group * 4 + lane_id;

    StoreType *output_values_ = reinterpret_cast<StoreType *>(output_values + (row_offset + n_index + out_col_id) * VecLength);


    if (out_col_id < nonzeros){
        *(output_values_) = *(acc_frag_float2);
    }
}


// In this version, we try to use the mma to avoid shared memory access and sync barriers
template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=2, int Tile_X=32, int Tile_K=64>
__global__ void mmaSddmmKernel2shfl(int m_vec, int k, int n, 
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

    float acc_frag[4][4] = {0};
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

    int oprand_src_line = (threadIdx.x + 16) % 32;

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
            #pragma unroll
            for (int i=0; i < 4; i++){
                rhs_fragment_int[(x_item_idx * 2 + 1) * 4 + i] = __shfl_sync(0xffffffff, rhs_fragment_int[(x_item_idx * 2 + 1) * 4 + i], oprand_src_line, 32);
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
                "+f"(acc_frag[x_item_idx][0]), "+f"(acc_frag[x_item_idx][1]), 
                "+f"(acc_frag[x_item_idx][2]), "+f"(acc_frag[x_item_idx][3]), 
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
                "+f"(acc_frag[x_item_idx][0]), "+f"(acc_frag[x_item_idx][1]), 
                "+f"(acc_frag[x_item_idx][2]), "+f"(acc_frag[x_item_idx][3]), 
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
            // float temp2 = __shfl_up_sync(mask, acc_frag[x_item_idx][acc_idx + 6], 2);
            if (lane_id > 1){
                acc_frag[x_item_idx][acc_idx] = temp1;
                // acc_frag[x_item_idx][acc_idx + 4] = temp2;
            }
        }
    }

    __syncwarp();
    /*
    int src_line = (threadIdx.x + 16) % 32;
    #pragma unroll
    for (int x_item_idx = 0; x_item_idx < 4; x_item_idx ++){
        #pragma unroll
        for (int acc_idx = 0; acc_idx < 2; acc_idx ++){
            acc_frag[x_item_idx][acc_idx] += __shfl_sync(0xffffffff, acc_frag[x_item_idx][(acc_idx + 4) % 8], src_line, 32);
        }
    }
    */

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
        for (int i = 0; i < 2; i++){
            acc_frag_half[i] = (OutType)acc_frag[0][i];
        }
    }

    StoreType *acc_frag_float2 = reinterpret_cast<StoreType *>(acc_frag);

    int out_col_id = octet_id * 8 + high_group * 4 + lane_id;

    StoreType *output_values_ = reinterpret_cast<StoreType *>(output_values + (row_offset + n_index + out_col_id) * VecLength);


    if (out_col_id < nonzeros){
        *(output_values_) = *(acc_frag_float2);
    }
}


// In this version, we try to use the mma to avoid shared memory access and sync barriers
template <typename LoadType, typename OutType, typename StoreType, int Residual=false, int VecLength=2, int Tile_X=32, int Tile_K=64>
__global__ void mmaSddmmKernel2fake(int m_vec, int k, int n, 
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

    float acc_frag[4][4] = {0};
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
                "+f"(acc_frag[x_item_idx][0]), "+f"(acc_frag[x_item_idx][1]), 
                "+f"(acc_frag[x_item_idx][2]), "+f"(acc_frag[x_item_idx][3]), 
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
                "+f"(acc_frag[x_item_idx][0]), "+f"(acc_frag[x_item_idx][1]), 
                "+f"(acc_frag[x_item_idx][2]), "+f"(acc_frag[x_item_idx][3]), 
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
            // float temp2 = __shfl_up_sync(mask, acc_frag[x_item_idx][acc_idx + 6], 2);
            if (lane_id > 1){
                acc_frag[x_item_idx][acc_idx] = temp1;
                // acc_frag[x_item_idx][acc_idx + 4] = temp2;
            }
        }
    }

    __syncwarp();
    /*
    int src_line = (threadIdx.x + 16) % 32;
    #pragma unroll
    for (int x_item_idx = 0; x_item_idx < 4; x_item_idx ++){
        #pragma unroll
        for (int acc_idx = 0; acc_idx < 2; acc_idx ++){
            acc_frag[x_item_idx][acc_idx] += __shfl_sync(0xffffffff, acc_frag[x_item_idx][(acc_idx + 4) % 8], src_line, 32);
        }
    }
    */

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
        for (int i = 0; i < 2; i++){
            acc_frag_half[i] = (OutType)acc_frag[0][i];
        }
    }

    StoreType *acc_frag_float2 = reinterpret_cast<StoreType *>(acc_frag);

    int out_col_id = octet_id * 8 + high_group * 4 + lane_id;

    StoreType *output_values_ = reinterpret_cast<StoreType *>(output_values + (row_offset + n_index + out_col_id) * VecLength);


    if (out_col_id < nonzeros){
        *(output_values_) = *(acc_frag_float2);
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
