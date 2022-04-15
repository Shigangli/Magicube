#include "../include/wmma_spmm.cuh"
#include "spmm_utils/dense_tile.h"
#include "spmm_utils/sparse_tile.h"
#include "spmm_utils/compute_utils.h"
#include "spmm_utils/output_tile.h"
#include <stdio.h>
#include <mma.h>

using namespace nvcuda;

namespace spmm{

template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, typename StoreType, int Tile_N, 
          int Tile_K, int BlockWidth, int VecLength=8>
__global__ void wmmaSpmmKernel8(
    int m, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
{
    // For the wmma based implementation, we have Tile_M = 1
    int m_index_vec = blockIdx.x;
    int k_index = blockIdx.y * Tile_K;
    const int lane_id = threadIdx.x % 4;
    const int thread_group = threadIdx.x / 4;
    
    // Threads that work on different m-dim indices are independent
    // If we're out of bounds in the m-dimension we can just return
    if (m_index_vec >= m) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset_vec;

    // For VecLength=8, we don't need the memory aligner

    // Shared memory tiles for the lhs values and indices
    __shared__ float4 values_tile_array[VecLength * Tile_N];
    __shared__ int column_indices_tile_array[Tile_N];

    // Pointers to the shared memory tiles
    float4 * values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    wmmaSparseTile<LoadType, VecType, VecLength, Tile_N, BlockWidth> sparse_tile_loader(
        k, row_offset_vec, threadIdx.x, values, column_indices,
        values_tile, column_indices_tile
    );

    // Register fragment for the dense matrix values
    constexpr int kDenseFragmentSize = Tile_N / 4 * 8;

    __align__(16) half dense_matrix_fragment[kDenseFragmentSize];

    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile<LoadType, Tile_N, Tile_K, BlockWidth> dense_tile_loader(
        k, k_index, lane_id, thread_group, rhs_matrix, column_indices_tile, dense_matrix_fragment
    );

    // Accumulator registers for the output values.
    constexpr int kOutputFragmentSize = 16;
    __align__(16) float output_fragment[kOutputFragmentSize] = {};
    wmmaComputeUtils8<VecType, Tile_N> computer(values_tile, dense_matrix_fragment, output_fragment, lane_id, thread_group);

    //
    // Begin kernel main loop
    //

    constexpr int InnerSteps = Tile_N / 4;

    for (; nonzeros >= Tile_N; nonzeros -= Tile_N){
        sparse_tile_loader.Load();
        __syncthreads();
        #pragma unroll
        for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
            dense_tile_loader.LoadRow(n_group_idx);
        }
        __threadfence_block();
        #pragma unroll
        for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
            computer.TileMAC(n_group_idx);
        }
        __syncthreads();
    }
    asm("");

    sparse_tile_loader.ZeroTiles();
    __syncthreads();
    sparse_tile_loader.Residue(nonzeros);
    __syncthreads();
    
    int n_group_idx = 0;

    #pragma unroll
    for (; n_group_idx < InnerSteps; n_group_idx ++){
        if (nonzeros < 4) break;
        dense_tile_loader.LoadRow(n_group_idx);
        computer.TileMAC(n_group_idx);
        nonzeros -= 4;
    }
    asm("");

    dense_tile_loader.ResidueLoad(n_group_idx, nonzeros);
    computer.TileMACResidue(n_group_idx);

    wmmaOutputTile8<OutType, StoreType> output_tile_storer(lane_id, thread_group, m_index_vec, 
        k_index, k, output_fragment, output_matrix);
    output_tile_storer.Store();
    
}

template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_N, 
          int Tile_K, int BlockWidth, int VecLength=4>
__global__ void wmmaSpmmKernel4(
    int m, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
{
    // For the wmma based implementation, we have Tile_M = 1
    int m_index_vec = blockIdx.x;
    int k_index = blockIdx.y * Tile_K;
    const int lane_id = threadIdx.x % 4;
    const int thread_group = threadIdx.x / 4;

    // Threads that work on different m-dim indices are independent
    // If we're out of bounds in the m-dimension we can just return
    if (m_index_vec >= m) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset_vec;

    // Shared memory tiles for the lhs values and indices
    __shared__ float2 values_tile_array[VecLength * Tile_N];
    __shared__ int column_indices_tile_array[Tile_N];

    // Pointers to the shared memory tiles
    float2 * values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    wmmaSparseTile<LoadType, VecType, VecLength, Tile_N, BlockWidth> sparse_tile_loader(
        k, row_offset_vec, threadIdx.x, values, column_indices,
        values_tile, column_indices_tile
    );

    // Register fragment for the dense matrix values
    constexpr int kDenseFragmentSize = Tile_N / 4 * 8;

    __align__(16) half dense_matrix_fragment[kDenseFragmentSize];

    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile<LoadType, Tile_N, Tile_K, BlockWidth> dense_tile_loader(
        k, k_index, lane_id, thread_group, rhs_matrix, column_indices_tile, dense_matrix_fragment
    );


    // Accumulator registers for the output values.
    constexpr int kOutputFragmentSize = 8;
    __align__(16) float output_fragment[kOutputFragmentSize] = {};
    wmmaComputeUtils4<VecType, Tile_N> computer(values_tile, dense_matrix_fragment, output_fragment, lane_id, thread_group);

    //
    // Begin kernel main loop
    //

    constexpr int InnerSteps = Tile_N / 4;

    for (; nonzeros >= Tile_N; nonzeros -= Tile_N){
        sparse_tile_loader.Load();
        __syncthreads();
        #pragma unroll
        for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
            dense_tile_loader.LoadRow(n_group_idx);
        }
        __threadfence_block();
        #pragma unroll
        for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
            computer.TileMAC(n_group_idx);
        }
        __syncthreads();
    }
    
    sparse_tile_loader.ZeroTiles();
    __syncthreads();
    sparse_tile_loader.Residue(nonzeros);
    __syncthreads();

    int n_group_idx = 0;

    #pragma unroll
    for (; n_group_idx < InnerSteps; n_group_idx ++){
        if (nonzeros < 4) break;
        dense_tile_loader.LoadRow(n_group_idx);
        computer.TileMAC(n_group_idx);
        nonzeros -= 4;
    }
    asm("");

    dense_tile_loader.ResidueLoad(n_group_idx, nonzeros);
    computer.TileMACResidue(n_group_idx);

    wmmaOutputTile4<OutType> output_tile_storer(lane_id, thread_group, m_index_vec, k_index, k, output_fragment, output_matrix);
    output_tile_storer.Store();
}

template <typename LoadType, typename IndexType, typename VecType, typename OutType, int Tile_N, int Tile_K, int BlockWidth, int VecLength=2>
__global__ void wmmaSpmmKernel2(
    int m, int k, int n,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
{
    // For the wmma based implementation, we have Tile_M = 1
    int m_index_vec = blockIdx.x;
    int k_index = blockIdx.y * Tile_K;
    const int lane_id = threadIdx.x % 4;
    const int thread_group = threadIdx.x / 4;

    // Threads that work on different m-dim indices are independent
    // If we're out of bounds in the m-dimension we can just return
    if (m_index_vec >= m) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset_vec;

    // Shared memory tiles for the lhs values and indices
    __shared__ float values_tile_array[VecLength * Tile_N];
    __shared__ int column_indices_tile_array[Tile_N];

    // Pointers to the shared memory tiles
    float * values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    wmmaSparseTile<LoadType, VecType, VecLength, Tile_N, BlockWidth> sparse_tile_loader(
        k, row_offset_vec, threadIdx.x, values, column_indices,
        values_tile, column_indices_tile
    );

    // Register fragment for the dense matrix values
    constexpr int kDenseFragmentSize = Tile_N / 4 * 8;

    __align__(16) half dense_matrix_fragment[kDenseFragmentSize];

    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile<LoadType, Tile_N, Tile_K, BlockWidth> dense_tile_loader(
        k, k_index, lane_id, thread_group, rhs_matrix, column_indices_tile, dense_matrix_fragment
    );

    // Accumulator registers for the output values.
    constexpr int kOutputFragmentSize = 4;
    __align__(16) float output_fragment[kOutputFragmentSize] = {};
    wmmaComputeUtils2<VecType, Tile_N> computer(values_tile, dense_matrix_fragment, output_fragment, lane_id, thread_group);

    //
    // Begin kernel main loop
    //

    constexpr int InnerSteps = Tile_N / 4;

    for (; nonzeros >= Tile_N; nonzeros -= Tile_N){
        sparse_tile_loader.Load();
        __syncthreads();
        #pragma unroll
        for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
            dense_tile_loader. LoadRow(n_group_idx);
        }
        __threadfence_block();
        #pragma unroll
        for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
            computer.TileMAC(n_group_idx);
        }
        __syncthreads();
    }

    sparse_tile_loader.ZeroTiles();
    __syncthreads();
    sparse_tile_loader.Residue(nonzeros);
    __syncthreads();

    int n_group_idx = 0;
    #pragma unroll
    for (; n_group_idx < InnerSteps; n_group_idx ++){
        if (nonzeros < 4) break;
        dense_tile_loader.LoadRow(n_group_idx);
        computer.TileMAC(n_group_idx);
        nonzeros -= 4;
    }
    asm("");

    dense_tile_loader.ResidueLoad(n_group_idx, nonzeros);
    computer.TileMACResidue(n_group_idx);

    wmmaOutputTile2<OutType> output_tile_storer(lane_id, thread_group, m_index_vec, k_index, k, output_fragment, output_matrix);
    output_tile_storer.Store();
}


template <typename LoadType, typename IndexType, int Tile_M, int Tile_N, int Tile_K, int BlockWidth>
cudaError_t wmmaSpmmEx(
    int m_vec, int vec_length, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(k) / Tile_K), 1);
    dim3 block_dim(BlockWidth, Tile_M, 1);
    switch(vec_length){
        case 2:
            wmmaSpmmKernel2<float4, int, float, float, Tile_N, Tile_K, BlockWidth, 2><<<grid_dim, block_dim>>>(
                m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
            break;
        case 4:
            wmmaSpmmKernel4<float4, int, float2, float, Tile_N, Tile_K, BlockWidth, 4><<<grid_dim, block_dim>>>(
                m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
            break;
        case 8:
            wmmaSpmmKernel8<float4, int, float4, float, float4, Tile_N, Tile_K, BlockWidth, 8><<<grid_dim, block_dim>>>(
                m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
            break;
        default:
            printf("Unsupported Vector Length!\n");
    }

    return cudaGetLastError();
}

template <typename LoadType, typename IndexType, int Tile_M, int Tile_N, int Tile_K, int BlockWidth>
cudaError_t wmmaSpmmEx(
    int m_vec, int vec_length, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(k) / Tile_K), 1);
    dim3 block_dim(BlockWidth, Tile_M, 1);
    switch(vec_length){
        case 2:
            wmmaSpmmKernel2<float4, int, float, half, Tile_N, Tile_K, BlockWidth, 2><<<grid_dim, block_dim>>>(
                m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
            break;
        case 4:
            wmmaSpmmKernel4<float4, int, float2, half, Tile_N, Tile_K, BlockWidth, 4><<<grid_dim, block_dim>>>(
                m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
            break;
        case 8:
            wmmaSpmmKernel8<float4, int, float4, half, float2, Tile_N, Tile_K, BlockWidth, 8><<<grid_dim, block_dim>>>(
                m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
            break;
        default:
            printf("Unsupported Vector Length!\n");
    }

    return cudaGetLastError();
}

// Function for mixed precision
cudaError_t wmmaSpmm(int m_vec, int vec_length, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix)
{
    return wmmaSpmmEx<float4, int, 1, 32, 64, 32>(m_vec, vec_length, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
}

// Function for half precision
cudaError_t wmmaSpmm(int m_vec, int vec_length, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix)
{
    return wmmaSpmmEx<float4, int, 1, 32, 64, 32>(m_vec, vec_length, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
}

// Function for single precision
cudaError_t wmmaSpmm(int m_vec, int vec_length, int k, int n,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const float* __restrict__ values,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix)
{
    printf("wmmaSpmm doesn't support float input.\n");
    return cudaSuccess;
}

}
