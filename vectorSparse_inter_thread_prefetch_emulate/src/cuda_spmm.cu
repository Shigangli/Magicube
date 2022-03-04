#include "../include/cuda_spmm.cuh"
#include "spmm_utils/barrier.h"
#include "spmm_utils/dense_tile.h"
#include "spmm_utils/memory_aligner.h"
#include "spmm_utils/sparse_tile.h"
#include "spmm_utils/compute_utils.h"
#include "spmm_utils/output_tile.h"
#include <stdio.h>


namespace spmm{

template <typename LoadType, typename IndexType, typename VecType, typename OutType, int VecLength, int Tile_M, int Tile_N, int Tile_K, int BlockWidth>
__global__ void cudaSpmmKernel(
    int m, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
{
    // Calculate this thread block's indices into the M and N dimensions
    int m_index_vec = blockIdx.x * Tile_M + threadIdx.y;
    int k_index = blockIdx.y * Tile_K;

    // Threads that work on different m-dim indices are independent.
    // If we're out of bounds in the m-dimension we can just return
    if (m_index_vec >= m) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset_vec;
    
    // Memory Aligner
    MemoryAligner<LoadType, BlockWidth, VecLength> memory_aligner(row_offset_vec, nonzeros);
    int aligned_nonzeros = memory_aligner.AlignedNonzeros();
    // The idea is to not align the row_offset when there is only residual to handle
    if (aligned_nonzeros >= Tile_N){
        nonzeros = aligned_nonzeros;
        row_offset_vec = memory_aligner.AlignedRowOffset();
    }
    
    // Shared memory tiles for the lhs values and indices
    __shared__ half values_tile_array[VecLength * Tile_N * Tile_M];
    __shared__ int column_indices_tile_array[Tile_N * Tile_M];

    // Pointers to the shared memory tiles
    half* values_tile = values_tile_array + VecLength * Tile_N * threadIdx.y;
    int* column_indices_tile = column_indices_tile_array + Tile_N * threadIdx.y;
    
    // Initialize the pointers to the sparse lhs matrix
    SparseTile<LoadType, IndexType, VecType, VecLength, Tile_N, BlockWidth> sparse_tile_loader(
        k, row_offset_vec, threadIdx.x, values, column_indices, 
        values_tile, column_indices_tile
    );
    
    // Register fragment for the dense_matrix values
    constexpr int kDenseFragmentSize = Tile_N * Tile_K / BlockWidth;

    __align__(16) half dense_matrix_fragment[kDenseFragmentSize];

    // Initialize the pointers to the dense rhs matrix
    DenseTile<LoadType, VecType, Tile_N, Tile_K, BlockWidth, VecLength> dense_tile_loader(
        k, k_index, threadIdx.x, rhs_matrix, column_indices_tile, dense_matrix_fragment);
    
    // Accumulator registers for the output values. 
    // Initialize the registers to zero s.t. we can always accumulate in-place.
    constexpr int kOutputFragmentSize = VecLength * Tile_K / BlockWidth;
    __align__(16) float output_fragment[kOutputFragmentSize] = {};
    
    ComputeUtils<VecType, Tile_N, Tile_K, BlockWidth, VecLength> computer(values_tile, dense_matrix_fragment, output_fragment);
    // Barrier
    Barrier<Tile_M, BlockWidth> barrier(threadIdx.y);

    
    //
    // Begin kernel main loop
    //

    // For the first iteration of our main loop, we need to possibly mask
    // the first few values from the sparse matrix in case we aligned our 
    // values and column indices pointers.
    if (nonzeros >= Tile_N){
        // Load a tile from the sparse lhs matrix and synchronize the cta.
        sparse_tile_loader.Load();
        barrier.Sync();

        memory_aligner.MaskPrefix(values_tile, column_indices_tile);
        barrier.Sync();

        dense_tile_loader.Load();
        computer.TileMAC();
        nonzeros -= Tile_N;
    }
    
    // Loop over the tiles in the n-dimension of the dense_matrix/lhs matrix
    for (; nonzeros >= Tile_N; nonzeros -= Tile_N){
        barrier.Sync();
        sparse_tile_loader.Load();
        barrier.Sync();
        dense_tile_loader.Load();
        computer.TileMAC();
    }
    
    barrier.Sync();
    
    sparse_tile_loader.ZeroTiles();
    barrier.Sync();

    // Load a tile from the sparse lhs matrix and synchronize the cta
    sparse_tile_loader.Residue(nonzeros);
    barrier.Sync();

    dense_tile_loader.ResidueLoadAndCompute(nonzeros, values_tile, output_fragment);
    
    OutputTile<LoadType, OutType, Tile_K, BlockWidth, VecLength> output_tile_storer(m_index_vec, k_index, k, threadIdx.x, output_fragment, output_matrix);
    output_tile_storer.Store();
}



// This kernel is implemented for Tile_M = 1
// 
template <typename LoadType, typename IndexType, typename VecType, typename OutType, int VecLength, int Tile_M, int Tile_N, int Tile_K, int BlockWidth>
__global__ void cudaSpmmKernel1D(
    int m, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
{
    // Calculate this thread block's indices into the M and N dimensions
    int m_index_vec = blockIdx.x;
    int k_index = blockIdx.y * Tile_K;

    // Threads that work on different m-dim indices are independent.
    // If we're out of bounds in the m-dimension we can just return
    if (m_index_vec >= m) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset_vec;
    
    // We skip the memory aligner for this kernel
    // Memory Aligner
    // MemoryAligner<LoadType, BlockWidth, VecLength> memory_aligner(row_offset_vec, nonzeros);
    // int aligned_nonzeros = memory_aligner.AlignedNonzeros();
    // The idea is to not align the row_offset when there is only residual to handle
    // if (aligned_nonzeros >= Tile_N){
    //     nonzeros = aligned_nonzeros;
    //     row_offset_vec = memory_aligner.AlignedRowOffset();
    // }
    
    // Shared memory tiles for the lhs values and indices
    __shared__ half values_tile_array[VecLength * Tile_N * Tile_M];
    __shared__ int column_indices_tile_array[Tile_N * Tile_M];

    // Pointers to the shared memory tiles
    VecType* values_tile = reinterpret_cast<VecType* >(values_tile_array + VecLength * Tile_N * threadIdx.y);
    int* column_indices_tile = column_indices_tile_array + Tile_N * threadIdx.y;
    
    // Initialize the pointers to the sparse lhs matrix
    SparseTile1D<LoadType, VecType, VecLength, Tile_N, BlockWidth> sparse_tile_loader(
        k, row_offset_vec, threadIdx.x, values, column_indices, 
        values_tile, column_indices_tile
    );
    
    // Register fragment for the dense_matrix values
    constexpr int kDenseFragmentSize = Tile_N * Tile_K / BlockWidth;

    __align__(16) half dense_matrix_fragment[kDenseFragmentSize];

    // Initialize the pointers to the dense rhs matrix
    DenseTile1D<LoadType, VecType, Tile_N, Tile_K, BlockWidth, VecLength> dense_tile_loader(
        k, k_index, threadIdx.x, rhs_matrix, column_indices_tile, dense_matrix_fragment);
    
    // Accumulator registers for the output values. 
    // Initialize the registers to zero s.t. we can always accumulate in-place.
    constexpr int kOutputFragmentSize = VecLength * Tile_K / BlockWidth;
    __align__(16) float output_fragment[kOutputFragmentSize] = {};
    
    ComputeUtils1D<VecType, Tile_N, Tile_K, BlockWidth, VecLength> computer(values_tile, dense_matrix_fragment, output_fragment);
    // Barrier
    Barrier<Tile_M, BlockWidth> barrier(threadIdx.y);

    
    //
    // Begin kernel main loop
    //

    // For the first iteration of our main loop, we need to possibly mask
    // the first few values from the sparse matrix in case we aligned our 
    // values and column indices pointers.
    if (nonzeros >= Tile_N){
        // Load a tile from the sparse lhs matrix and synchronize the cta.
        sparse_tile_loader.Load();
        barrier.Sync();

        // memory_aligner.MaskPrefix(values_tile, column_indices_tile);
        barrier.Sync();

        dense_tile_loader.Load();
        computer.TileMAC();
        nonzeros -= Tile_N;
    }
    
    // Loop over the tiles in the n-dimension of the dense_matrix/lhs matrix
    for (; nonzeros >= Tile_N; nonzeros -= Tile_N){
        barrier.Sync();
        sparse_tile_loader.Load();
        barrier.Sync();
        dense_tile_loader.Load();
        computer.TileMAC();
    }
    
    barrier.Sync();
    
    sparse_tile_loader.ZeroTiles();
    barrier.Sync();

    // Load a tile from the sparse lhs matrix and synchronize the cta
    sparse_tile_loader.Residue(nonzeros);
    barrier.Sync();

    dense_tile_loader.ResidueLoadAndCompute(nonzeros, values_tile, output_fragment);
    
    OutputTile<LoadType, OutType, Tile_K, BlockWidth, VecLength> output_tile_storer(m_index_vec, k_index, k, threadIdx.x, output_fragment, output_matrix);
    output_tile_storer.Store();
}




template <typename LoadType, typename IndexType, typename OutType, int Tile_M, int Tile_N, int Tile_K, int BlockWidth>
cudaError_t cudaSpmmEx(
    int m_vec, int vec_length, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
{
    dim3* grid_dim;
    dim3* block_dim;

    switch(vec_length){
        case 1: 
            printf("V=1\n");
            grid_dim = new dim3(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(k) / Tile_K), 1);
            block_dim = new dim3(BlockWidth, Tile_M, 1);
            cudaSpmmKernel<LoadType, IndexType, half, OutType, 1, Tile_M, Tile_N, Tile_K, BlockWidth><<<*grid_dim, *block_dim>>>(
                m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
            break;
        case 2:
            printf("V=2\n");
            // grid_dim = new dim3(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(k) / Tile_K), 1);
            // block_dim = new dim3(BlockWidth, Tile_M, 1);
            // cudaSpmmKernel<LoadType, int, half2, OutType, 2, Tile_M, Tile_N, Tile_K, BlockWidth><<<grid_dim, block_dim>>>(
            //     m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
            
            grid_dim = new dim3(ceil(static_cast<float>(m_vec) / 1), ceil(static_cast<float>(k) / Tile_K), 1);
            block_dim = new dim3(32, 1, 1);
            cudaSpmmKernel<float, int, half2, OutType, 2, 1, Tile_N, Tile_K, 32><<<*grid_dim, *block_dim>>>(
                m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
            break;
        case 4:
            printf("V=4\n");
            // grid_dim = new dim3(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(k) / Tile_K), 1);
            // block_dim = new dim3(BlockWidth, Tile_M, 1);
            // cudaSpmmKernel<LoadType, int, float2, OutType, 4, Tile_M, Tile_N, Tile_K, BlockWidth><<<*grid_dim, *block_dim>>>(
            //   m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
            
            // Use 1D Tile
            grid_dim = new dim3(ceil(static_cast<float>(m_vec) / 1), ceil(static_cast<float>(k) / Tile_K), 1);
            block_dim = new dim3(32, 1, 1);
            cudaSpmmKernel1D<float, int, float2, OutType, 4, 1, Tile_N, Tile_K, 32><<<*grid_dim, *block_dim>>>(
                m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
            break;
        case 8:
            printf("V=8\n");
            // grid_dim = new dim3(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(k) / Tile_K), 1);
            // block_dim = new dim3(BlockWidth, Tile_M, 1);
            // cudaSpmmKernel<LoadType, int, float4, OutType, 8, Tile_M, Tile_N, Tile_K, BlockWidth><<<*grid_dim, *block_dim>>>(
            //    m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
            
            grid_dim = new dim3(ceil(static_cast<float>(m_vec) / 1), ceil(static_cast<float>(k) / Tile_K), 1);
            block_dim = new dim3(32, 1, 1);
            cudaSpmmKernel1D<float, int, float4, OutType, 8, 1, Tile_N, Tile_K, 32><<<*grid_dim, *block_dim>>>(
                m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
            break;
        default:
            printf("Unsupported Vector Length!\n");
    }

    return cudaGetLastError();
}

cudaError_t cudaSpmm(int m_vec, int vec_length, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix)
{
    return cudaSpmmEx<float4, int4, float, 4, 64, 64, 8>(m_vec, vec_length, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
}


cudaError_t cudaSpmm(int m_vec, int vec_length, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix)
{
    return cudaSpmmEx<float4, int4, half, 4, 64, 64, 8>(m_vec, vec_length, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
}


cudaError_t cudaSpmm(int m_vec, int vec_length, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const float* __restrict__ values,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix)
{
    // TODO: the kernel
    return cudaSuccess;
}

}