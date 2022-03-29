#include <cuda.h>
#include "cuda_fp16.h"
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include "spmm_utils/dense_tile.h"
#include "spmm_utils/sparse_tile.h"
#include "spmm_utils/compute_utils.h"
#include "spmm_utils/output_tile.h"
#include <stdio.h>
#include <mma.h>

using namespace nvcuda;

//4-bit Tile_N = 64 with 2 warps
template <typename LoadType, typename IndexType, typename VecType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__device__ void wmmaSpmm_kernel_4b_(
    int m_vec, int dimN, int dimK, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix)
{
    // For the wmma based implementation, we have Tile_M = 1
    int m_index_vec = blockIdx.x;
    int dimN_index = blockIdx.y * Tile_N;
    const int lane_id = threadIdx.x;
    // Threads that work on different m-dim indices are independent
    // If we're out of bounds in the m-dimension we can just return
    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec*2);
    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset_vec;

    // Shared memory tiles for the lhs values and indices
    // Each int32 has 8 4-bit integers with double buffers
    __shared__ int values_tile_array[Tile_K*VecLength/4];
    __shared__ int column_indices_tile_array[Tile_K*2];

    //padding to avoid bank conflict 
    __shared__ int dense_tile_array[Tile_N*Tile_K/8 + 8*3];

    // Pointers to the shared memory tiles
    int* values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;
    int* dense_tile = dense_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    // One int32 has eight 4-bit integers
    wmmaSparseTile_4b<LoadType, VecType, Tile_K * VecLength / 8, Tile_K> sparse_tile_loader(
        row_offset_vec, threadIdx.x % 32, threadIdx.x / 32, values, column_indices,
        values_tile, column_indices_tile
    );

    __align__(16) int rhs_prefetch[4] = {};
    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile_4b<LoadType, Tile_K, Tile_N> dense_tile_loader(
        dimN/8, dimN_index/8, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    // Tile_N / warps / four threads in x-dim of output matrix
    __align__(16) int output_fragment[Tile_N / Warps / 4] = {};
    wmmaComputeUtils_4b<Tile_K * VecLength / 8> computer(values_tile, dense_tile, output_fragment, lane_id);

    int steps = nonzeros / Tile_K;
    int residue = nonzeros % Tile_K;

    if(steps > 0){
        sparse_tile_loader.Load(0);
        __syncthreads();
        dense_tile_loader.Prefetch(0);

        int i = 1;
        #pragma unroll
        for(; i < steps; i++){
            dense_tile_loader.LoadRowfromRegister(i-1);
            sparse_tile_loader.Load(i);
            __syncthreads();
            dense_tile_loader.Prefetch(i);
            computer.TileMAC(i-1);
            __syncthreads();
        }

        dense_tile_loader.LoadRowfromRegister(i-1);
        __syncthreads();
        computer.TileMAC(i-1);
    }
   
    if(residue > 0){
        sparse_tile_loader.Residue();
        __syncthreads();
        dense_tile_loader.ResidueLoad(residue);
        __syncthreads();
        computer.TileMACResidue();
    } 

    wmmaOutputTile_4b output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment, output_matrix, scale);
    output_tile_storer.Store();
}

//8-bit Tile_N = 64 with 2 warps
template <typename LoadType, typename IndexType, typename VecType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__device__ void wmmaSpmm_kernel_8b_(
    int m_vec, int dimN, int dimK, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix)
{
    // For the wmma based implementation, we have Tile_M = 1
    int m_index_vec = blockIdx.x;
    int dimN_index = blockIdx.y * Tile_N;
    const int lane_id = threadIdx.x;
    // Threads that work on different m-dim indices are independent
    // If we're out of bounds in the m-dimension we can just return
    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec*2);
    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset_vec;

    // Shared memory tiles for the lhs values and indices
    __shared__ int values_tile_array[Tile_K*VecLength/2];
    __shared__ int column_indices_tile_array[Tile_K*2];

    // One int32 has four 8-bit integers
    // Padding to avoid bank conflict 
    __shared__ int dense_tile_array[Tile_N*Tile_K/4 + 8*3];

    // Pointers to the shared memory tiles
    int* values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;
    int* dense_tile = dense_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    // One int32 has four 8-bit integers
    wmmaSparseTile_8b<LoadType, VecType, Tile_K * VecLength / 4, Tile_K> sparse_tile_loader(
        row_offset_vec, threadIdx.x % 32, threadIdx.x / 32, values, column_indices,
        values_tile, column_indices_tile
    );

    __align__(16) int rhs_prefetch[4] = {};
    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile_8b<LoadType, Tile_K, Tile_N> dense_tile_loader(
        dimN/4, dimN_index/4, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    // Tile_N / warps / four threads in x-dim of output matrix
    __align__(16) int output_fragment[Tile_N / Warps / 4] = {};
    wmmaComputeUtils_8b<Tile_K * VecLength / 4> computer(values_tile, dense_tile, output_fragment, lane_id);

    int steps = nonzeros / Tile_K;
    int residue = nonzeros % Tile_K;

    if(steps > 0){
        sparse_tile_loader.Load(0);
        __syncthreads();
        dense_tile_loader.Prefetch(0);

        int i = 1;
        #pragma unroll
        for(; i < steps; i++){
            dense_tile_loader.LoadRowfromRegister(i-1);
            sparse_tile_loader.Load(i);
            __syncthreads();
            dense_tile_loader.Prefetch(i);
            computer.TileMAC(i-1);
            __syncthreads();
        }

        dense_tile_loader.LoadRowfromRegister(i-1);
        __syncthreads();
        computer.TileMAC(i-1);
    }
   
    if(residue > 0){
        sparse_tile_loader.Residue();
        __syncthreads();
        dense_tile_loader.ResidueLoad(residue);
        __syncthreads();
        computer.TileMACResidue();
    } 

    wmmaOutputTile_8b output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment, output_matrix, scale);
    output_tile_storer.Store();
}

//16-bit 8-bit Tile_N = 64 with 2 warps
template <typename LoadType, typename IndexType, typename VecType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__device__ void wmmaSpmm_kernel_16b8b_(
    int m_vec, int dimN, int dimK, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix)
{
    // For the wmma based implementation, we have Tile_M = 1
    int m_index_vec = blockIdx.x;
    int dimN_index = blockIdx.y * Tile_N;
    const int lane_id = threadIdx.x;
    // Threads that work on different m-dim indices are independent
    // If we're out of bounds in the m-dimension we can just return
    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec*2);
    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset_vec;

    // Shared memory tiles for the lhs values and indices
    __shared__ int values_tile_array[Tile_K*VecLength];
    __shared__ int column_indices_tile_array[Tile_K*2];

    // One int32 has four 8-bit integers
    // Padding to avoid bank conflict 
    __shared__ int dense_tile_array[Tile_N*Tile_K/4 + 8*3];

    // Pointers to the shared memory tiles
    int* values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;
    int* dense_tile = dense_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    // One int32 has two 16-bit integers
    wmmaSparseTile_16b8b<LoadType, VecType, Tile_K * VecLength / 2, Tile_K> sparse_tile_loader(
        row_offset_vec, threadIdx.x % 32, threadIdx.x / 32, values, column_indices,
        values_tile, column_indices_tile
    );

    __align__(16) int rhs_prefetch[4] = {};
    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile_8b<LoadType, Tile_K, Tile_N> dense_tile_loader(
        dimN/4, dimN_index/4, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    // Tile_N / warps / four threads in x-dim of output matrix
    __align__(16) int output_fragment[Tile_N / Warps / 4] = {};
    wmmaComputeUtils_16b8b<Tile_K * VecLength / 2> computer(values_tile, dense_tile, output_fragment, lane_id);

    int steps = nonzeros / Tile_K;
    int residue = nonzeros % Tile_K;

    if(steps > 0){
        sparse_tile_loader.Load(0);
        __syncthreads();
        dense_tile_loader.Prefetch(0);

        int i = 1;
        #pragma unroll
        for(; i < steps; i++){
            dense_tile_loader.LoadRowfromRegister(i-1);
            sparse_tile_loader.Load(i);
            __syncthreads();
            dense_tile_loader.Prefetch(i);
            computer.TileMAC(i-1);
            __syncthreads();
        }

        dense_tile_loader.LoadRowfromRegister(i-1);
        __syncthreads();
        computer.TileMAC(i-1);
    }
   
    if(residue > 0){
        sparse_tile_loader.Residue();
        __syncthreads();
        dense_tile_loader.ResidueLoad(residue);
        __syncthreads();
        computer.TileMACResidue();
    } 

    wmmaOutputTile_16b8b output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment, output_matrix, scale);
    output_tile_storer.Store();
}


//16-bit 8-bit Tile_N = 64 with 2 warps 8v
template <typename LoadType, typename IndexType, typename VecType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__device__ void wmmaSpmm_kernel_16b8b8v_(
    int m_vec, int dimN, int dimK, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix)
{
    // For the wmma based implementation, we have Tile_M = 1
    int m_index_vec = blockIdx.x;
    int dimN_index = blockIdx.y * Tile_N;
    const int lane_id = threadIdx.x;
    // Threads that work on different m-dim indices are independent
    // If we're out of bounds in the m-dimension we can just return
    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec*2);
    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset_vec;

    // Shared memory tiles for the lhs values and indices
    __shared__ int values_tile_array[Tile_K*VecLength];
    __shared__ int column_indices_tile_array[Tile_K*2];

    // One int32 has four 8-bit integers
    // Padding to avoid bank conflict 
    __shared__ int dense_tile_array[Tile_N*Tile_K/4 + 8*3];

    // Pointers to the shared memory tiles
    int* values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;
    int* dense_tile = dense_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    // One int32 has two 16-bit integers
    wmmaSparseTile_16b8b8v<LoadType, VecType, Tile_K * VecLength / 2, Tile_K> sparse_tile_loader(
        row_offset_vec, lane_id, values, column_indices,
        values_tile, column_indices_tile
    );

    __align__(16) int rhs_prefetch[4] = {};
    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile_8b<LoadType, Tile_K, Tile_N> dense_tile_loader(
        dimN/4, dimN_index/4, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    // Tile_N / warps / four threads in x-dim of output matrix
    __align__(16) int output_fragment_0[Tile_N / Warps / 4] = {};
    __align__(16) int output_fragment_1[Tile_N / Warps / 4] = {};
    wmmaComputeUtils_16b8b8v<Tile_K * VecLength / 2> computer(values_tile, dense_tile, output_fragment_0, output_fragment_1, lane_id);

    int steps = nonzeros / Tile_K;
    int residue = nonzeros % Tile_K;

    if(steps > 0){
        sparse_tile_loader.Load(0);
        __syncthreads();
        dense_tile_loader.Prefetch(0);

        int i = 1;
        #pragma unroll
        for(; i < steps; i++){
            dense_tile_loader.LoadRowfromRegister(i-1);
            sparse_tile_loader.Load(i);
            __syncthreads();
            dense_tile_loader.Prefetch(i);
            computer.TileMAC(i-1);
            __syncthreads();
        }

        dense_tile_loader.LoadRowfromRegister(i-1);
        __syncthreads();
        computer.TileMAC(i-1);
    }
   
    if(residue > 0){
        sparse_tile_loader.Residue();
        __syncthreads();
        dense_tile_loader.ResidueLoad(residue);
        __syncthreads();
        computer.TileMACResidue();
    } 

    wmmaOutputTile_16b8b8v output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment_0, output_fragment_1, output_matrix, scale);
    output_tile_storer.Store();
}

//8-bit A 4-bit B Tile_N = 64 warps = 2
template <typename LoadType, typename IndexType, typename VecType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__device__ void wmmaSpmm_kernel_8b4b_(
    int m_vec, int dimN, int dimK, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix)
{
    // For the wmma based implementation, we have Tile_M = 1
    int m_index_vec = blockIdx.x;
    int dimN_index = blockIdx.y * Tile_N;
    const int lane_id = threadIdx.x;
    // Threads that work on different m-dim indices are independent
    // If we're out of bounds in the m-dimension we can just return
    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec*2);
    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset_vec;

    // Shared memory tiles for the lhs values and indices
    __shared__ int values_tile_array[Tile_K*VecLength/2];
    __shared__ int column_indices_tile_array[Tile_K*2];

    // each int value has four 4-bit values, padding to avoid bank conflict 
    __shared__ int dense_tile_array[Tile_N*Tile_K/8 + 8*3];

    // Pointers to the shared memory tiles
    int* values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;
    int* dense_tile = dense_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    wmmaSparseTile_8b4b<LoadType, VecType, Tile_K * VecLength / 4, Tile_K> sparse_tile_loader(
        row_offset_vec, threadIdx.x % 32, threadIdx.x / 32, values, column_indices,
        values_tile, column_indices_tile
    );

    __align__(16) int rhs_prefetch[4] = {};
    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile_4b<LoadType, Tile_K, Tile_N> dense_tile_loader(
        dimN/8, dimN_index/8, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    __align__(16) int output_fragment[Tile_N / Warps / 4] = {};
    //wmmaComputeUtils_8b4b<Tile_K * VecLength / 4> computer(values_tile, dense_tile, output_fragment, lane_id);
    wmmaComputeUtils_4b<Tile_K * VecLength / 4> computer(values_tile, dense_tile, output_fragment, lane_id);

    int steps = nonzeros / Tile_K;
    int residue = nonzeros % Tile_K;

    if(steps > 0){
        sparse_tile_loader.Load(0);
        __syncthreads();
        dense_tile_loader.Prefetch(0);

        int i = 1;
        #pragma unroll
        for(; i < steps; i++){
            dense_tile_loader.LoadRowfromRegister(i-1);
            sparse_tile_loader.Load(i);
            __syncthreads();
            dense_tile_loader.Prefetch(i);
            computer.TileMAC(i-1);
            __syncthreads();
        }

        dense_tile_loader.LoadRowfromRegister(i-1);
        __syncthreads();
        computer.TileMAC(i-1);
    }
   
    if(residue > 0){
        sparse_tile_loader.Residue();
        __syncthreads();
        dense_tile_loader.ResidueLoad(residue);
        __syncthreads();
        computer.TileMACResidue();
    } 

    wmmaOutputTile_8b4b output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment, output_matrix, scale);
    output_tile_storer.Store();
}


//8-bit A 4-bit B Tile_N = 64 warps = 2, 8v
template <typename LoadType, typename IndexType, typename VecType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__device__ void wmmaSpmm_kernel_8b4b8v_(
    int m_vec, int dimN, int dimK, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix)
{
    // For the wmma based implementation, we have Tile_M = 1
    int m_index_vec = blockIdx.x;
    int dimN_index = blockIdx.y * Tile_N;
    const int lane_id = threadIdx.x;
    // Threads that work on different m-dim indices are independent
    // If we're out of bounds in the m-dimension we can just return
    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec*2);
    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset_vec;

    // Shared memory tiles for the lhs values and indices
    __shared__ int values_tile_array[Tile_K*VecLength/2];
    __shared__ int column_indices_tile_array[Tile_K*2];

    // each int value has four 4-bit values, padding to avoid bank conflict 
    __shared__ int dense_tile_array[Tile_N*Tile_K/8 + 8*3];

    // Pointers to the shared memory tiles
    int* values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;
    int* dense_tile = dense_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    wmmaSparseTile_8b4b8v<LoadType, VecType, Tile_K * VecLength / 4, Tile_K> sparse_tile_loader(
        row_offset_vec, lane_id, values, column_indices,
        values_tile, column_indices_tile
    );

    __align__(16) int rhs_prefetch[4] = {};
    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile_4b<LoadType, Tile_K, Tile_N> dense_tile_loader(
        dimN/8, dimN_index/8, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    __align__(16) int output_fragment_0[Tile_N / Warps / 4] = {};
    __align__(16) int output_fragment_1[Tile_N / Warps / 4] = {};
    wmmaComputeUtils_8b4b8v<Tile_K * VecLength / 4> computer(values_tile, dense_tile, output_fragment_0, output_fragment_1, lane_id);

    int steps = nonzeros / Tile_K;
    int residue = nonzeros % Tile_K;

    if(steps > 0){
        sparse_tile_loader.Load(0);
        __syncthreads();
        dense_tile_loader.Prefetch(0);

        int i = 1;
        #pragma unroll
        for(; i < steps; i++){
            dense_tile_loader.LoadRowfromRegister(i-1);
            sparse_tile_loader.Load(i);
            __syncthreads();
            dense_tile_loader.Prefetch(i);
            computer.TileMAC(i-1);
            __syncthreads();
        }

        dense_tile_loader.LoadRowfromRegister(i-1);
        __syncthreads();
        computer.TileMAC(i-1);
    }
   
    if(residue > 0){
        sparse_tile_loader.Residue();
        __syncthreads();
        dense_tile_loader.ResidueLoad(residue);
        __syncthreads();
        computer.TileMACResidue();
    } 

    wmmaOutputTile_8b4b8v output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment_0, output_fragment_1, output_matrix, scale);
    output_tile_storer.Store();
}


//4-bit Tile_N = 64 with 2 warps
template <typename LoadType, typename IndexType, typename VecType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_4b(
    int m_vec, int dimN, int dimK, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix)
{
    wmmaSpmm_kernel_4b_<LoadType, IndexType, VecType, Tile_K, Tile_N, Warps, VecLength>(
    m_vec, dimN, dimK, scale,
    row_indices, 
    row_offsets,
    column_indices,
    values,
    rhs_matrix,
    output_matrix);
}

//4-bit Tile_N = 64 with 2 warps
template <typename LoadType, typename IndexType, typename VecType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void batched_wmmaSpmm_kernel_4b(
    int m_vec, int dimN, int dimK, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values_b,
    int values_stride,
    const int* __restrict__ rhs_matrix_b,
    int rhs_stride,
    half* __restrict__ output_matrix_b,
    int output_stride)
{
    int entry_idx = blockIdx.z;
    const VecType* values = values_b + entry_idx * values_stride;
    const int* rhs_matrix = rhs_matrix_b + entry_idx * rhs_stride;
    half* output_matrix = output_matrix_b + entry_idx * output_stride;

    wmmaSpmm_kernel_4b_<LoadType, IndexType, VecType, Tile_K, Tile_N, Warps, VecLength>(
    m_vec, dimN, dimK, scale,
    row_indices, 
    row_offsets,
    column_indices,
    values,
    rhs_matrix,
    output_matrix);
}


//8-bit Tile_N = 64 with 2 waprs
template <typename LoadType, typename IndexType, typename VecType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_8b(
    int m_vec, int dimN, int dimK, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix)
{
    wmmaSpmm_kernel_8b_<LoadType, IndexType, VecType, Tile_K, Tile_N, Warps, VecLength>(
    m_vec, dimN, dimK, scale,
    row_indices, 
    row_offsets,
    column_indices,
    values,
    rhs_matrix,
    output_matrix);
}

template <typename LoadType, typename IndexType, typename VecType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void batched_wmmaSpmm_kernel_8b(
    int m_vec, int dimN, int dimK, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values_b,
    int values_stride,
    const int* __restrict__ rhs_matrix_b,
    int rhs_stride,
    half* __restrict__ output_matrix_b,
    int output_stride)
{
    int entry_idx = blockIdx.z;
    const VecType* values = values_b + entry_idx * values_stride;
    const int* rhs_matrix = rhs_matrix_b + entry_idx * rhs_stride;
    half* output_matrix = output_matrix_b + entry_idx * output_stride;

    wmmaSpmm_kernel_8b_<LoadType, IndexType, VecType, Tile_K, Tile_N, Warps, VecLength>(
    m_vec, dimN, dimK, scale,
    row_indices, 
    row_offsets,
    column_indices,
    values,
    rhs_matrix,
    output_matrix);
}


//16-bit 8-bit Tile_N = 64 with 2 waprs
template <typename LoadType, typename IndexType, typename VecType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_16b8b(
    int m_vec, int dimN, int dimK, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix)
{
    wmmaSpmm_kernel_16b8b_<LoadType, IndexType, VecType, Tile_K, Tile_N, Warps, VecLength>(
    m_vec, dimN, dimK, scale,
    row_indices, 
    row_offsets,
    column_indices,
    values,
    rhs_matrix,
    output_matrix);
}

template <typename LoadType, typename IndexType, typename VecType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void batched_wmmaSpmm_kernel_16b8b(
    int m_vec, int dimN, int dimK, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values_b,
    int values_stride,
    const int* __restrict__ rhs_matrix_b,
    int rhs_stride,
    half* __restrict__ output_matrix_b,
    int output_stride)
{
    int entry_idx = blockIdx.z;
    const VecType* values = values_b + entry_idx * values_stride;
    const int* rhs_matrix = rhs_matrix_b + entry_idx * rhs_stride;
    half* output_matrix = output_matrix_b + entry_idx * output_stride;

    wmmaSpmm_kernel_16b8b_<LoadType, IndexType, VecType, Tile_K, Tile_N, Warps, VecLength>(
    m_vec, dimN, dimK, scale,
    row_indices, 
    row_offsets,
    column_indices,
    values,
    rhs_matrix,
    output_matrix);
}

//16-bit 8-bit Tile_N = 64 with 2 waprs 8v
template <typename LoadType, typename IndexType, typename VecType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_16b8b8v(
    int m_vec, int dimN, int dimK, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix)
{
    wmmaSpmm_kernel_16b8b8v_<LoadType, IndexType, VecType, Tile_K, Tile_N, Warps, VecLength>(
    m_vec, dimN, dimK, scale,
    row_indices, 
    row_offsets,
    column_indices,
    values,
    rhs_matrix,
    output_matrix);
}

template <typename LoadType, typename IndexType, typename VecType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void batched_wmmaSpmm_kernel_16b8b8v(
    int m_vec, int dimN, int dimK, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values_b,
    int values_stride,
    const int* __restrict__ rhs_matrix_b,
    int rhs_stride,
    half* __restrict__ output_matrix_b,
    int output_stride)
{
    int entry_idx = blockIdx.z;
    const VecType* values = values_b + entry_idx * values_stride;
    const int* rhs_matrix = rhs_matrix_b + entry_idx * rhs_stride;
    half* output_matrix = output_matrix_b + entry_idx * output_stride;

    wmmaSpmm_kernel_16b8b8v_<LoadType, IndexType, VecType, Tile_K, Tile_N, Warps, VecLength>(
    m_vec, dimN, dimK, scale,
    row_indices, 
    row_offsets,
    column_indices,
    values,
    rhs_matrix,
    output_matrix);
}

//8-bit A 4-bit B Tile_N = 64 warps = 2
template <typename LoadType, typename IndexType, typename VecType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_8b4b(
    int m_vec, int dimN, int dimK, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix)
{
    wmmaSpmm_kernel_8b4b_<LoadType, IndexType, VecType, Tile_K, Tile_N, Warps, VecLength>(
    m_vec, dimN, dimK, scale,
    row_indices, 
    row_offsets,
    column_indices,
    values,
    rhs_matrix,
    output_matrix);
}

template <typename LoadType, typename IndexType, typename VecType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void batched_wmmaSpmm_kernel_8b4b(
    int m_vec, int dimN, int dimK, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values_b,
    int values_stride,
    const int* __restrict__ rhs_matrix_b,
    int rhs_stride,
    half* __restrict__ output_matrix_b,
    int output_stride)
{
    int entry_idx = blockIdx.z;
    const VecType* values = values_b + entry_idx * values_stride;
    const int* rhs_matrix = rhs_matrix_b + entry_idx * rhs_stride;
    half* output_matrix = output_matrix_b + entry_idx * output_stride;

    wmmaSpmm_kernel_8b4b_<LoadType, IndexType, VecType, Tile_K, Tile_N, Warps, VecLength>(
    m_vec, dimN, dimK, scale,
    row_indices, 
    row_offsets,
    column_indices,
    values,
    rhs_matrix,
    output_matrix);
}


//8-bit A 4-bit B Tile_N = 64 warps = 2, 8v
template <typename LoadType, typename IndexType, typename VecType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_8b4b8v(
    int m_vec, int dimN, int dimK, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix)
{
    wmmaSpmm_kernel_8b4b8v_<LoadType, IndexType, VecType, Tile_K, Tile_N, Warps, VecLength>(
    m_vec, dimN, dimK, scale,
    row_indices, 
    row_offsets,
    column_indices,
    values,
    rhs_matrix,
    output_matrix);
}


template <typename LoadType, typename IndexType, typename VecType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void batched_wmmaSpmm_kernel_8b4b8v(
    int m_vec, int dimN, int dimK, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values_b,
    int values_stride,
    const int* __restrict__ rhs_matrix_b,
    int rhs_stride,
    half* __restrict__ output_matrix_b,
    int output_stride)
{
    int entry_idx = blockIdx.z;
    const VecType* values = values_b + entry_idx * values_stride;
    const int* rhs_matrix = rhs_matrix_b + entry_idx * rhs_stride;
    half* output_matrix = output_matrix_b + entry_idx * output_stride;

    wmmaSpmm_kernel_8b4b8v_<LoadType, IndexType, VecType, Tile_K, Tile_N, Warps, VecLength>(
    m_vec, dimN, dimK, scale,
    row_indices, 
    row_offsets,
    column_indices,
    values,
    rhs_matrix,
    output_matrix);
}



template <typename IndexType, typename VecType, int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t wmmaSpmm_4b_template(
    int m_vec, int vec_length, int n, int k, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), 1);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    wmmaSpmm_kernel_4b<int, int, VecType, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
        m_vec, n, k, scale, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);

    return cudaGetLastError();
}


template <typename IndexType, typename VecType, int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t wmmaSpmm_8b_template(
    int m_vec, int vec_length, int n, int k, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), 1);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);

    wmmaSpmm_kernel_8b<int, int, VecType, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
        m_vec, n, k, scale, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    return cudaGetLastError();
}

//8-bit 4-bit Tile_N = 64 with 2 warps
template <typename IndexType, typename VecType, int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t wmmaSpmm_8b4b_template(
    int m_vec, int vec_length, int n, int k, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), 1);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    if(vec_length == 8)
        wmmaSpmm_kernel_8b4b8v<int, int, VecType, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, scale, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    else
        wmmaSpmm_kernel_8b4b<int, int, VecType, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, scale, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    return cudaGetLastError();
}

//16-bit 8-bit Tile_N = 64 with 2 waprs
template <typename IndexType, typename VecType, int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t wmmaSpmm_16b8b_template(
    int m_vec, int vec_length, int n, int k, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), 1);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    if(vec_length == 8)
        wmmaSpmm_kernel_16b8b8v<int, int, VecType, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, scale, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    else
        wmmaSpmm_kernel_16b8b<int, int, VecType, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, scale, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    return cudaGetLastError();
}


template <typename IndexType, typename VecType, int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t batched_wmmaSpmm_4b_template(
    int m_vec, int vec_length, int n, int k, int batch_size, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values_b,
    int values_stride,
    const int* __restrict__ rhs_matrix_b,
    int rhs_stride,
    half* __restrict__ output_matrix_b,
    int output_stride)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), batch_size);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    batched_wmmaSpmm_kernel_4b<int, int, VecType, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
        m_vec, n, k, scale, row_indices, row_offsets, column_indices, values_b, values_stride, rhs_matrix_b, rhs_stride, output_matrix_b, output_stride);

    return cudaGetLastError();
}


template <typename IndexType, typename VecType, int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t batched_wmmaSpmm_8b_template(
    int m_vec, int vec_length, int n, int k, int batch_size, float scale, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values_b,
    int values_stride,
    const int* __restrict__ rhs_matrix_b,
    int rhs_stride,
    half* __restrict__ output_matrix_b,
    int output_stride)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), batch_size);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);

    batched_wmmaSpmm_kernel_8b<int, int, VecType, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
        m_vec, n, k, scale, row_indices, row_offsets, column_indices, values_b, values_stride, rhs_matrix_b, rhs_stride, output_matrix_b, output_stride);
    return cudaGetLastError();
}

//8-bit 4-bit Tile_N = 64 with 2 warps
template <typename IndexType, typename VecType, int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t batched_wmmaSpmm_8b4b_template(
    int m_vec, int vec_length, int n, int k, int batch_size, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values_b,
    int values_stride,
    const int* __restrict__ rhs_matrix_b,
    int rhs_stride,
    half* __restrict__ output_matrix_b,
    int output_stride)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), batch_size);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    if(vec_length == 8)
        batched_wmmaSpmm_kernel_8b4b8v<int, int, VecType, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, scale, row_indices, row_offsets, column_indices, values_b, values_stride, rhs_matrix_b, rhs_stride, output_matrix_b, output_stride);
    else
        batched_wmmaSpmm_kernel_8b4b<int, int, VecType, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, scale, row_indices, row_offsets, column_indices, values_b, values_stride, rhs_matrix_b, rhs_stride, output_matrix_b, output_stride);
    return cudaGetLastError();
}

//16-bit 8-bit Tile_N = 64 with 2 waprs
template <typename IndexType, typename VecType, int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t batched_wmmaSpmm_16b8b_template(
    int m_vec, int vec_length, int n, int k, int batch_size, float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values_b,
    int values_stride,
    const int* __restrict__ rhs_matrix_b,
    int rhs_stride,
    half* __restrict__ output_matrix_b,
    int output_stride)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), batch_size);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    if(vec_length == 8)
        batched_wmmaSpmm_kernel_16b8b8v<int, int, VecType, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, scale, row_indices, row_offsets, column_indices, values_b, values_stride, rhs_matrix_b, rhs_stride, output_matrix_b, output_stride);
    else
        batched_wmmaSpmm_kernel_16b8b<int, int, VecType, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, scale, row_indices, row_offsets, column_indices, values_b, values_stride, rhs_matrix_b, rhs_stride, output_matrix_b, output_stride);
    return cudaGetLastError();
}


torch::Tensor batched_deq_spmm_mma_16b8b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits_lhs,
    int bits_rhs,
    float scale)
{
    int rhs_num_items_per_int32 = 32 / bits_rhs;

    int m_vec = row_offsets.size(-1)/2;
    int m = m_vec * vec_length;

    int n_int32 = rhs_matrix.size(-1);

    int n = n_int32 * rhs_num_items_per_int32;

    int k = rhs_matrix.size(-2);

    //int batch_size = rhs_matrix.numel() / (n * k);
    int batch_size = rhs_matrix.size(-3);

    int nnz = column_indices.numel();

    int values_stride = 0;
    if(vec_length == 8)
	values_stride = nnz * 2; //2xlong long for 16b8v
    else
	values_stride = nnz;
    int rhs_stride = k * n_int32;
    int output_stride = m * n;

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(rhs_matrix.device());

    auto output_matrix = torch::empty({batch_size, m, n}, options);


    switch(vec_length){
        case 2:
            batched_wmmaSpmm_16b8b_template<int, int, 1, 16, 64, 32, 2, 2>(m_vec, vec_length, n, k, batch_size, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<int *>(values.data<int>()),
                values_stride,
                reinterpret_cast<int *>(rhs_matrix.data<int>()),
                rhs_stride, 
                reinterpret_cast<half *>(output_matrix.data<torch::Half>()),
	        output_stride);
            break;
        case 4:
            batched_wmmaSpmm_16b8b_template<int, long long, 1, 16, 64, 32, 2, 4>(m_vec, vec_length, n, k, batch_size, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<long long *>(values.data<int>()),
                values_stride,
                reinterpret_cast<int *>(rhs_matrix.data<int>()),
                rhs_stride, 
                reinterpret_cast<half *>(output_matrix.data<torch::Half>()),
	        output_stride);
            break;
        case 8:
            batched_wmmaSpmm_16b8b_template<int, long long, 1, 16, 64, 32, 2, 8>(m_vec, vec_length, n, k, batch_size, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<long long *>(values.data<int>()),
                values_stride,
                reinterpret_cast<int *>(rhs_matrix.data<int>()),
                rhs_stride, 
                reinterpret_cast<half *>(output_matrix.data<torch::Half>()),
	        output_stride);
            break;
        default:
            printf("Unsupported Vector Length!\n");
    }
    return output_matrix;
}



torch::Tensor batched_deq_spmm_mma_4b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits_lhs,
    int bits_rhs,
    float scale)
{
    int rhs_num_items_per_int32 = 32 / bits_rhs;

    int m_vec = row_offsets.size(-1)/2;
    int m = m_vec * vec_length;

    int n_int32 = rhs_matrix.size(-1);

    int n = n_int32 * rhs_num_items_per_int32;

    int k = rhs_matrix.size(-2);

    //int batch_size = rhs_matrix.numel() / (n * k);
    int batch_size = rhs_matrix.size(-3);

    int nnz = column_indices.numel();


    int values_stride = nnz; //stride in vector format
    int rhs_stride = k * n_int32;
    int output_stride = m * n;

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(rhs_matrix.device());

    auto output_matrix = torch::empty({batch_size, m, n}, options);


    switch(vec_length){
        case 2:
            batched_wmmaSpmm_4b_template<int, char, 1, 32, 64, 32, 2, 2>(m_vec, vec_length, n, k, batch_size, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<char *>(values.data<int>()),
                values_stride,
                reinterpret_cast<int *>(rhs_matrix.data<int>()),
                rhs_stride, 
                reinterpret_cast<half *>(output_matrix.data<torch::Half>()),
	        output_stride);
            break;
        case 4:
            batched_wmmaSpmm_4b_template<int, short, 1, 32, 64, 32, 2, 4>(m_vec, vec_length, n, k, batch_size, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<short *>(values.data<int>()),
                values_stride,
                reinterpret_cast<int *>(rhs_matrix.data<int>()),
                rhs_stride, 
                reinterpret_cast<half *>(output_matrix.data<torch::Half>()),
	        output_stride);
            break;
        case 8:
            batched_wmmaSpmm_4b_template<int, int, 1, 32, 64, 32, 2, 8>(m_vec, vec_length, n, k, batch_size, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<int *>(values.data<int>()),
                values_stride,
                reinterpret_cast<int *>(rhs_matrix.data<int>()),
                rhs_stride, 
                reinterpret_cast<half *>(output_matrix.data<torch::Half>()),
	        output_stride);
            break;
        default:
            printf("Unsupported Vector Length!\n");
    }
    return output_matrix;

}



torch::Tensor batched_deq_spmm_mma_8b4b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits_lhs,
    int bits_rhs,
    float scale)
{
    int rhs_num_items_per_int32 = 32 / bits_rhs;

    int m_vec = row_offsets.size(-1)/2;
    int m = m_vec * vec_length;

    int n_int32 = rhs_matrix.size(-1);

    int n = n_int32 * rhs_num_items_per_int32;

    int k = rhs_matrix.size(-2);

    //int batch_size = rhs_matrix.numel() / (n * k);
    int batch_size = rhs_matrix.size(-3);

    int nnz = column_indices.numel();


    int values_stride = nnz; //stride in vector format
    int rhs_stride = k * n_int32;
    int output_stride = m * n;

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(rhs_matrix.device());

    auto output_matrix = torch::empty({batch_size, m, n}, options);


    switch(vec_length){
        case 2:
            batched_wmmaSpmm_8b4b_template<int, short, 1, 32, 64, 32, 2, 2>(m_vec, vec_length, n, k, batch_size, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<short *>(values.data<int>()),
                values_stride,
                reinterpret_cast<int *>(rhs_matrix.data<int>()),
                rhs_stride, 
                reinterpret_cast<half *>(output_matrix.data<torch::Half>()),
	        output_stride);
            break;
        case 4:
            batched_wmmaSpmm_8b4b_template<int, int, 1, 32, 64, 32, 2, 4>(m_vec, vec_length, n, k, batch_size, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<int *>(values.data<int>()),
                values_stride,
                reinterpret_cast<int *>(rhs_matrix.data<int>()),
                rhs_stride, 
                reinterpret_cast<half *>(output_matrix.data<torch::Half>()),
	        output_stride);
            break;
        case 8:
            batched_wmmaSpmm_8b4b_template<int, long long, 1, 32, 64, 32, 2, 8>(m_vec, vec_length, n, k, batch_size, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<long long *>(values.data<int>()),
                values_stride,
                reinterpret_cast<int *>(rhs_matrix.data<int>()),
                rhs_stride, 
                reinterpret_cast<half *>(output_matrix.data<torch::Half>()),
	        output_stride);
            break;
        default:
            printf("Unsupported Vector Length!\n");
    }
    return output_matrix;
}

torch::Tensor batched_deq_spmm_mma_8b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits_lhs,
    int bits_rhs,
    float scale)
{
    //int lhs_num_items_per_int32 = 32 / bits_lhs;
    int rhs_num_items_per_int32 = 32 / bits_rhs;

    int m_vec = row_offsets.size(-1)/2;
    int m = m_vec * vec_length;

    int n_int32 = rhs_matrix.size(-1);

    int n = n_int32 * rhs_num_items_per_int32;

    int k = rhs_matrix.size(-2);

    //int batch_size = rhs_matrix.numel() / (n * k);
    int batch_size = rhs_matrix.size(-3);

    int nnz = column_indices.numel();


    int values_stride = nnz; //stride in vector format
    int rhs_stride = k * n_int32;
    int output_stride = m * n;

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(rhs_matrix.device());

    auto output_matrix = torch::empty({batch_size, m, n}, options);


    switch(vec_length){
        case 2:
            batched_wmmaSpmm_8b_template<int, short, 1, 16, 64, 32, 2, 2>(m_vec, vec_length, n, k, batch_size, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<short *>(values.data<int>()),
                values_stride,
                reinterpret_cast<int *>(rhs_matrix.data<int>()),
                rhs_stride, 
                reinterpret_cast<half *>(output_matrix.data<torch::Half>()),
	        output_stride);
            break;
        case 4:
            batched_wmmaSpmm_8b_template<int, int, 1, 16, 64, 32, 2, 4>(m_vec, vec_length, n, k, batch_size, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<int *>(values.data<int>()),
                values_stride,
                reinterpret_cast<int *>(rhs_matrix.data<int>()),
                rhs_stride, 
                reinterpret_cast<half *>(output_matrix.data<torch::Half>()),
	        output_stride);
            break;
        case 8:
            batched_wmmaSpmm_8b_template<int, long long, 1, 16, 64, 32, 2, 8>(m_vec, vec_length, n, k, batch_size, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<long long *>(values.data<int>()),
                values_stride,
                reinterpret_cast<int *>(rhs_matrix.data<int>()),
                rhs_stride, 
                reinterpret_cast<half *>(output_matrix.data<torch::Half>()),
	        output_stride);
            break;
        default:
            printf("Unsupported Vector Length!\n");
    }
    return output_matrix;

}
