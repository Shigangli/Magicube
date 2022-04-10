#include "../include/wmma_spmm.cuh"
#include "spmm_utils/dense_tile.h"
#include "spmm_utils/sparse_tile.h"
#include "spmm_utils/compute_utils.h"
#include "spmm_utils/output_tile.h"
#include <stdio.h>
#include <mma.h>

using namespace nvcuda;

namespace spmm{

//template <typename LoadType, typename IndexType, typename VecType, 
//          typename OutType, typename StoreType, int Tile_K, 
//          int Tile_N, int BlockWidth, int VecLength=8>
//__global__ void wmmaSpmmKernel8(
//    int m, int k, int n, 
//    const int* __restrict__ row_indices, 
//    const int* __restrict__ row_offsets,
//    const int* __restrict__ column_indices,
//    const half* __restrict__ values,
//    const half* __restrict__ rhs_matrix,
//    OutType* __restrict__ output_matrix)
//{
//    // For the wmma based implementation, we have Tile_M = 1
//    int m_index_vec = blockIdx.x;
//    int dimN_index = blockIdx.y * Tile_N;
//    const int lane_id = threadIdx.x % 4;
//    const int thread_group = threadIdx.x / 4;
//    
//    // Threads that work on different m-dim indices are independent
//    // If we're out of bounds in the m-dimension we can just return
//    if (m_index_vec >= m) return;
//    m_index_vec = __ldg(row_indices + m_index_vec);
//
//    // Load the row offset and calculate the number of nonzeros in the row
//    int row_offset_vec = __ldg(row_offsets + m_index_vec);
//    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset_vec;
//
//    // For VecLength=8, we don't need the memory aligner
//
//    // Shared memory tiles for the lhs values and indices
//    __shared__ float4 values_tile_array[VecLength * Tile_K];
//    //__shared__ float4 values_tile_array[Tile_K];
//    __shared__ int column_indices_tile_array[Tile_K];
//
//    // Pointers to the shared memory tiles
//    float4 * values_tile = values_tile_array;
//    int* column_indices_tile = column_indices_tile_array;
//
//    // Initialize the pointers to the sparse lhs matrix
//    wmmaSparseTile<LoadType, VecType, VecLength, Tile_K, BlockWidth> sparse_tile_loader(
//        k, row_offset_vec, threadIdx.x, values, column_indices,
//        values_tile, column_indices_tile
//    );
//
//    // Register fragment for the dense matrix values
//    constexpr int kDenseFragmentSize = Tile_K / 4 * 8;
//
//    __align__(16) half dense_matrix_fragment[kDenseFragmentSize];
//
//    // Initialize the pointers to the dense rhs matrix
//    wmmaDenseTile<LoadType, Tile_K, Tile_N, BlockWidth> dense_tile_loader(
//        k, dimN_index, lane_id, thread_group, rhs_matrix, column_indices_tile, dense_matrix_fragment
//    );
//
//    // Accumulator registers for the output values.
//    constexpr int kOutputFragmentSize = 16;
//    __align__(16) float output_fragment[kOutputFragmentSize] = {};
//    wmmaComputeUtils8<VecType, Tile_K> computer(values_tile, dense_matrix_fragment, output_fragment, lane_id, thread_group);
//
//    //
//    // Begin kernel main loop
//    //
//
//    constexpr int InnerSteps = Tile_K / 4;
//
//    for (; nonzeros >= Tile_K; nonzeros -= Tile_K){
//        sparse_tile_loader.Load();
//        __syncthreads();
//        #pragma unroll
//        for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
//            dense_tile_loader.LoadRow(n_group_idx);
//        }
//        __threadfence_block();
//        #pragma unroll
//        for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
//            computer.TileMAC(n_group_idx);
//        }
//        __syncthreads();
//    }
//    asm("");
//
//    sparse_tile_loader.ZeroTiles();
//    __syncthreads();
//    sparse_tile_loader.Residue(nonzeros);
//    __syncthreads();
//    
//    int n_group_idx = 0;
//
//    #pragma unroll
//    for (; n_group_idx < InnerSteps; n_group_idx ++){
//        if (nonzeros < 4) break;
//        dense_tile_loader.LoadRow(n_group_idx);
//        computer.TileMAC(n_group_idx);
//        nonzeros -= 4;
//    }
//    asm("");
//
//    dense_tile_loader.ResidueLoad(n_group_idx, nonzeros);
//    computer.TileMACResidue(n_group_idx);
//
//    wmmaOutputTile8<OutType, StoreType> output_tile_storer(lane_id, thread_group, m_index_vec, 
//        dimN_index, k, output_fragment, output_matrix);
//    output_tile_storer.Store();
//    
//}
//
//
//template <typename LoadType, typename IndexType, typename VecType, 
//          typename OutType, int Tile_K, 
//          int Tile_N, int BlockWidth, int VecLength=4>
//__global__ void wmmaSpmmKernel4(
//    int m, int k, int n, 
//    const int* __restrict__ row_indices, 
//    const int* __restrict__ row_offsets,
//    const int* __restrict__ column_indices,
//    const half* __restrict__ values,
//    const half* __restrict__ rhs_matrix,
//    OutType* __restrict__ output_matrix)
//{
//    // For the wmma based implementation, we have Tile_M = 1
//    int m_index_vec = blockIdx.x;
//    int dimN_index = blockIdx.y * Tile_N;
//    const int lane_id = threadIdx.x % 4;
//    const int thread_group = threadIdx.x / 4;
//
//    // Threads that work on different m-dim indices are independent
//    // If we're out of bounds in the m-dimension we can just return
//    if (m_index_vec >= m) return;
//    m_index_vec = __ldg(row_indices + m_index_vec);
//
//    // Load the row offset and calculate the number of nonzeros in the row
//    int row_offset_vec = __ldg(row_offsets + m_index_vec);
//    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset_vec;
//
//    // Shared memory tiles for the lhs values and indices
//    __shared__ float2 values_tile_array[VecLength * Tile_K];
//    __shared__ int column_indices_tile_array[Tile_K];
//
//    // Pointers to the shared memory tiles
//    float2 * values_tile = values_tile_array;
//    int* column_indices_tile = column_indices_tile_array;
//
//    // Initialize the pointers to the sparse lhs matrix
//    wmmaSparseTile<LoadType, VecType, VecLength, Tile_K, BlockWidth> sparse_tile_loader(
//        k, row_offset_vec, threadIdx.x, values, column_indices,
//        values_tile, column_indices_tile
//    );
//
//    // Register fragment for the dense matrix values
//    constexpr int kDenseFragmentSize = Tile_K / 4 * 8;
//
//    __align__(16) half dense_matrix_fragment[kDenseFragmentSize];
//
//    // Initialize the pointers to the dense rhs matrix
//    wmmaDenseTile<LoadType, Tile_K, Tile_N, BlockWidth> dense_tile_loader(
//        k, dimN_index, lane_id, thread_group, rhs_matrix, column_indices_tile, dense_matrix_fragment
//    );
//
//
//    // Accumulator registers for the output values.
//    constexpr int kOutputFragmentSize = 8;
//    __align__(16) float output_fragment[kOutputFragmentSize] = {};
//    wmmaComputeUtils4<VecType, Tile_K> computer(values_tile, dense_matrix_fragment, output_fragment, lane_id, thread_group);
//
//    //
//    // Begin kernel main loop
//    //
//
//    constexpr int InnerSteps = Tile_K / 4;
//
//    for (; nonzeros >= Tile_K; nonzeros -= Tile_K){
//        sparse_tile_loader.Load();
//        __syncthreads();
//        #pragma unroll
//        for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
//            dense_tile_loader.LoadRow(n_group_idx);
//        }
//        __threadfence_block();
//        #pragma unroll
//        for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
//            computer.TileMAC(n_group_idx);
//        }
//        __syncthreads();
//    }
//    
//    sparse_tile_loader.ZeroTiles();
//    __syncthreads();
//    sparse_tile_loader.Residue(nonzeros);
//    __syncthreads();
//
//    int n_group_idx = 0;
//
//    #pragma unroll
//    for (; n_group_idx < InnerSteps; n_group_idx ++){
//        if (nonzeros < 4) break;
//        dense_tile_loader.LoadRow(n_group_idx);
//        computer.TileMAC(n_group_idx);
//        nonzeros -= 4;
//    }
//    asm("");
//
//    dense_tile_loader.ResidueLoad(n_group_idx, nonzeros);
//    computer.TileMACResidue(n_group_idx);
//
//    wmmaOutputTile4<OutType> output_tile_storer(lane_id, thread_group, m_index_vec, dimN_index, k, output_fragment, output_matrix);
//    output_tile_storer.Store();
//}


////4-bit 8-v integer larger Tile_N
//template <typename LoadType, typename IndexType, typename VecType, 
//          typename OutType, int Tile_K, 
//          int Tile_N, int BlockWidth, int VecLength>
//__global__ void wmmaSpmm_kernel_4b8v(
//    int m_vec, int dimN, int dimK, 
//    const int* __restrict__ row_indices, 
//    const int* __restrict__ row_offsets,
//    const int* __restrict__ column_indices,
//    const VecType* __restrict__ values,
//    const int* __restrict__ rhs_matrix,
//    OutType* __restrict__ output_matrix)
//{
//    // For the wmma based implementation, we have Tile_M = 1
//    int m_index_vec = blockIdx.x;
//    int dimN_index = blockIdx.y * Tile_N;
//    const int lane_id = threadIdx.x;
//    //printf("land_id = %d\n", lane_id);
//    //printf("BlockWidth = %d\n", BlockWidth);
//    // Threads that work on different m-dim indices are independent
//    // If we're out of bounds in the m-dimension we can just return
//    if (m_index_vec >= m_vec) return;
//    m_index_vec = __ldg(row_indices + m_index_vec);
//
//    // Load the row offset and calculate the number of nonzeros in the row
//    int row_offset_vec = __ldg(row_offsets + m_index_vec*2);
//    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset_vec;
//
//    // Shared memory tiles for the lhs values and indices
//    __shared__ int values_tile_array[Tile_K*2];
//    __shared__ int column_indices_tile_array[Tile_K*2];
//
//    //padding to avoid bank conflict 
//    __shared__ int dense_tile_array[Tile_N*Tile_K/8 + 8*3];
//    //__shared__ int dense_tile_array[536];
//    //__shared__ int dense_tile_array[Tile_N*Tile_K/8 + 8*7];
//    //__shared__ int dense_tile_array[Tile_N*Tile_K/8];
//
//    // Pointers to the shared memory tiles
//    int* values_tile = values_tile_array;
//    int* column_indices_tile = column_indices_tile_array;
//    int* dense_tile = dense_tile_array;
//
//    // Initialize the pointers to the sparse lhs matrix
//    // ToDo: VecType is useless
//    wmmaSparseTile_4b8v<LoadType, VecType, VecLength, Tile_K, BlockWidth> sparse_tile_loader(
//        dimN/8, row_offset_vec, threadIdx.x % 32, threadIdx.x / 32, values, column_indices,
//        values_tile, column_indices_tile
//    );
//
//    // Register fragment for the dense matrix values
//    //constexpr int kDenseFragmentSize = Tile_K / 4 * 8;
//    //__align__(16) half dense_matrix_fragment[kDenseFragmentSize];
//
//    __align__(16) int rhs_prefetch[8] = {};
//    // Initialize the pointers to the dense rhs matrix
//    wmmaDenseTile_4b<LoadType, Tile_K, Tile_N> dense_tile_loader(
//        dimN/8, dimN_index/8, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
//    );
//
//    // Accumulator registers for the output values.
//    __align__(16) int output_fragment[16] = {};
//    wmmaComputeUtils_4b8v<Tile_K> computer(values_tile, dense_tile, output_fragment, lane_id);
//
//    //
//    // Begin kernel main loop
//    //
//
//    //BlockWidth should be equal to Tile_K
//    int steps = nonzeros / Tile_K;
//    int residue = nonzeros % Tile_K;
//
//    if(steps > 0){
//        sparse_tile_loader.Load(0);
//        __syncthreads();
//        dense_tile_loader.Prefetch(0);
//
//        int i = 1;
//        #pragma unroll
//        for(; i < steps; i++){
//            dense_tile_loader.LoadRowfromRegister(i-1);
//            sparse_tile_loader.Load(i);
//            __syncthreads();
//            dense_tile_loader.Prefetch(i);
//            computer.TileMAC(i-1);
//            __syncthreads();
//        }
//
//        dense_tile_loader.LoadRowfromRegister(i-1);
//        __syncthreads();
//        computer.TileMAC(i-1);
//    }
//   
//    if(residue > 0){
//        sparse_tile_loader.Residue();
//        __syncthreads();
//        dense_tile_loader.ResidueLoad(residue);
//        __syncthreads();
//        computer.TileMACResidue();
//    } 
//
//    wmmaOutputTile_4b8v<OutType> output_tile_storer(lane_id, m_index_vec, dimN_index, dimN, output_fragment, output_matrix);
//    output_tile_storer.Store();
//}


////8-bit 2-v integer larger Tile_N
//template <typename LoadType, typename IndexType, typename VecType, 
//          typename OutType, int Tile_K, 
//          int Tile_N, int WarpWidth, int VecLength>
//__global__ void wmmaSpmm_kernel_8b2v(
//    int m_vec, int dimN, int dimK, 
//    const int* __restrict__ row_indices, 
//    const int* __restrict__ row_offsets,
//    const int* __restrict__ column_indices,
//    const VecType* __restrict__ values,
//    const int* __restrict__ rhs_matrix,
//    OutType* __restrict__ output_matrix)
//{
//    // For the wmma based implementation, we have Tile_M = 1
//    int m_index_vec = blockIdx.x;
//    int dimN_index = blockIdx.y * Tile_N;
//    const int lane_id = threadIdx.x;
//    // Threads that work on different m-dim indices are independent
//    // If we're out of bounds in the m-dimension we can just return
//    if (m_index_vec >= m_vec) return;
//    m_index_vec = __ldg(row_indices + m_index_vec);
//
//    // Load the row offset and calculate the number of nonzeros in the row
//    int row_offset_vec = __ldg(row_offsets + m_index_vec*2);
//    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset_vec;
//
//    // Shared memory tiles for the lhs values and indices
//    // Tile_K short integers plus double buffer
//    __shared__ int values_tile_array[Tile_K];
//    __shared__ int column_indices_tile_array[Tile_K*2];
//
//    //padding to avoid bank conflict 
//    __shared__ int dense_tile_array[Tile_N*Tile_K/4 + 8*7];
//
//    // Pointers to the shared memory tiles
//    int* values_tile = values_tile_array;
//    int* column_indices_tile = column_indices_tile_array;
//    int* dense_tile = dense_tile_array;
//
//    // Initialize the pointers to the sparse lhs matrix
//    //one int32 has four 8-bit integers
//    wmmaSparseTile_8b<LoadType, VecType, Tile_K * VecLength / 4, Tile_K> sparse_tile_loader(
//        row_offset_vec, threadIdx.x % 32, threadIdx.x / 32, values, column_indices,
//        values_tile, column_indices_tile
//    );
//
//    // Register fragment for the dense matrix values
//    //constexpr int kDenseFragmentSize = Tile_K / 4 * 8;
//    //__align__(16) half dense_matrix_fragment[kDenseFragmentSize];
//
//    __align__(16) int rhs_prefetch[4] = {};
//    // Initialize the pointers to the dense rhs matrix
//    wmmaDenseTile_8b<LoadType, Tile_K, Tile_N> dense_tile_loader(
//        dimN/4, dimN_index/4, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
//    );
//
//    // Accumulator registers for the output values.
//    __align__(16) int output_fragment[8] = {};
//    wmmaComputeUtils_8b<Tile_K * VecLength / 4> computer(values_tile, dense_tile, output_fragment, lane_id);
//
//    //
//    // Begin kernel main loop
//    //
//
//    int steps = nonzeros / Tile_K;
//    int residue = nonzeros % Tile_K;
//
//    if(steps > 0){
//        sparse_tile_loader.Load(0);
//        __syncthreads();
//        dense_tile_loader.Prefetch(0);
//
//        int i = 1;
//        #pragma unroll
//        for(; i < steps; i++){
//            dense_tile_loader.LoadRowfromRegister(i-1);
//            sparse_tile_loader.Load(i);
//            __syncthreads();
//            dense_tile_loader.Prefetch(i);
//            computer.TileMAC(i-1);
//            __syncthreads();
//        }
//
//        dense_tile_loader.LoadRowfromRegister(i-1);
//        __syncthreads();
//        computer.TileMAC(i-1);
//    }
//   
//    if(residue > 0){
//        sparse_tile_loader.Residue();
//        __syncthreads();
//        dense_tile_loader.ResidueLoad(residue);
//        __syncthreads();
//        computer.TileMACResidue();
//    } 
//
//    wmmaOutputTile_8b<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment, output_matrix);
//    output_tile_storer.Store();
//}
//
////8-bit 4-v integer larger Tile_N
//template <typename LoadType, typename IndexType, typename VecType, 
//          typename OutType, int Tile_K, 
//          int Tile_N, int WarpWidth, int VecLength>
//__global__ void wmmaSpmm_kernel_8b4v(
//    int m_vec, int dimN, int dimK, 
//    const int* __restrict__ row_indices, 
//    const int* __restrict__ row_offsets,
//    const int* __restrict__ column_indices,
//    const VecType* __restrict__ values,
//    const int* __restrict__ rhs_matrix,
//    OutType* __restrict__ output_matrix)
//{
//    // For the wmma based implementation, we have Tile_M = 1
//    int m_index_vec = blockIdx.x;
//    int dimN_index = blockIdx.y * Tile_N;
//    const int lane_id = threadIdx.x;
//    // Threads that work on different m-dim indices are independent
//    // If we're out of bounds in the m-dimension we can just return
//    if (m_index_vec >= m_vec) return;
//    m_index_vec = __ldg(row_indices + m_index_vec);
//
//    // Load the row offset and calculate the number of nonzeros in the row
//    int row_offset_vec = __ldg(row_offsets + m_index_vec*2);
//    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset_vec;
//
//    // Shared memory tiles for the lhs values and indices
//    __shared__ int values_tile_array[Tile_K*2];
//    __shared__ int column_indices_tile_array[Tile_K*2];
//
//    //padding to avoid bank conflict 
//    __shared__ int dense_tile_array[Tile_N*Tile_K/4 + 8*7];
//
//    // Pointers to the shared memory tiles
//    int* values_tile = values_tile_array;
//    int* column_indices_tile = column_indices_tile_array;
//    int* dense_tile = dense_tile_array;
//
//    // Initialize the pointers to the sparse lhs matrix
//    //one int32 has four 8-bit integers
//    wmmaSparseTile_8b<LoadType, VecType, Tile_K * VecLength / 4, Tile_K> sparse_tile_loader(
//        row_offset_vec, threadIdx.x % 32, threadIdx.x / 32, values, column_indices,
//        values_tile, column_indices_tile
//    );
//
//    // Register fragment for the dense matrix values
//    //constexpr int kDenseFragmentSize = Tile_K / 4 * 8;
//    //__align__(16) half dense_matrix_fragment[kDenseFragmentSize];
//
//    __align__(16) int rhs_prefetch[4] = {};
//    // Initialize the pointers to the dense rhs matrix
//    wmmaDenseTile_8b<LoadType, Tile_K, Tile_N> dense_tile_loader(
//        dimN/4, dimN_index/4, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
//    );
//
//    // Accumulator registers for the output values.
//    __align__(16) int output_fragment[8] = {};
//    wmmaComputeUtils_8b<Tile_K * VecLength / 4> computer(values_tile, dense_tile, output_fragment, lane_id);
//
//    //
//    // Begin kernel main loop
//    //
//
//    int steps = nonzeros / Tile_K;
//    int residue = nonzeros % Tile_K;
//
//    if(steps > 0){
//        sparse_tile_loader.Load(0);
//        __syncthreads();
//        dense_tile_loader.Prefetch(0);
//
//        int i = 1;
//        #pragma unroll
//        for(; i < steps; i++){
//            dense_tile_loader.LoadRowfromRegister(i-1);
//            sparse_tile_loader.Load(i);
//            __syncthreads();
//            dense_tile_loader.Prefetch(i);
//            computer.TileMAC(i-1);
//            __syncthreads();
//        }
//
//        dense_tile_loader.LoadRowfromRegister(i-1);
//        __syncthreads();
//        computer.TileMAC(i-1);
//    }
//   
//    if(residue > 0){
//        sparse_tile_loader.Residue();
//        __syncthreads();
//        dense_tile_loader.ResidueLoad(residue);
//        __syncthreads();
//        computer.TileMACResidue();
//    } 
//
//    wmmaOutputTile_8b<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment, output_matrix);
//    output_tile_storer.Store();
//}
//
////8-bit 8-v integer larger Tile_N
//template <typename LoadType, typename IndexType, typename VecType, 
//          typename OutType, int Tile_K, 
//          int Tile_N, int WarpWidth, int VecLength>
//__global__ void wmmaSpmm_kernel_8b8v(
//    int m_vec, int dimN, int dimK, 
//    const int* __restrict__ row_indices, 
//    const int* __restrict__ row_offsets,
//    const int* __restrict__ column_indices,
//    const VecType* __restrict__ values,
//    const int* __restrict__ rhs_matrix,
//    OutType* __restrict__ output_matrix)
//{
//    // For the wmma based implementation, we have Tile_M = 1
//    int m_index_vec = blockIdx.x;
//    int dimN_index = blockIdx.y * Tile_N;
//    const int lane_id = threadIdx.x;
//    // Threads that work on different m-dim indices are independent
//    // If we're out of bounds in the m-dimension we can just return
//    if (m_index_vec >= m_vec) return;
//    m_index_vec = __ldg(row_indices + m_index_vec);
//
//    // Load the row offset and calculate the number of nonzeros in the row
//    int row_offset_vec = __ldg(row_offsets + m_index_vec*2);
//    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset_vec;
//
//    // Shared memory tiles for the lhs values and indices
//    // Tile_K long long integers plus double buffer
//    __shared__ int values_tile_array[Tile_K*4];
//    __shared__ int column_indices_tile_array[Tile_K*2];
//
//    //padding to avoid bank conflict 
//    __shared__ int dense_tile_array[Tile_N*Tile_K/4 + 8*7];
//
//    // Pointers to the shared memory tiles
//    int* values_tile = values_tile_array;
//    int* column_indices_tile = column_indices_tile_array;
//    int* dense_tile = dense_tile_array;
//
//    // Initialize the pointers to the sparse lhs matrix
//    //one int32 has four 8-bit integers
//    wmmaSparseTile_8b<LoadType, VecType, Tile_K * VecLength / 4, Tile_K> sparse_tile_loader(
//        row_offset_vec, threadIdx.x % 32, threadIdx.x / 32, values, column_indices,
//        values_tile, column_indices_tile
//    );
//
//    __align__(16) int rhs_prefetch[4] = {};
//    // Initialize the pointers to the dense rhs matrix
//    wmmaDenseTile_8b<LoadType, Tile_K, Tile_N> dense_tile_loader(
//        dimN/4, dimN_index/4, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
//    );
//
//    // Accumulator registers for the output values.
//    __align__(16) int output_fragment[8] = {};
//    wmmaComputeUtils_8b<Tile_K * VecLength / 4> computer(values_tile, dense_tile, output_fragment, lane_id);
//
//    //
//    // Begin kernel main loop
//    //
//
//    int steps = nonzeros / Tile_K;
//    int residue = nonzeros % Tile_K;
//
//    if(steps > 0){
//        sparse_tile_loader.Load(0);
//        __syncthreads();
//        dense_tile_loader.Prefetch(0);
//
//        int i = 1;
//        #pragma unroll
//        for(; i < steps; i++){
//            dense_tile_loader.LoadRowfromRegister(i-1);
//            sparse_tile_loader.Load(i);
//            __syncthreads();
//            dense_tile_loader.Prefetch(i);
//            computer.TileMAC(i-1);
//            __syncthreads();
//        }
//
//        dense_tile_loader.LoadRowfromRegister(i-1);
//        __syncthreads();
//        computer.TileMAC(i-1);
//    }
//   
//    if(residue > 0){
//        sparse_tile_loader.Residue();
//        __syncthreads();
//        dense_tile_loader.ResidueLoad(residue);
//        __syncthreads();
//        computer.TileMACResidue();
//    } 
//
//    wmmaOutputTile_8b<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment, output_matrix);
//    output_tile_storer.Store();
//}

//4-bit Tile_N = 128 with 2 warps
template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_4b(
    int m_vec, int dimN, int dimK, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
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

    __align__(16) int rhs_prefetch[8] = {};
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

    wmmaOutputTile_4b<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment, output_matrix);
    output_tile_storer.Store();
}

//8-bit Tile_N = 128 with 4 warps
template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_8b(
    int m_vec, int dimN, int dimK, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
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
    __shared__ int dense_tile_array[Tile_N*Tile_K/4 + 8*7];

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

    wmmaOutputTile_8b<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment, output_matrix);
    output_tile_storer.Store();
}

//16-bit 8-bit Tile_N = 128 with 4 warps
template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_16b8b(
    int m_vec, int dimN, int dimK, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
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
    __shared__ int dense_tile_array[Tile_N*Tile_K/4 + 8*7];

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

    wmmaOutputTile_16b8b<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment, output_matrix);
    output_tile_storer.Store();
}


//16-bit 8-bit Tile_N = 128 with 4 warps 8v
template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_16b8b8v(
    int m_vec, int dimN, int dimK, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
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
    __shared__ int dense_tile_array[Tile_N*Tile_K/4 + 8*7];

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

    wmmaOutputTile_16b8b8v<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment_0, output_fragment_1, output_matrix);
    output_tile_storer.Store();
}

//8-bit A 4-bit B Tile_N = 128 warps = 2
template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_8b4b(
    int m_vec, int dimN, int dimK, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
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

    __align__(16) int rhs_prefetch[8] = {};
    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile_4b<LoadType, Tile_K, Tile_N> dense_tile_loader(
        dimN/8, dimN_index/8, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    __align__(16) int output_fragment[Tile_N / Warps / 4] = {};
    wmmaComputeUtils_8b4b<Tile_K * VecLength / 4> computer(values_tile, dense_tile, output_fragment, lane_id);

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

    wmmaOutputTile_8b4b<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment, output_matrix);
    output_tile_storer.Store();
}

//12-bit A 4-bit B Tile_N = 128 warps = 2
template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_12b4b2v(
    int m_vec, int dimN, int dimK, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
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

    // Shared memory tiles for the lhs values and indices, double buffers
    __shared__ int values_tile_array[Tile_K*VecLength];
    __shared__ int column_indices_tile_array[Tile_K*2];

    // each int value has four 4-bit values, padding to avoid bank conflict 
    __shared__ int dense_tile_array[Tile_N*Tile_K/8 + 8*3];

    // Pointers to the shared memory tiles
    int* values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;
    int* dense_tile = dense_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    wmmaSparseTile_12b4b2v<LoadType, VecType, Tile_K * VecLength / 2, Tile_K> sparse_tile_loader(
        row_offset_vec, threadIdx.x % 32, threadIdx.x / 32, values, column_indices,
        values_tile, column_indices_tile
    );

    __align__(16) int rhs_prefetch[8] = {};
    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile_4b<LoadType, Tile_K, Tile_N> dense_tile_loader(
        dimN/8, dimN_index/8, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    __align__(16) int output_fragment[Tile_N / Warps / 4] = {};
    wmmaComputeUtils_12b4b2v<Tile_K * VecLength / 2> computer(values_tile, dense_tile, output_fragment, lane_id);

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

    wmmaOutputTile_12b4b2v<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment, output_matrix);
    output_tile_storer.Store();
}

//12-bit A 4-bit B Tile_N = 128 warps = 2
template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_12b4b4v(
    int m_vec, int dimN, int dimK, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
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

    // Shared memory tiles for the lhs values and indices, double buffers
    __shared__ int values_tile_array[Tile_K*VecLength];
    __shared__ int column_indices_tile_array[Tile_K*2];

    // each int value has four 4-bit values, padding to avoid bank conflict 
    __shared__ int dense_tile_array[Tile_N*Tile_K/8 + 8*3];

    // Pointers to the shared memory tiles
    int* values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;
    int* dense_tile = dense_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    wmmaSparseTile_12b4b4v<LoadType, VecType, Tile_K * VecLength / 2, Tile_K> sparse_tile_loader(
        row_offset_vec, lane_id, values, column_indices,
        values_tile, column_indices_tile
    );

    __align__(16) int rhs_prefetch[8] = {};
    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile_4b<LoadType, Tile_K, Tile_N> dense_tile_loader(
        dimN/8, dimN_index/8, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    __align__(16) int output_fragment_0[Tile_N / Warps / 4] = {};
    __align__(16) int output_fragment_1[Tile_N / Warps / 4] = {};
    wmmaComputeUtils_12b4b4v<Tile_K * VecLength / 2> computer(values_tile, dense_tile, output_fragment_0, output_fragment_1, lane_id);

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

    wmmaOutputTile_12b4b4v<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment_0, output_fragment_1, output_matrix);
    output_tile_storer.Store();
}

//12-bit A 4-bit B Tile_N = 128 warps = 2
template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_12b4b8v(
    int m_vec, int dimN, int dimK, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
{
    // For the wmma based implementation, we have Tile_M = 1
    int m_index_vec = blockIdx.x;
    int dimN_index = blockIdx.y * Tile_N;
    const int lane_id = threadIdx.x;
    const int lane_size = blockDim.x;
    // Threads that work on different m-dim indices are independent
    // If we're out of bounds in the m-dimension we can just return
    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec*2);
    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset_vec;

    // Shared memory tiles for the lhs values and indices, double buffers
    __shared__ int values_tile_array[Tile_K*6]; //8v 12bit only requires Tile_k * 6
    __shared__ int column_indices_tile_array[Tile_K*2];

    // each int value has four 4-bit values, padding to avoid bank conflict 
    __shared__ int dense_tile_array[Tile_N*Tile_K/8 + 8*3];

    // Pointers to the shared memory tiles
    int* values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;
    int* dense_tile = dense_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    wmmaSparseTile_12b4b8v<LoadType, VecType, Tile_K * 3, Tile_K * VecLength / 2, Tile_K> sparse_tile_loader(
        row_offset_vec, lane_id, lane_size, values, column_indices,
        values_tile, column_indices_tile
    );

    __align__(16) int rhs_prefetch[8] = {};
    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile_4b<LoadType, Tile_K, Tile_N> dense_tile_loader(
        dimN/8, dimN_index/8, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    __align__(16) int output_fragment_0[Tile_N / Warps / 4] = {};
    __align__(16) int output_fragment_1[Tile_N / Warps / 4] = {};
    __align__(16) int output_fragment_2[Tile_N / Warps / 4] = {};
    wmmaComputeUtils_12b4b8v<Tile_K * 3> computer(values_tile, dense_tile, output_fragment_0, output_fragment_1, output_fragment_2, lane_id);

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

    wmmaOutputTile_12b4b8v<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment_0, output_fragment_1, output_fragment_2, output_matrix);
    output_tile_storer.Store();
}

//16-bit A 4-bit B Tile_N = 128 warps = 2
template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_16b4b2v(
    int m_vec, int dimN, int dimK, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
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

    // Shared memory tiles for the lhs values and indices, double buffers
    __shared__ int values_tile_array[Tile_K*VecLength];
    __shared__ int column_indices_tile_array[Tile_K*2];

    // each int value has four 4-bit values, padding to avoid bank conflict 
    __shared__ int dense_tile_array[Tile_N*Tile_K/8 + 8*3];

    // Pointers to the shared memory tiles
    int* values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;
    int* dense_tile = dense_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    wmmaSparseTile_12b4b2v<LoadType, VecType, Tile_K * VecLength / 2, Tile_K> sparse_tile_loader(
        row_offset_vec, threadIdx.x % 32, threadIdx.x / 32, values, column_indices,
        values_tile, column_indices_tile
    );

    __align__(16) int rhs_prefetch[8] = {};
    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile_4b<LoadType, Tile_K, Tile_N> dense_tile_loader(
        dimN/8, dimN_index/8, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    __align__(16) int output_fragment[Tile_N / Warps / 4] = {};
    wmmaComputeUtils_12b4b2v<Tile_K * VecLength / 2> computer(values_tile, dense_tile, output_fragment, lane_id);

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

    wmmaOutputTile_12b4b2v<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment, output_matrix);
    output_tile_storer.Store();
}

//16-bit A 4-bit B Tile_N = 128 warps = 2
template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_16b4b4v(
    int m_vec, int dimN, int dimK, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
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

    // Shared memory tiles for the lhs values and indices, double buffers
    __shared__ int values_tile_array[Tile_K*VecLength];
    __shared__ int column_indices_tile_array[Tile_K*2];

    // each int value has four 4-bit values, padding to avoid bank conflict 
    __shared__ int dense_tile_array[Tile_N*Tile_K/8 + 8*3];

    // Pointers to the shared memory tiles
    int* values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;
    int* dense_tile = dense_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    wmmaSparseTile_16b4b4v<LoadType, VecType, Tile_K * VecLength / 2, Tile_K> sparse_tile_loader(
        row_offset_vec, lane_id, values, column_indices,
        values_tile, column_indices_tile
    );

    __align__(16) int rhs_prefetch[8] = {};
    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile_4b<LoadType, Tile_K, Tile_N> dense_tile_loader(
        dimN/8, dimN_index/8, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    __align__(16) int output_fragment_0[Tile_N / Warps / 4] = {};
    __align__(16) int output_fragment_1[Tile_N / Warps / 4] = {};
    wmmaComputeUtils_16b4b4v<Tile_K * VecLength / 2> computer(values_tile, dense_tile, output_fragment_0, output_fragment_1, lane_id);

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

    wmmaOutputTile_16b4b4v<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment_0, output_fragment_1, output_matrix);
    output_tile_storer.Store();
}

//16-bit A 4-bit B Tile_N = 128 warps = 2
template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_16b4b8v(
    int m_vec, int dimN, int dimK, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
{
    // For the wmma based implementation, we have Tile_M = 1
    int m_index_vec = blockIdx.x;
    int dimN_index = blockIdx.y * Tile_N;
    const int lane_id = threadIdx.x;
    const int lane_size = blockDim.x;
    // Threads that work on different m-dim indices are independent
    // If we're out of bounds in the m-dimension we can just return
    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec*2);
    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset_vec;

    // Shared memory tiles for the lhs values and indices, double buffers
    __shared__ int values_tile_array[Tile_K*VecLength]; //8v 12bit only requires Tile_k * 6
    __shared__ int column_indices_tile_array[Tile_K*2];

    // each int value has four 4-bit values, padding to avoid bank conflict 
    __shared__ int dense_tile_array[Tile_N*Tile_K/8 + 8*3];

    // Pointers to the shared memory tiles
    int* values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;
    int* dense_tile = dense_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    wmmaSparseTile_16b4b8v<LoadType, VecType, Tile_K * VecLength / 2, Tile_K> sparse_tile_loader(
        row_offset_vec, lane_id, lane_size, values, column_indices,
        values_tile, column_indices_tile
    );

    __align__(16) int rhs_prefetch[8] = {};
    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile_4b<LoadType, Tile_K, Tile_N> dense_tile_loader(
        dimN/8, dimN_index/8, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    __align__(16) int output_fragment_0[Tile_N / Warps / 4] = {};
    __align__(16) int output_fragment_1[Tile_N / Warps / 4] = {};
    __align__(16) int output_fragment_2[Tile_N / Warps / 4] = {};
    __align__(16) int output_fragment_3[Tile_N / Warps / 4] = {};
    wmmaComputeUtils_16b4b8v<Tile_K * VecLength / 2> computer(values_tile, dense_tile, output_fragment_0, output_fragment_1, output_fragment_2, output_fragment_3, lane_id);

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

    wmmaOutputTile_16b4b8v<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment_0, output_fragment_1, output_fragment_2, output_fragment_3, output_matrix);
    output_tile_storer.Store();
}

//8-bit A 4-bit B Tile_N = 128 warps = 2, 8v
template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_8b4b8v(
    int m_vec, int dimN, int dimK, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
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

    __align__(16) int rhs_prefetch[8] = {};
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

    wmmaOutputTile_8b4b8v<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment_0, output_fragment_1, output_matrix);
    output_tile_storer.Store();
}


//16-bit 16-bit Tile_N = 64 with 4 warps
template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_16b(
    int m_vec, int dimN, int dimK, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
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

    // One int32 has two 16-bit integers
    // Padding to avoid bank conflict 
    __shared__ int dense_tile_array[Tile_N*Tile_K/2 + 8*7];

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
    // One int32 has two 16-bit integers
    wmmaDenseTile_16b<LoadType, Tile_K, Tile_N> dense_tile_loader(
        dimN/2, dimN_index/2, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    // Tile_N / warps / four threads in x-dim of output matrix
    // 16-bit decomposes into two 8-bits, x2
    __align__(16) int output_fragment[Tile_N / Warps / 2] = {};
    wmmaComputeUtils_16b<Tile_K * VecLength / 2> computer(values_tile, dense_tile, output_fragment, lane_id);

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

    wmmaOutputTile_16b<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment, output_matrix);
    output_tile_storer.Store();
}

//16-bit 16-bit Tile_N = 64 with 4 warps
template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_16b8v(
    int m_vec, int dimN, int dimK, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
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

    // One int32 has two 16-bit integers
    // Padding to avoid bank conflict 
    __shared__ int dense_tile_array[Tile_N*Tile_K/2 + 8*7];

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
    // One int32 has two 16-bit integers
    wmmaDenseTile_16b<LoadType, Tile_K, Tile_N> dense_tile_loader(
        dimN/2, dimN_index/2, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    // Tile_N / warps / four threads in x-dim of output matrix
    // 16-bit decomposes into two 8-bits, x2
    __align__(16) int output_fragment_0[Tile_N / Warps / 2] = {};
    __align__(16) int output_fragment_1[Tile_N / Warps / 2] = {};
    wmmaComputeUtils_16b8v<Tile_K * VecLength / 2> computer(values_tile, dense_tile, output_fragment_0, output_fragment_1, lane_id);

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

    wmmaOutputTile_16b8v<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment_0, output_fragment_1, output_matrix);
    output_tile_storer.Store();
}

////4-bit integer
//template <typename LoadType, typename IndexType, typename VecType, 
//          typename OutType, int Tile_K, 
//          int Tile_N, int BlockWidth, int VecLength=4>
//__global__ void wmmaSpmmKernel4_4bit(
//    int m_vec, int k, int n, 
//    const int* __restrict__ row_indices, 
//    const int* __restrict__ row_offsets,
//    const int* __restrict__ column_indices,
//    const short* __restrict__ values,
//    const int* __restrict__ rhs_matrix,
//    OutType* __restrict__ output_matrix)
//{
//    // For the wmma based implementation, we have Tile_M = 1
//    int m_index_vec = blockIdx.x;
//    int dimN_index = blockIdx.y * Tile_N;
//    const int lane_id = threadIdx.x;
//
//    // Threads that work on different m-dim indices are independent
//    // If we're out of bounds in the m-dimension we can just return
//    if (m_index_vec >= m_vec) return;
//    m_index_vec = __ldg(row_indices + m_index_vec);
//
//    // Load the row offset and calculate the number of nonzeros in the row
//    int row_offset_vec = __ldg(row_offsets + m_index_vec*2);
//    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset_vec;
//
//    // Shared memory tiles for the lhs values and indices
//    __shared__ int values_tile_array[Tile_K/2];
//    __shared__ int column_indices_tile_array[Tile_K];
//
//    // each int value has four 4-bit values, padding to avoid bank conflict, assuming Tile_K=64 
//    __shared__ int dense_tile_array[Tile_N*32/8 + 8*3];
//    //__shared__ int dense_tile_array[Tile_N*Tile_K/8 + 8*7];
//    //__shared__ int dense_tile_array[Tile_N*Tile_K/8];
//
//    // Pointers to the shared memory tiles
//    int* values_tile = values_tile_array;
//    int* column_indices_tile = column_indices_tile_array;
//    int* dense_tile = dense_tile_array;
//
//    // Initialize the pointers to the sparse lhs matrix
//    // ToDo: VecType is useless?
//    wmmaSparseTile_4b<LoadType, VecType, VecLength, Tile_K, BlockWidth> sparse_tile_loader(
//        k, row_offset_vec, threadIdx.x, values, column_indices,
//        values_tile, column_indices_tile
//    );
//
//    // Register fragment for the dense matrix values
//    //constexpr int kDenseFragmentSize = Tile_K / 4 * 8;
//    //__align__(16) half dense_matrix_fragment[kDenseFragmentSize];
//
//    // Initialize the pointers to the dense rhs matrix
//    wmmaDenseTile_4b<LoadType, Tile_K, Tile_N, BlockWidth> dense_tile_loader(
//        k, dimN_index/8, lane_id, rhs_matrix, column_indices_tile, dense_tile
//    );
//
//    // Accumulator registers for the output values.
//    constexpr int kOutputFragmentSize = 16;
//    __align__(16) int output_fragment[kOutputFragmentSize] = {};
//    wmmaComputeUtils4_4bit<Tile_K> computer(values_tile, dense_tile, output_fragment, lane_id);
//
//    //
//    // Begin kernel main loop
//    //
//
//    constexpr int InnerSteps = Tile_K / 32;
//
//    for (; nonzeros >= Tile_K; nonzeros -= Tile_K){
//        sparse_tile_loader.Load();
//        //__syncthreads();
//        #pragma unroll
//        for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
//            dense_tile_loader.LoadRow(n_group_idx);
//            computer.TileMAC(n_group_idx);
//        }
//        //__threadfence_block();
//        //#pragma unroll
//        //for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
//        //    computer.TileMAC(n_group_idx);
//        //}
//        //__syncthreads();
//    }
//   
//    if(nonzeros > 0){
//        //sparse_tile_loader.ZeroTiles();
//        //__syncthreads();
//        sparse_tile_loader.Residue(nonzeros);
//        //__syncthreads();
//
//        int n_group_idx_red = 0;
//
//        #pragma unroll
//        for (; n_group_idx_red < InnerSteps; n_group_idx_red++){
//            if (nonzeros < 32) break;
//            dense_tile_loader.LoadRow(n_group_idx_red);
//            computer.TileMAC(n_group_idx_red);
//            nonzeros -= 32;
//        }
//        asm("");
//
//        if(nonzeros > 0){
//            dense_tile_loader.ResidueLoad(n_group_idx_red, nonzeros);
//            //computer.TileMACResidue(n_group_idx_red);
//            computer.TileMAC(n_group_idx_red);
//        }
//    } 
//
//    wmmaOutputTile4_4bit<OutType> output_tile_storer(lane_id, m_index_vec, dimN_index, k, output_fragment, output_matrix);
//    output_tile_storer.Store();
//}

//template <typename IndexType, int Tile_M, int Tile_K, int Tile_N, int BlockWidth>
//cudaError_t wmmaSpmm_4b_template(
//    int m_vec, int vec_length, int k, int n, 
//    const int* __restrict__ row_indices, 
//    const int* __restrict__ row_offsets,
//    const int* __restrict__ column_indices,
//    const short* __restrict__ values,
//    const int* __restrict__ rhs_matrix,
//    int* __restrict__ output_matrix)
//{
//    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(k) / Tile_N), 1);
//    dim3 block_dim(BlockWidth, Tile_M, 1);
//    switch(vec_length){
//        //case 2:
//        //    //printf("V=2\n");
//        //    wmmaSpmmKernel2<int, int, short, int4, Tile_K, Tile_N, BlockWidth, 2><<<grid_dim, block_dim>>>(
//        //        m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
//        //    break;
//        case 4:
//            //printf("V=4\n");
//            wmmaSpmmKernel4_4bit<int, int, short, int, Tile_K, Tile_N, BlockWidth, 4><<<grid_dim, block_dim>>>(
//                m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
//            break;
//        //case 8:
//        //    //printf("V=8\n");
//        //    wmmaSpmmKernel8<int, int, int2, int4, int2, Tile_K, Tile_N, BlockWidth, 8><<<grid_dim, block_dim>>>(
//        //        m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
//        //    break;
//        default:
//            printf("Unsupported Vector Length!\n");
//    }
//
//    return cudaGetLastError();
//}




//cudaError_t wmmaSpmm_8b4v(int m_vec, int vec_length, int n, int k, 
//    const int* __restrict__ row_indices, 
//    const int* __restrict__ row_offsets,
//    const int* __restrict__ column_indices,
//    const int* __restrict__ values,
//    const int* __restrict__ rhs_matrix,
//    int* __restrict__ output_matrix)
//{
//    return wmmaSpmm_8b_template<int, int, 1, 16, 128, 32>(m_vec, vec_length, n, k, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
//}

//cudaError_t wmmaSpmm_8b8v(int m_vec, int vec_length, int n, int k, 
//    const int* __restrict__ row_indices, 
//    const int* __restrict__ row_offsets,
//    const int* __restrict__ column_indices,
//    const int* __restrict__ values,
//    const int* __restrict__ rhs_matrix,
//    int* __restrict__ output_matrix)
//{
//    return wmmaSpmm_8b_template<int, long long, 1, 16, 128, 32>(m_vec, vec_length, n, k, row_indices, row_offsets, column_indices, reinterpret_cast<const long long *>(values), rhs_matrix, output_matrix);
//}

template <typename IndexType, typename VecType, int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t wmmaSpmm_4b_template(
    int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), 1);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    wmmaSpmm_kernel_4b<int, int, VecType, int, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
        m_vec, n, k, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);

    return cudaGetLastError();
}

//4-bit Tile_N = 128 with 2 warps
cudaError_t wmmaSpmm_4b(int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    switch(vec_length){
        case 2:
            return wmmaSpmm_4b_template<int, char, 1, 32, 128, 32, 2, 2>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const char *>(values), rhs_matrix, output_matrix);
            break;
        case 4:
            return wmmaSpmm_4b_template<int, short, 1, 32, 128, 32, 2, 4>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const short *>(values), rhs_matrix, output_matrix);
            break;
        case 8:
            return wmmaSpmm_4b_template<int, int, 1, 32, 128, 32, 2, 8>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, values, rhs_matrix, output_matrix);
            break;
        default:
            printf("Unsupported Vector Length!\n");
            return cudaGetLastError();
    }
}

template <typename IndexType, typename VecType, int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t wmmaSpmm_8b_template(
    int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), 1);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);

    wmmaSpmm_kernel_8b<int, int, VecType, int, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
        m_vec, n, k, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    return cudaGetLastError();
}

//8-bit Tile_N = 128 with 4 warps
cudaError_t wmmaSpmm_8b(int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    switch(vec_length){
        case 2:
            return wmmaSpmm_8b_template<int, short, 1, 16, 128, 32, 4, 2>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const short *>(values), rhs_matrix, output_matrix);
            break;
        case 4:
            return wmmaSpmm_8b_template<int, int, 1, 16, 128, 32, 4, 4>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, values, rhs_matrix, output_matrix);
            break;
        case 8:
            return wmmaSpmm_8b_template<int, long long, 1, 16, 128, 32, 4, 8>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const long long *>(values), rhs_matrix, output_matrix);
            break;
        default:
            printf("Unsupported Vector Length!\n");
            return cudaGetLastError();
    }
}

//12-bit 4-bit Tile_N = 128 with 2 warps
template <typename IndexType, typename VecType, int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t wmmaSpmm_12b4b_template(
    int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), 1);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    if(vec_length == 8)
        wmmaSpmm_kernel_12b4b8v<int, int, VecType, int, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    if(vec_length == 4)
        wmmaSpmm_kernel_12b4b4v<int, int, VecType, int, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    if(vec_length == 2)
        wmmaSpmm_kernel_12b4b2v<int, int, VecType, int, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    return cudaGetLastError();
}

cudaError_t wmmaSpmm_12b4b(int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    switch(vec_length){
        case 2:
            return wmmaSpmm_12b4b_template<int, int, 1, 32, 128, 32, 2, 2>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, values, rhs_matrix, output_matrix);
            break;
        case 4:
            return wmmaSpmm_12b4b_template<int, long long, 1, 32, 128, 32, 2, 4>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const long long *>(values), rhs_matrix, output_matrix);
            break;
        case 8:
            return wmmaSpmm_12b4b_template<int, long long, 1, 32, 128, 32, 2, 8>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const long long *>(values), rhs_matrix, output_matrix);
            break;
        default:
            printf("Unsupported Vector Length!\n");
            return cudaGetLastError();
    }
}

//16-bit 4-bit Tile_N = 128 with 2 warps
template <typename IndexType, typename VecType, int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t wmmaSpmm_16b4b_template(
    int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), 1);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    if(vec_length == 8)
        wmmaSpmm_kernel_16b4b8v<int, int, VecType, int, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    if(vec_length == 4)
        wmmaSpmm_kernel_16b4b4v<int, int, VecType, int, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    if(vec_length == 2)
        wmmaSpmm_kernel_16b4b2v<int, int, VecType, int, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    return cudaGetLastError();
}

cudaError_t wmmaSpmm_16b4b(int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    switch(vec_length){
        case 2:
            return wmmaSpmm_16b4b_template<int, int, 1, 32, 128, 32, 2, 2>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, values, rhs_matrix, output_matrix);
            break;
        case 4:
            return wmmaSpmm_16b4b_template<int, long long, 1, 32, 128, 32, 2, 4>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const long long *>(values), rhs_matrix, output_matrix);
            break;
        case 8:
            return wmmaSpmm_16b4b_template<int, long long, 1, 32, 128, 32, 2, 8>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const long long *>(values), rhs_matrix, output_matrix);
            break;
        default:
            printf("Unsupported Vector Length!\n");
            return cudaGetLastError();
    }
}

//8-bit 4-bit Tile_N = 128 with 2 warps
template <typename IndexType, typename VecType, int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t wmmaSpmm_8b4b_template(
    int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), 1);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    if(vec_length == 8)
        wmmaSpmm_kernel_8b4b8v<int, int, VecType, int, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    else
        wmmaSpmm_kernel_8b4b<int, int, VecType, int, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    return cudaGetLastError();
}

cudaError_t wmmaSpmm_8b4b(int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    switch(vec_length){
        case 2:
            return wmmaSpmm_8b4b_template<int, short, 1, 32, 128, 32, 2, 2>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const short *>(values), rhs_matrix, output_matrix);
            break;
        case 4:
            return wmmaSpmm_8b4b_template<int, int, 1, 32, 128, 32, 2, 4>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, values, rhs_matrix, output_matrix);
            break;
        case 8:
            return wmmaSpmm_8b4b_template<int, long long, 1, 32, 128, 32, 2, 8>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const long long *>(values), rhs_matrix, output_matrix);
            break;
        default:
            printf("Unsupported Vector Length!\n");
            return cudaGetLastError();
    }
}

//16-bit 8-bit Tile_N = 128 with 4 warps
template <typename IndexType, typename VecType, int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t wmmaSpmm_16b8b_template(
    int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), 1);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    if(vec_length == 8)
        wmmaSpmm_kernel_16b8b8v<int, int, VecType, int, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    else
        wmmaSpmm_kernel_16b8b<int, int, VecType, int, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    return cudaGetLastError();
}

cudaError_t wmmaSpmm_16b8b(int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    switch(vec_length){
        case 2:
            return wmmaSpmm_16b8b_template<int, int, 1, 16, 128, 32, 4, 2>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const int *>(values), rhs_matrix, output_matrix);
            break;
        case 4:
            return wmmaSpmm_16b8b_template<int, long long, 1, 16, 128, 32, 4, 4>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const long long *>(values), rhs_matrix, output_matrix);
            break;
        case 8:
            return wmmaSpmm_16b8b_template<int, long long, 1, 16, 128, 32, 4, 8>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const long long *>(values), rhs_matrix, output_matrix);
            break;
        default:
            printf("Unsupported Vector Length!\n");
            return cudaGetLastError();
    }
}

cudaError_t wmmaSpmm_12b8b(int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    switch(vec_length){
        case 2:
            return wmmaSpmm_16b8b_template<int, int, 1, 16, 128, 32, 4, 2>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const int *>(values), rhs_matrix, output_matrix);
            break;
        case 4:
            return wmmaSpmm_16b8b_template<int, long long, 1, 16, 128, 32, 4, 4>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const long long *>(values), rhs_matrix, output_matrix);
            break;
        case 8:
            return wmmaSpmm_16b8b_template<int, long long, 1, 16, 128, 32, 4, 8>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const long long *>(values), rhs_matrix, output_matrix);
            break;
        default:
            printf("Unsupported Vector Length!\n");
            return cudaGetLastError();
    }
}

//16-bit 16-bit Tile_N = 64 with 4 warps
template <typename IndexType, typename VecType, int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t wmmaSpmm_16b_template(
    int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), 1);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    if(vec_length == 8)
        wmmaSpmm_kernel_16b8v<int, int, VecType, int, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    else
        wmmaSpmm_kernel_16b<int, int, VecType, int, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    return cudaGetLastError();
}

cudaError_t wmmaSpmm_16b(int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    switch(vec_length){
        case 2:
            return wmmaSpmm_16b_template<int, int, 1, 16, 64, 32, 4, 2>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const int *>(values), rhs_matrix, output_matrix);
            break;
        case 4:
            return wmmaSpmm_16b_template<int, long long, 1, 16, 64, 32, 4, 4>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const long long *>(values), rhs_matrix, output_matrix);
            break;
        case 8:
            return wmmaSpmm_16b_template<int, long long, 1, 16, 64, 32, 4, 8>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const long long *>(values), rhs_matrix, output_matrix);
            break;
        default:
            printf("Unsupported Vector Length!\n");
            return cudaGetLastError();
    }
}
//cudaError_t wmmaSpmm_8b4b4v(int m_vec, int vec_length, int n, int k, 
//    const int* __restrict__ row_indices, 
//    const int* __restrict__ row_offsets,
//    const int* __restrict__ column_indices,
//    const long long* __restrict__ values,
//    const int* __restrict__ rhs_matrix,
//    int* __restrict__ output_matrix)
//{
//    printf("Incorrect vector type\n");
//    return cudaGetLastError();
//}

//template <typename IndexType, int Tile_M, int Tile_K, int Tile_N, int WarpWidth>
//cudaError_t wmmaSpmmEx_8bit(
//    int m_vec, int vec_length, int k, int n, 
//    const int* __restrict__ row_indices, 
//    const int* __restrict__ row_offsets,
//    const int* __restrict__ column_indices,
//    const int* __restrict__ values,
//    const int* __restrict__ rhs_matrix,
//    int* __restrict__ output_matrix)
//{
//    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(k) / Tile_N), 1);
//    dim3 block_dim(WarpWidth, Tile_M, 1);
//    switch(vec_length){
//        //case 2:
//        //    //printf("V=2\n");
//        //    wmmaSpmmKernel2<int, int, short, int4, Tile_K, Tile_N, WarpWidth, 2><<<grid_dim, block_dim>>>(
//        //        m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
//        //    break;
//        case 4:
//            //printf("V=4\n");
//            wmmaSpmmKernel4<int, int, int, int, Tile_K, Tile_N, WarpWidth, 4><<<grid_dim, block_dim>>>(
//                m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
//            break;
//        //case 8:
//        //    //printf("V=8\n");
//        //    wmmaSpmmKernel8<int, int, int2, int4, int2, Tile_K, Tile_N, WarpWidth, 8><<<grid_dim, block_dim>>>(
//        //        m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
//        //    break;
//        default:
//            printf("Unsupported Vector Length!\n");
//    }
//
//    return cudaGetLastError();
//}
//
//// Function for 8-bit int
//cudaError_t wmmaSpmm(int m_vec, int vec_length, int k, int n, 
//    const int* __restrict__ row_indices, 
//    const int* __restrict__ row_offsets,
//    const int* __restrict__ column_indices,
//    const int* __restrict__ values,
//    const int* __restrict__ rhs_matrix,
//    int* __restrict__ output_matrix)
//{
//    //printf("8-bit wmmaSpmm\n");
//    return wmmaSpmmEx_8bit<int, 1, 32, 64, 32>(m_vec, vec_length, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
//}
//
//// Function for 4-bit int
//cudaError_t wmmaSpmm(int m_vec, int vec_length, int k, int n, 
//    const int* __restrict__ row_indices, 
//    const int* __restrict__ row_offsets,
//    const int* __restrict__ column_indices,
//    const short* __restrict__ values,
//    const int* __restrict__ rhs_matrix,
//    int* __restrict__ output_matrix)
//{
//    //printf("4-bit wmmaSpmm\n");
//    //return wmmaSpmm_4b_template<int, 1, 64, 64, 32>(m_vec, vec_length, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
//    return wmmaSpmm_4b_template<int, 1, 64, 128, 32>(m_vec, vec_length, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
//}
//
//// Function for 4-bit int 8v
//cudaError_t wmmaSpmm_4b8v(int m_vec, int vec_length, int k, int n, 
//    const int* __restrict__ row_indices, 
//    const int* __restrict__ row_offsets,
//    const int* __restrict__ column_indices,
//    const int* __restrict__ values,
//    const int* __restrict__ rhs_matrix,
//    int* __restrict__ output_matrix)
//{
//    //printf("4-bit wmmaSpmm\n");
//    //printf("4-bit 8v wmmaSpMM\n");
//    //return wmmaSpmm_4b_template<int, 1, 64, 64, 32>(m_vec, vec_length, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
//    return wmmaSpmm_4b_template<int, int, 1, 32, 128, 32>(m_vec, vec_length, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
//}
//// Function for 4-bit int 4v
//cudaError_t wmmaSpmm_4b8v(int m_vec, int vec_length, int k, int n, 
//    const int* __restrict__ row_indices, 
//    const int* __restrict__ row_offsets,
//    const int* __restrict__ column_indices,
//    const short* __restrict__ values,
//    const int* __restrict__ rhs_matrix,
//    int* __restrict__ output_matrix)
//{
//    //printf("4-bit wmmaSpmm\n");
//    //return wmmaSpmm_4b_template<int, 1, 64, 64, 32>(m_vec, vec_length, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
//    return wmmaSpmm_4b_template<int, short, 1, 64, 128, 32>(m_vec, vec_length, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
//}
//// Function for mixed precision
//cudaError_t wmmaSpmm(int m_vec, int vec_length, int k, int n, 
//    const int* __restrict__ row_indices, 
//    const int* __restrict__ row_offsets,
//    const int* __restrict__ column_indices,
//    const half* __restrict__ values,
//    const half* __restrict__ rhs_matrix,
//    float* __restrict__ output_matrix)
//{
//    return wmmaSpmmEx<float4, int, 1, 32, 64, 32>(m_vec, vec_length, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
//}
//
//// Function for half precision
//cudaError_t wmmaSpmm(int m_vec, int vec_length, int k, int n, 
//    const int* __restrict__ row_indices, 
//    const int* __restrict__ row_offsets,
//    const int* __restrict__ column_indices,
//    const half* __restrict__ values,
//    const half* __restrict__ rhs_matrix,
//    half* __restrict__ output_matrix)
//{
//    return wmmaSpmmEx<float4, int, 1, 32, 64, 32>(m_vec, vec_length, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
//}
//
//// Function for single precision
//cudaError_t wmmaSpmm(int m_vec, int vec_length, int k, int n,
//    const int* __restrict__ row_indices,
//    const int* __restrict__ row_offsets,
//    const int* __restrict__ column_indices,
//    const float* __restrict__ values,
//    const float* __restrict__ rhs_matrix,
//    float* __restrict__ output_matrix)
//{
//    printf("wmmaSpmm doesn't support float input.\n");
//    return cudaSuccess;
//}

}
