#include "../include/wmma_sddmm.cuh"
#include "sddmm_utils/lhs_tile.h"
#include "sddmm_utils/rhs_tile.h"
#include "sddmm_utils/compute_utils.h"
#include "sddmm_utils/output_tile.h"
#include <cstdint>
#include <stdio.h>
#include <mma.h>

using namespace nvcuda;


namespace sddmm{

template <int Tile_K=64, int Tile_N=64, int VecLength=8>
__global__ void wmmaSddmm_kernel_4b(int m_vec, int n, int k, 
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_values)
{

    int m_index_vec = blockIdx.x;
    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset = __ldg(row_offsets + m_index_vec*2);
    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset;
    int n_index = blockIdx.y * Tile_N;

    // If this thread block has no nonzeros in the row to process, exit early
    if(n_index >= nonzeros) return;

    int workset = 0;
    if(n_index + Tile_N <= nonzeros)
        workset = Tile_N;
    else
	workset = nonzeros - n_index;

    int m_index = m_index_vec * VecLength;
    const int lane_id = threadIdx.x;

    // Each int32 has 8 4-bit integers and double buffers
    //__shared__ int lhs_tile_array[Tile_K*VecLength/4];
    //single buffer
    __shared__ int lhs_tile_array[Tile_K*VecLength/8];
    __shared__ int column_indices_tile_array[Tile_N];

    if(lane_id < workset)
        column_indices_tile_array[lane_id] = __ldg(column_indices + row_offset + n_index + lane_id) * (k / 8);
   
    __syncthreads();
    __align__(16) int lhs_prefetch[1] = {};
    __align__(16) int rhs_prefetch[2] = {};

    int* lhs_tile = lhs_tile_array;
    int* column_indices_tile = column_indices_tile_array;

    // Initialize the pointers to the lhs matrix
    // One int32 has eight 4-bit integers
    wmma_lhs_4b<Tile_K * VecLength / 8, Tile_K / 8, VecLength * 4, 4> lhs_loader(
        k / 8, m_index, lane_id, lhs_matrix, lhs_prefetch, lhs_tile
    );

    // Initialize the pointers to the rhs matrix
    wmma_rhs_4b<Tile_K / 8, 4, Tile_K / 32> rhs_loader(
        workset, lane_id, rhs_matrix, column_indices_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    __align__(16) int output_fragment[2] = {};
    wmmaComputeUtils_4b<4 * VecLength, Tile_K / 32> computer(lane_id, workset, lhs_tile, rhs_prefetch, output_fragment);

    int steps = k / Tile_K;

    if(steps > 0){
        #pragma unroll
        for(int i=0; i < steps; i++){
            lhs_loader.Fetch(i);
            rhs_loader.Fetch(i);
            __syncthreads();
            computer.TileMAC(i);
            __syncthreads(); //need by single buffer
        }
    }
   
    //if(residue > 0){
    //    lhs_loader.ResidueLoad(residue);
    //    rhs_loader.ResidueLoad(residue);
    //    __syncthreads();
    //    computer.TileMACResidue();
    //} 

    //__syncthreads();
    wmmaOutputTile_4b<VecLength> output_tile_storer(lane_id, row_offset + n_index, workset, output_fragment, output_values);
    output_tile_storer.Store();
    
}

template <int Tile_K=64, int Tile_N=64, int VecLength=8>
__global__ void wmmaSddmm_kernel_8b(int m_vec, int n, int k, 
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_values)
{

    int m_index_vec = blockIdx.x;
    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset = __ldg(row_offsets + m_index_vec*2);
    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset;
    int n_index = blockIdx.y * Tile_N;

    // If this thread block has no nonzeros in the row to process, exit early
    if(n_index >= nonzeros) return;

    int workset = 0;
    if(n_index + Tile_N <= nonzeros)
        workset = Tile_N;
    else
	workset = nonzeros - n_index;

    int m_index = m_index_vec * VecLength;
    const int lane_id = threadIdx.x;

    // Each int32 has 4 8-bit integers and double buffers
    //__shared__ int lhs_tile_array[Tile_K*VecLength/2];
    //single buffer
    __shared__ int lhs_tile_array[Tile_K*VecLength/4];
    __shared__ int column_indices_tile_array[Tile_N];

    if(lane_id < workset)
        column_indices_tile_array[lane_id] = __ldg(column_indices + row_offset + n_index + lane_id) * (k / 4);
   
    __syncthreads();
    __align__(16) int lhs_prefetch[1] = {};
    __align__(16) int rhs_prefetch[4] = {};

    int* lhs_tile = lhs_tile_array;
    int* column_indices_tile = column_indices_tile_array;

    // Initialize the pointers to the lhs matrix
    // One int32 has 4 8-bit integers
    wmma_lhs_8b<Tile_K * VecLength / 4, Tile_K / 4, VecLength * 4, 4> lhs_loader(
        k / 4, m_index, lane_id, lhs_matrix, lhs_prefetch, lhs_tile
    );

    // Initialize the pointers to the rhs matrix
    wmma_rhs_8b<Tile_K / 4, 4, Tile_K / 16> rhs_loader(
        workset, lane_id, rhs_matrix, column_indices_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    __align__(16) int output_fragment[2] = {};
    wmmaComputeUtils_8b<4 * VecLength, Tile_K / 16> computer(lane_id, workset, lhs_tile, rhs_prefetch, output_fragment);

    int steps = k / Tile_K;

    if(steps > 0){
        #pragma unroll
        for(int i=0; i < steps; i++){
            lhs_loader.Fetch(i);
            rhs_loader.Fetch(i);
            __syncthreads();
            computer.TileMAC(i);
            __syncthreads(); //need by single buffer
        }
    }
   
    wmmaOutputTile_8b<VecLength> output_tile_storer(lane_id, row_offset + n_index, workset, output_fragment, output_values);
    output_tile_storer.Store();
}

template <int Tile_K=64, int Tile_N=64, int VecLength=2>
__global__ void wmmaSddmm_kernel_16b(int m_vec, int n, int k, 
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_values)
{
    int m_index_vec = blockIdx.x;
    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset = __ldg(row_offsets + m_index_vec*2);
    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset;
    int n_index = blockIdx.y * Tile_N;

    // If this thread block has no nonzeros in the row to process, exit early
    if(n_index >= nonzeros) return;

    int workset = 0;
    if(n_index + Tile_N <= nonzeros)
        workset = Tile_N;
    else
	workset = nonzeros - n_index;

    int m_index = m_index_vec * VecLength;
    const int lane_id = threadIdx.x;

    // Each int32 has 2 16-bit integers and double buffers
    //__shared__ int lhs_tile_array[Tile_K*VecLength];
    //single buffer
    __shared__ int lhs_tile_array[Tile_K*VecLength/2];
    __shared__ int column_indices_tile_array[Tile_N];

    if(lane_id < workset)
        column_indices_tile_array[lane_id] = __ldg(column_indices + row_offset + n_index + lane_id) * (k / 2);
   
    __syncthreads();
    __align__(16) int lhs_prefetch[1] = {};
    __align__(16) int rhs_prefetch[8] = {};

    int* lhs_tile = lhs_tile_array;
    int* column_indices_tile = column_indices_tile_array;

    // Initialize the pointers to the lhs matrix
    // One int32 has 2 16-bit integers
    wmma_lhs_16b<Tile_K * VecLength / 2, Tile_K / 2, VecLength * 8, 8> lhs_loader(
        k / 2, m_index, lane_id, lhs_matrix, lhs_prefetch, lhs_tile
    );

    // Initialize the pointers to the rhs matrix
    wmma_rhs_16b<Tile_K / 2, 4, Tile_K / 16> rhs_loader(
        workset, lane_id, rhs_matrix, column_indices_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    __align__(16) int output_fragment[4] = {};
    wmmaComputeUtils_16b<8 * VecLength, Tile_K / 16> computer(lane_id, workset, lhs_tile, rhs_prefetch, output_fragment);

    int steps = k / Tile_K;

    if(steps > 0){
        #pragma unroll
        for(int i=0; i < steps; i++){
            lhs_loader.Fetch(i);
            rhs_loader.Fetch(i);
            __syncthreads();
            computer.TileMAC(i);
            __syncthreads(); //need by single buffer
        }
    }
   
    wmmaOutputTile_16b<VecLength> output_tile_storer(lane_id, row_offset + n_index, workset, output_fragment, output_values);
    output_tile_storer.Store();
}

template <int Tile_K=32, int Tile_N=64, int VecLength=8>
__global__ void wmmaSddmm_kernel_16b8v(int m_vec, int n, int k, 
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_values)
{

    int m_index_vec = blockIdx.x;
    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset = __ldg(row_offsets + m_index_vec*2);
    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset;
    int n_index = blockIdx.y * Tile_N;

    // If this thread block has no nonzeros in the row to process, exit early
    if(n_index >= nonzeros) return;

    int workset = 0;
    if(n_index + Tile_N <= nonzeros)
        workset = Tile_N;
    else
	workset = nonzeros - n_index;

    int m_index = m_index_vec * VecLength;
    const int lane_id = threadIdx.x;

    // Each int32 has 2 16-bit integers and double buffers
    //__shared__ int lhs_tile_array[Tile_K*VecLength];
    //single buffer
    __shared__ int lhs_tile_array[Tile_K*VecLength/2];
    __shared__ int column_indices_tile_array[Tile_N];

    if(lane_id < workset)
        column_indices_tile_array[lane_id] = __ldg(column_indices + row_offset + n_index + lane_id) * (k / 2);
   
    __syncthreads();
    __align__(16) int lhs_prefetch[1] = {};
    __align__(16) int rhs_prefetch[4] = {};

    int* lhs_tile = lhs_tile_array;
    int* column_indices_tile = column_indices_tile_array;

    // Initialize the pointers to the lhs matrix
    // One int32 has 2 16-bit integers
    wmma_lhs_16b<Tile_K * VecLength / 2, Tile_K / 2, VecLength * 8, 8> lhs_loader(
        k / 2, m_index, lane_id, lhs_matrix, lhs_prefetch, lhs_tile
    );

    // Initialize the pointers to the rhs matrix
    wmma_rhs_16b<Tile_K / 2, 4, Tile_K / 16> rhs_loader(
        workset, lane_id, rhs_matrix, column_indices_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    __align__(16) int output_fragment[8] = {};
    wmmaComputeUtils_16b8v<8 * VecLength, Tile_K / 16> computer(lane_id, workset, lhs_tile, rhs_prefetch, output_fragment);

    int steps = k / Tile_K;

    if(steps > 0){
        #pragma unroll
        for(int i=0; i < steps; i++){
            lhs_loader.Fetch(i);
            rhs_loader.Fetch(i);
            __syncthreads();
            computer.TileMAC(i);
            __syncthreads(); //need by single buffer
        }
    }
   
    wmmaOutputTile_16b8v<VecLength> output_tile_storer(lane_id, row_offset + n_index, workset, output_fragment, output_values);
    output_tile_storer.Store();
}


template <int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t wmmaSddmm_4b_template(
    int m_vec, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_values)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), 1);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    wmmaSddmm_kernel_4b<Tile_K, Tile_N, VecLength><<<grid_dim, block_dim>>>(
        m_vec, n, k, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
    return cudaGetLastError();
}

//4-bit Tile_K = 64 Tile_N = 32 with 4 warps
cudaError_t wmmaSddmm_4b(int m_vec, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_values,
    int vec_length)
{
    switch(vec_length){
        case 2:
            return wmmaSddmm_4b_template<1, 64, 64, 32, 8, 2>(m_vec, n, k, row_indices, 
            	   row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
	    break;
        case 4:
            return wmmaSddmm_4b_template<1, 64, 64, 32, 8, 4>(m_vec, n, k, row_indices, 
            	   row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
	    break;
        case 8:
            return wmmaSddmm_4b_template<1, 64, 64, 32, 8, 8>(m_vec, n, k, row_indices, 
            	   row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
	    break;
        default:
            printf("Unsupported Vector Length!\n");
            return cudaGetLastError();
    }
}

template <int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t wmmaSddmm_8b_template(
    int m_vec, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_values)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), 1);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    wmmaSddmm_kernel_8b<Tile_K, Tile_N, VecLength><<<grid_dim, block_dim>>>(
        m_vec, n, k, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
    return cudaGetLastError();
}

cudaError_t wmmaSddmm_8b(int m_vec, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_values,
    int vec_length)
{
    switch(vec_length){
        case 2:
            return wmmaSddmm_8b_template<1, 64, 64, 32, 8, 2>(m_vec, n, k, row_indices, 
            	   row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
	    break;
        case 4:
            return wmmaSddmm_8b_template<1, 64, 64, 32, 8, 4>(m_vec, n, k, row_indices, 
            	   row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
	    break;
        case 8:
            return wmmaSddmm_8b_template<1, 64, 64, 32, 8, 8>(m_vec, n, k, row_indices, 
            	   row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
	    break;
        default:
            printf("Unsupported Vector Length!\n");
            return cudaGetLastError();
    }
}

template <int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t wmmaSddmm_16b_template(
    int m_vec, int n, int k, int vec_length,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_values)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), 1);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    if(vec_length == 8)
        wmmaSddmm_kernel_16b8v<Tile_K, Tile_N, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
    else
        wmmaSddmm_kernel_16b<Tile_K, Tile_N, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
    return cudaGetLastError();
}

cudaError_t wmmaSddmm_16b(int m_vec, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_values,
    int vec_length)
{
    switch(vec_length){
        case 2:
            return wmmaSddmm_16b_template<1, 64, 64, 32, 8, 2>(m_vec, n, k, vec_length, row_indices, 
            	   row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
	    break;
        case 4:
            return wmmaSddmm_16b_template<1, 64, 64, 32, 8, 4>(m_vec, n, k, vec_length, row_indices, 
            	   row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
	    break;
        case 8:
            return wmmaSddmm_16b_template<1, 32, 64, 32, 8, 8>(m_vec, n, k, vec_length, row_indices, 
            	   row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
	    break;
        default:
            printf("Unsupported Vector Length!\n");
            return cudaGetLastError();
    }
}

}
