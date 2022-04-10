#include "sddmm_utils/lhs_tile.h"
#include "sddmm_utils/rhs_tile.h"
#include "sddmm_utils/compute_utils.h"
#include "sddmm_utils/output_tile.h"
#include <cstdint>
#include <stdio.h>
#include <mma.h>
#include <cuda.h>
#include "cuda_fp16.h"
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>


using namespace nvcuda;


template <int Tile_K=64, int Tile_N=64, int VecLength=8>
__device__ void wmmaSddmm_kernel_4b_(int m_vec, int n, int k, 
    float scale,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_values)
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
    __shared__ int lhs_tile_array[Tile_K*VecLength/4];
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
        lhs_loader.Prefetch(0);
        //rhs_loader.Fetch(0);

        int i = 1;
        #pragma unroll
        for(; i < steps; i++){
            lhs_loader.LoadRowfromRegister(i-1);
            rhs_loader.Fetch(i-1);
            __syncthreads();
            lhs_loader.Prefetch(i);
            computer.TileMAC(i-1);
            //__syncthreads();
        }

        lhs_loader.LoadRowfromRegister(i-1);
        rhs_loader.Fetch(i-1);
        __syncthreads();
        computer.TileMAC(i-1);
    }
   
    //if(residue > 0){
    //    lhs_loader.ResidueLoad(residue);
    //    rhs_loader.ResidueLoad(residue);
    //    __syncthreads();
    //    computer.TileMACResidue();
    //} 

    //__syncthreads();
    wmmaOutputTile_4b<VecLength> output_tile_storer(lane_id, row_offset + n_index, workset, output_fragment, output_values, scale);
    output_tile_storer.Store();
    
}

template <int Tile_K=64, int Tile_N=64, int VecLength=8>
__device__ void wmmaSddmm_kernel_8b_(int m_vec, int n, int k, 
    float scale,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_values)
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
    __shared__ int lhs_tile_array[Tile_K*VecLength/2];
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
        lhs_loader.Prefetch(0);

        int i = 1;
        #pragma unroll
        for(; i < steps; i++){
            lhs_loader.LoadRowfromRegister(i-1);
            rhs_loader.Fetch(i-1);
            __syncthreads();
            lhs_loader.Prefetch(i);
            computer.TileMAC(i-1);
        }

        lhs_loader.LoadRowfromRegister(i-1);
        rhs_loader.Fetch(i-1);
        __syncthreads();
        computer.TileMAC(i-1);
    }
   
    wmmaOutputTile_8b<VecLength> output_tile_storer(lane_id, row_offset + n_index, workset, output_fragment, output_values, scale);
    output_tile_storer.Store();
}

template <int Tile_K=64, int Tile_N=64, int VecLength=2>
__device__ void wmmaSddmm_kernel_16b_(int m_vec, int n, int k, 
    float scale,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_values)
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
    __shared__ int lhs_tile_array[Tile_K*VecLength];
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
        lhs_loader.Prefetch(0);

        int i = 1;
        #pragma unroll
        for(; i < steps; i++){
            lhs_loader.LoadRowfromRegister(i-1);
            rhs_loader.Fetch(i-1);
            __syncthreads();
            lhs_loader.Prefetch(i);
            computer.TileMAC(i-1);
        }

        lhs_loader.LoadRowfromRegister(i-1);
        rhs_loader.Fetch(i-1);
        __syncthreads();
        computer.TileMAC(i-1);
    }
   
    wmmaOutputTile_16b<VecLength> output_tile_storer(lane_id, row_offset + n_index, workset, output_fragment, output_values, scale);
    output_tile_storer.Store();
}

template <int Tile_K=32, int Tile_N=64, int VecLength=8>
__device__ void wmmaSddmm_kernel_16b8v_(int m_vec, int n, int k, 
    float scale,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_values)
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
    //__shared__ int lhs_tile_array[Tile_K*VecLength + 7]; //padding
    __shared__ int lhs_tile_array[Tile_K*VecLength];
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
        lhs_loader.Prefetch(0);

        int i = 1;
        #pragma unroll
        for(; i < steps; i++){
            lhs_loader.LoadRowfromRegister(i-1);
            rhs_loader.Fetch(i-1);
            __syncthreads();
            lhs_loader.Prefetch(i);
            computer.TileMAC(i-1);
        }

        lhs_loader.LoadRowfromRegister(i-1);
        rhs_loader.Fetch(i-1);
        __syncthreads();
        computer.TileMAC(i-1);
    }
   
    wmmaOutputTile_16b8v<VecLength> output_tile_storer(lane_id, row_offset + n_index, workset, output_fragment, output_values, scale);
    output_tile_storer.Store();
}



template <int Tile_K=64, int Tile_N=64, int VecLength=8>
__global__ void wmmaSddmm_kernel_4b(int m_vec, int n, int k, 
    float scale,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_values){

    wmmaSddmm_kernel_4b_<Tile_K, Tile_N, VecLength>(m_vec, n, k, 
    scale,
    row_indices,
    row_offsets,
    column_indices,
    lhs_matrix,
    rhs_matrix,
    output_values);
}


template <int Tile_K=64, int Tile_N=64, int VecLength=8>
__global__ void batched_wmmaSddmm_kernel_4b(int m_vec, int n, int k, 
    float scale,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix_b,
    int lhs_stride,
    const int* __restrict__ rhs_matrix_b,
    int rhs_stride,
    half* __restrict__ output_values_b,
    int output_stride){

    int entry_idx = blockIdx.z;
    const int* lhs_matrix = lhs_matrix_b + entry_idx * lhs_stride;
    const int* rhs_matrix = rhs_matrix_b + entry_idx * rhs_stride;
    half* output_values = output_values_b + entry_idx * output_stride;

    wmmaSddmm_kernel_4b_<Tile_K, Tile_N, VecLength>(m_vec, n, k, 
    scale,
    row_indices,
    row_offsets,
    column_indices,
    lhs_matrix,
    rhs_matrix,
    output_values);
}


template <int Tile_K=64, int Tile_N=64, int VecLength=8>
__global__ void wmmaSddmm_kernel_8b(int m_vec, int n, int k, 
    float scale,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_values){

    wmmaSddmm_kernel_8b_<Tile_K, Tile_N, VecLength>(m_vec, n, k, 
    scale,
    row_indices,
    row_offsets,
    column_indices,
    lhs_matrix,
    rhs_matrix,
    output_values);
}


template <int Tile_K=64, int Tile_N=64, int VecLength=8>
__global__ void batched_wmmaSddmm_kernel_8b(int m_vec, int n, int k, 
    float scale,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix_b,
    int lhs_stride,
    const int* __restrict__ rhs_matrix_b,
    int rhs_stride,
    half* __restrict__ output_values_b,
    int output_stride){

    int entry_idx = blockIdx.z;
    const int* lhs_matrix = lhs_matrix_b + entry_idx * lhs_stride;
    const int* rhs_matrix = rhs_matrix_b + entry_idx * rhs_stride;
    half* output_values = output_values_b + entry_idx * output_stride;

    wmmaSddmm_kernel_8b_<Tile_K, Tile_N, VecLength>(m_vec, n, k, 
    scale,
    row_indices,
    row_offsets,
    column_indices,
    lhs_matrix,
    rhs_matrix,
    output_values);
}


template <int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t wmmaSddmm_4b_template(
    int m_vec, int n, int k, 
    float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_values)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), 1);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    wmmaSddmm_kernel_4b<Tile_K, Tile_N, VecLength><<<grid_dim, block_dim>>>(
        m_vec, n, k, scale, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
    return cudaGetLastError();
}




template <int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t batched_wmmaSddmm_4b_template(
    int m_vec, int n, int k, int batch_size,
    float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix_b,
    int lhs_stride,
    const int* __restrict__ rhs_matrix_b,
    int rhs_stride,
    half* __restrict__ output_values_b,
    int output_stride)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), batch_size);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    batched_wmmaSddmm_kernel_4b<Tile_K, Tile_N, VecLength><<<grid_dim, block_dim>>>(
        m_vec, n, k, scale, row_indices, row_offsets, column_indices, lhs_matrix_b, lhs_stride, rhs_matrix_b, rhs_stride, output_values_b, output_stride);
    return cudaGetLastError();
}



template <int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t wmmaSddmm_8b_template(
    int m_vec, int n, int k, 
    float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix,
    const int* __restrict__ rhs_matrix,
    half* __restrict__ output_values)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), 1);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    wmmaSddmm_kernel_8b<Tile_K, Tile_N, VecLength><<<grid_dim, block_dim>>>(
        m_vec, n, k, scale, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, output_values);
    return cudaGetLastError();
}



template <int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t batched_wmmaSddmm_8b_template(
    int m_vec, int n, int k, int batch_size,
    float scale,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ lhs_matrix_b,
    int lhs_stride,
    const int* __restrict__ rhs_matrix_b,
    int rhs_stride,
    half* __restrict__ output_values_b,
    int output_stride)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), batch_size);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    batched_wmmaSddmm_kernel_8b<Tile_K, Tile_N, VecLength><<<grid_dim, block_dim>>>(
        m_vec, n, k, scale, row_indices, row_offsets, column_indices, lhs_matrix_b, lhs_stride, rhs_matrix_b, rhs_stride, output_values_b, output_stride);
    return cudaGetLastError();
}





//8-bit Tile_K = 64 Tile_N = 64 with 8 warps
torch::Tensor batched_deq_sddmm_mma_8b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits,
    float scale)
{
    //lhs shape {batch, m, k}
    //rhs shape {batch, n, k}
    int num_items_per_int32 = 32 / bits;
    int k_int32 = lhs_matrix.size(-1);
    int k = k_int32 * num_items_per_int32;

    int m = lhs_matrix.size(-2);
    //int k = lhs_matrix.size(-1);
    int n = rhs_matrix.size(-2);
    //int batch_size = lhs_matrix.numel() / (m * k);
    int batch_size = rhs_matrix.size(-3);

    int m_vec = m / vec_length;
    int nnz = column_indices.numel();


    int lhs_stride = m * k_int32;
    int rhs_stride = n * k_int32;
    int output_stride = nnz * vec_length;

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(lhs_matrix.device());

    auto output_vals = torch::empty({batch_size, nnz * vec_length, }, options);

    switch(vec_length){
        case 2:
            batched_wmmaSddmm_8b_template<1, 64, 64, 32, 8, 2>(m_vec, n, k, batch_size, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<int *>(lhs_matrix.data<int>()), 
                lhs_stride, 
                reinterpret_cast<int *>(rhs_matrix.data<int>()),  
                rhs_stride, 
                reinterpret_cast<half *>(output_vals.data<torch::Half>()), 
                output_stride);
            break;
        case 4:
            batched_wmmaSddmm_8b_template<1, 64, 64, 32, 8, 4>(m_vec, n, k, batch_size, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<int *>(lhs_matrix.data<int>()), 
                lhs_stride, 
                reinterpret_cast<int *>(rhs_matrix.data<int>()),  
                rhs_stride, 
                reinterpret_cast<half *>(output_vals.data<torch::Half>()), 
                output_stride);                
            break;
        case 8:
            batched_wmmaSddmm_8b_template<1, 64, 64, 32, 8, 8>(m_vec, n, k, batch_size, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<int *>(lhs_matrix.data<int>()), 
                lhs_stride, 
                reinterpret_cast<int *>(rhs_matrix.data<int>()),  
                rhs_stride, 
                reinterpret_cast<half *>(output_vals.data<torch::Half>()), 
                output_stride);   
            break;
        default:
            printf("Unsupported Vector Length!\n");
    }

    return output_vals;
}

//8-bit Tile_K = 64 Tile_N = 64 with 8 warps
torch::Tensor deq_sddmm_mma_8b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits,
    float scale)
{
    //lhs shape {m, k}
    //rhs shape {n, k}
    int num_items_per_int32 = 32 / bits;
    int k_int32 = lhs_matrix.size(-1);
    int k = k_int32 * num_items_per_int32;

    int m = lhs_matrix.size(-2);
    //int k = lhs_matrix.size(-1);
    int n = rhs_matrix.size(-2);

    int m_vec = m / vec_length;
    int nnz = column_indices.numel();


    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(lhs_matrix.device());

    auto output_vals = torch::empty({nnz * vec_length, }, options);

    switch(vec_length){
        case 2:
            wmmaSddmm_8b_template<1, 64, 64, 32, 8, 2>(m_vec, n, k, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<int *>(lhs_matrix.data<int>()), 
                reinterpret_cast<int *>(rhs_matrix.data<int>()),  
                reinterpret_cast<half *>(output_vals.data<torch::Half>()));
            break;
        case 4:
            wmmaSddmm_8b_template<1, 64, 64, 32, 8, 4>(m_vec, n, k, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<int *>(lhs_matrix.data<int>()),
                reinterpret_cast<int *>(rhs_matrix.data<int>()),
                reinterpret_cast<half *>(output_vals.data<torch::Half>()));                
            break;
        case 8:
            wmmaSddmm_8b_template<1, 64, 64, 32, 8, 8>(m_vec, n, k, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<int *>(lhs_matrix.data<int>()), 
                reinterpret_cast<int *>(rhs_matrix.data<int>()), 
                reinterpret_cast<half *>(output_vals.data<torch::Half>()));   
            break;
        default:
            printf("Unsupported Vector Length!\n");
    }

    return output_vals;
}

//4-bit Tile_K = 64 Tile_N = 64 with 8 warps
torch::Tensor batched_deq_sddmm_mma_4b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits,
    float scale)
{
    //lhs shape {batch, m, k}
    //rhs shape {batch, n, k}

    int m = lhs_matrix.size(-2);

    int num_items_per_int32 = 32 / bits;
    int k_int32 = lhs_matrix.size(-1);
    int k = k_int32 * num_items_per_int32;

    int n = rhs_matrix.size(-2);

    //int batch_size = lhs_matrix.numel() / (m * k);
    int batch_size = lhs_matrix.size(-3);

    int m_vec = m / vec_length;
    int nnz = column_indices.numel();


    int lhs_stride = m * k_int32;
    int rhs_stride = n * k_int32;
    int output_stride = nnz * vec_length;

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(lhs_matrix.device());

    auto output_vals = torch::empty({batch_size, nnz * vec_length, }, options);

    switch(vec_length){
        case 2:
            batched_wmmaSddmm_4b_template<1, 64, 64, 32, 8, 2>(m_vec, n, k, batch_size, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<int *>(lhs_matrix.data<int>()), 
                lhs_stride, 
                reinterpret_cast<int *>(rhs_matrix.data<int>()),  
                rhs_stride, 
                reinterpret_cast<half *>(output_vals.data<torch::Half>()), 
                output_stride);
            break;
        case 4:
            batched_wmmaSddmm_4b_template<1, 64, 64, 32, 8, 4>(m_vec, n, k, batch_size, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<int *>(lhs_matrix.data<int>()), 
                lhs_stride, 
                reinterpret_cast<int *>(rhs_matrix.data<int>()),  
                rhs_stride, 
                reinterpret_cast<half *>(output_vals.data<torch::Half>()), 
                output_stride);                
            break;
        case 8:
            batched_wmmaSddmm_4b_template<1, 64, 64, 32, 8, 8>(m_vec, n, k, batch_size, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<int *>(lhs_matrix.data<int>()), 
                lhs_stride, 
                reinterpret_cast<int *>(rhs_matrix.data<int>()),  
                rhs_stride, 
                reinterpret_cast<half *>(output_vals.data<torch::Half>()), 
                output_stride);   
            break;
        default:
            printf("Unsupported Vector Length!\n");
    }

    return output_vals;
}

//4-bit Tile_K = 64 Tile_N = 64 with 8 warps
torch::Tensor deq_sddmm_mma_4b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits,
    float scale)
{
    //lhs shape {m, k}
    //rhs shape {n, k}

    int num_items_per_int32 = 32 / bits;
    int k_int32 = lhs_matrix.size(-1);
    int k = k_int32 * num_items_per_int32;

    int m = lhs_matrix.size(-2);
    //int k = lhs_matrix.size(-1);
    int n = rhs_matrix.size(-2);

    int m_vec = m / vec_length;
    int nnz = column_indices.numel();


    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(lhs_matrix.device());

    auto output_vals = torch::empty({nnz * vec_length, }, options);

    switch(vec_length){
        case 2:
            wmmaSddmm_4b_template<1, 64, 64, 32, 8, 2>(m_vec, n, k, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<int *>(lhs_matrix.data<int>()), 
                reinterpret_cast<int *>(rhs_matrix.data<int>()),  
                reinterpret_cast<half *>(output_vals.data<torch::Half>()));
            break;
        case 4:
            wmmaSddmm_4b_template<1, 64, 64, 32, 8, 4>(m_vec, n, k, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<int *>(lhs_matrix.data<int>()),
                reinterpret_cast<int *>(rhs_matrix.data<int>()),
                reinterpret_cast<half *>(output_vals.data<torch::Half>()));                
            break;
        case 8:
            wmmaSddmm_4b_template<1, 64, 64, 32, 8, 8>(m_vec, n, k, scale,
                row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(), 
                reinterpret_cast<int *>(lhs_matrix.data<int>()), 
                reinterpret_cast<int *>(rhs_matrix.data<int>()), 
                reinterpret_cast<half *>(output_vals.data<torch::Half>()));   
            break;
        default:
            printf("Unsupported Vector Length!\n");
    }

    return output_vals;

}
