#include <cuda.h>
#include "cuda_fp16.h"
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include <mma.h>

using namespace nvcuda;


template <int Bits=4>
__device__ void quantizationKernel_4b_(
    int m, int n, float scale,
    const half* __restrict__ input_matrix,
    int* __restrict__ output_matrix)
{

    int offset = 32 * 8 * blockIdx.x + threadIdx.x * 8;
    const int4 *input_matrix_ = reinterpret_cast<const int4 *>(input_matrix + offset);
    int4 input_buffer = __ldg(input_matrix_);

    int quantized_vec = 0;
    half *inputs = reinterpret_cast<half *>(&input_buffer);

    float tempf;
    int tempr;
    int mask = 15;
    #pragma unroll
    for (int i = 0; i < 8; i++){
        tempf = __half2float(inputs[i])*scale;
        if(tempf < -8.0)
            tempf = -8.0;
        if(tempf > 7.0)
            tempf = 7.0;
        tempr = (int)tempf;
        quantized_vec |= ((tempr & mask) << (i*Bits));
    }

    int output_offset = blockIdx.x * 32 + threadIdx.x;
    *(output_matrix + output_offset) = quantized_vec;
}


template <int Bits=8>
__device__ void quantizationKernel_8b_(
    int m, int n, float scale,
    const half* __restrict__ input_matrix,
    int* __restrict__ output_matrix)
{

    int offset = 32 * 8 * blockIdx.x + threadIdx.x * 8;
    const int4 *input_matrix_ = reinterpret_cast<const int4 *>(input_matrix + offset);
    int4 input_buffer = __ldg(input_matrix_);

    char quantized_vec[8] = {};
    half *inputs = reinterpret_cast<half *>(&input_buffer);

    float tempf;

    #pragma unroll
    for (int i = 0; i < 8; i++){
        tempf = __half2float(inputs[i])*scale;
        if(tempf < -128.0)
            tempf = -128.0;
        if(tempf > 127.0)
            tempf = 127.0;
        quantized_vec[i] = (char)tempf;
    }

    int output_offset = blockIdx.x * 64 + threadIdx.x * 2;
    *(reinterpret_cast<long long *>(output_matrix + output_offset)) = *(reinterpret_cast<long long *>(quantized_vec));
}



template <int Bits=4>
__global__ void quantizationKernel_4b(
    int m, int n, float scale,
    const half* __restrict__ input_matrix,
    int* __restrict__ output_matrix)
{
    quantizationKernel_4b_<Bits>(
        m, n, scale, input_matrix, output_matrix
    );
}

template <int Bits=4>
__global__ void batched_quantizationKernel_4b(
    int m, int n, int input_stride, int output_stride, float scale,
    const half* __restrict__ input_matrix_b,
    int* __restrict__ output_matrix_b)
{
    // Get the entry index
    int entry_idx = blockIdx.y;
    const half* input_matrix = input_matrix_b + entry_idx * input_stride;
    int* output_matrix = output_matrix_b + entry_idx * output_stride;

    quantizationKernel_4b_<Bits>(
        m, n, scale, input_matrix, output_matrix
    );
}


template <int Bits=8>
__global__ void quantizationKernel_8b(
    int m, int n, float scale,
    const half* __restrict__ input_matrix,
    int* __restrict__ output_matrix)
{
    quantizationKernel_8b_<Bits>(
        m, n, scale, input_matrix, output_matrix
    );
}

template <int Bits=8>
__global__ void batched_quantizationKernel_8b(
    int m, int n, int input_stride, int output_stride, float scale,
    const half* __restrict__ input_matrix_b,
    int* __restrict__ output_matrix_b)
{
    // Get the entry index
    int entry_idx = blockIdx.y;
    const half* input_matrix = input_matrix_b + entry_idx * input_stride;
    int* output_matrix = output_matrix_b + entry_idx * output_stride;

    quantizationKernel_8b_<Bits>(
        m, n, scale, input_matrix, output_matrix
    );
}



torch::Tensor quantization_cuda(
    torch::Tensor input_matrix,
    int bits,
    float scale)
{
    int m = input_matrix.size(-2);
    int n = input_matrix.size(-1);
    
    int num_items_per_int32 = 32 / bits;
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(input_matrix.device());
    auto output_matrix = torch::empty({m, n/num_items_per_int32}, options);

    int num_half_per_int4 = 8;
    dim3 block_dim(32, 1, 1);
    int grid_x = m * n / (32 * num_half_per_int4);  //TODO: support non-multiple-of-256
    dim3 grid_dim(grid_x, 1, 1);

    switch(bits){
        case 8:
            quantizationKernel_8b<8><<<grid_dim, block_dim>>>(
                m, n, scale,
                reinterpret_cast<half *>(input_matrix.data<torch::Half>()),
                reinterpret_cast<int *>(output_matrix.data<int>())
            );
            break;
        case 4:
            quantizationKernel_4b<4><<<grid_dim, block_dim>>>(
                m, n, scale,
                reinterpret_cast<half *>(input_matrix.data<torch::Half>()),
                reinterpret_cast<int *>(output_matrix.data<int>())
            );
            break;
    }

    return output_matrix;
}


torch::Tensor batched_quantization_cuda(
    torch::Tensor input_matrix,
    int bits,
    float scale)
{

    int m = input_matrix.size(-2);
    int n = input_matrix.size(-1);
    int batch_size = input_matrix.size(-3);

    int num_items_per_int32 = 32 / bits;
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(input_matrix.device());
    auto output_matrix = torch::empty({batch_size, m, n/num_items_per_int32}, options);

    int num_half_per_int4 = 8;
    dim3 block_dim(32, 1, 1);
    int grid_x = m * n / (32 * num_half_per_int4);  //TODO: support non-multiple-of-256
    dim3 grid_dim(grid_x, batch_size, 1);

    int input_stride = m * n;
    int output_stride = m * n / num_items_per_int32;

    switch(bits){
        case 8:
            batched_quantizationKernel_8b<8><<<grid_dim, block_dim>>>(
                m, n, input_stride, output_stride, scale,
                reinterpret_cast<half *>(input_matrix.data<torch::Half>()),
                reinterpret_cast<int *>(output_matrix.data<int>())
            );
            break;
        case 4:
            batched_quantizationKernel_4b<4><<<grid_dim, block_dim>>>(
                m, n, input_stride, output_stride, scale,
                reinterpret_cast<half *>(input_matrix.data<torch::Half>()),
                reinterpret_cast<int *>(output_matrix.data<int>())
            );
            break;
    }

    return output_matrix;
}
