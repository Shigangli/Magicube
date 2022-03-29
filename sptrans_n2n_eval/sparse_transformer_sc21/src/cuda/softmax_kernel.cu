#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <stdio.h>

__device__ void print_val(int blockid, int threadid, float value){
    if (blockid == 0 && threadid == 0) printf("tid: %d, value is: %.8f\n", threadid, float(value));
}


template <int Len>
struct Array {
    __device__ __forceinline__ Array(){}
    __device__ __forceinline__ Array(float* inputs){
        #pragma unroll
        for (int i = 0; i < Len; i++){
            data[i] = inputs[i];
        }
    }

    float data[Len];
};

template <int Len>
struct ArrayMaxFunc{
    __device__ __forceinline__ Array<Len> operator()(
        const Array<Len>& p1, const Array<Len>& p2)
    {
        Array<Len> result;
        #pragma unroll
        for (int i = 0; i < Len; i ++){
            result.data[i] = p1.data[i] > p2.data[i] ? p1.data[i] : p2.data[i];
        }
        return result;
    }
};

template <int Len>
struct ArraySumFunc{
    __device__ __forceinline__ Array<Len> operator()(
        const Array<Len>& p1, const Array<Len>& p2)
    {
        Array<Len> result;
        #pragma unroll
        for (int i = 0; i < Len; i ++){
            result.data[i] = p1.data[i] + p2.data[i];
        }
        return result;
    }
};


template <int VecLength, typename LoadType>
__device__ __forceinline__ void vmax(float* a, LoadType b, const float scaler)
{
    half* b_h = reinterpret_cast<half*>(&b);
    
    #pragma unroll
    for (int i = 0; i < VecLength; i++){
        a[i] = a[i] > __half2float(b_h[i]) * scaler ? a[i] : __half2float(b_h[i]) * scaler;
    }
}


template <int VecLength, typename LoadType>
__device__ __forceinline__ void vexpsum(float* a, LoadType b, Array<VecLength> max, const float scaler)
{
    half* b_h = reinterpret_cast<half*>(&b);

    #pragma unroll
    for (int i = 0; i < VecLength; i++){
        a[i] += __expf(__half2float(b_h[i]) * scaler - max.data[i]);
    }
}


template <int VecLength, typename LoadType>
__device__ __forceinline__ void vsoftmax(LoadType a, Array<VecLength> max, Array<VecLength> sum, LoadType* out, const float scaler)
{
    half* a_h = reinterpret_cast<half*>(&a);
    half out_reg[VecLength];
    #pragma unroll
    for (int i = 0; i < VecLength; i++){
        out_reg[i] = __float2half(__expf(__half2float(a_h[i]) * scaler - max.data[i]) / sum.data[i]);
    }
    LoadType* out_reg_v = reinterpret_cast<LoadType*>(out_reg);
    *(out) = *(out_reg_v);
}


template <int VecLength, typename LoadType, int BlockDim>
__device__ void csrSoftmaxKernel_(
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const half* __restrict__ values,
    half* __restrict__ attn,
    const float scaler)
{
    int m_index = blockIdx.x; 
    m_index = __ldg(row_indices + m_index);

    int row_offset = __ldg(row_offsets + m_index);
    int nonzeros = __ldg(row_offsets + m_index + 1) - row_offset;


    // Specialized BlockReduce for a 1D block on type half
    typedef cub::BlockReduce<Array<VecLength>, BlockDim> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    __shared__ Array<VecLength> row_max;
    __shared__ Array<VecLength> row_sum;

    // Private register file that holds the loaded data
    float in_attn[VecLength] = {-1e+10};

    // Pointer to the input attention weight
    const LoadType* values_v = reinterpret_cast<const LoadType *>(values) + row_offset;
    LoadType* attn_v = reinterpret_cast<LoadType *>(attn) + row_offset;

    // First Run: get the maximum number of the current row
    for (int i = threadIdx.x; i < nonzeros; i += BlockDim){
        // Load data to register file
        vmax<VecLength, LoadType>(in_attn, __ldg(values_v + i), scaler);
    }
    Array<VecLength> local_max(in_attn);
    Array<VecLength> max_val = BlockReduce(temp_storage).Reduce(local_max, ArrayMaxFunc<VecLength>());

    if (threadIdx.x == 0) row_max = max_val;

    __syncthreads();

    // print_val(blockIdx.x, threadIdx.x, row_max.data[0]);

    #pragma unroll
    for (int i = 0; i < VecLength; i ++){
        in_attn[i] = 0.0f;
    }

    max_val = row_max;

    // print_val(blockIdx.x, threadIdx.x, max_val.data[0]);

    // Second Run: Compute the sum
    for (int i = threadIdx.x; i < nonzeros; i += BlockDim){
        vexpsum<VecLength, LoadType>(in_attn, __ldg(values_v + i), max_val, scaler);
    }

    Array<VecLength> local_sum(in_attn);
    Array<VecLength> sum_val = BlockReduce(temp_storage).Reduce(local_sum, ArraySumFunc<VecLength>());

    if (threadIdx.x == 0) row_sum = sum_val;

    __syncthreads();
    sum_val = row_sum;

    // print_val(blockIdx.x, threadIdx.x, sum_val.data[0]);

    // Last Run: Do softmax
    for (int i = threadIdx.x; i < nonzeros; i += BlockDim){
        vsoftmax<VecLength, LoadType>(__ldg(values_v + i), max_val, sum_val, attn_v + i, scaler);
    }
}

template <int VecLength, typename LoadType, int BlockDim>
__global__ void csrSoftmaxKernel(
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const half* __restrict__ values,
    half* __restrict__ attn,
    const float scaler)
{
    csrSoftmaxKernel_<VecLength, LoadType, BlockDim>(
        row_indices, row_offsets, values, attn, scaler
    );
}


template <int VecLength, typename LoadType, int BlockDim>
__global__ void batchedCsrSoftmaxKernel(
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const half* __restrict__ values_b,
    int values_stride,
    half* __restrict__ attn_b,
    int attn_stride,
    const float scaler)
{
    int entry_idx = blockIdx.y;
    const half* values = values_b + values_stride * entry_idx;
    half* attn = attn_b + attn_stride * entry_idx;

    csrSoftmaxKernel_<VecLength, LoadType, BlockDim>(
        row_indices, row_offsets, values, attn, scaler
    );
}


torch::Tensor csr_softmax_cuda(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor values,
    float scaler,
    int vec_length)
{
    int m = row_indices.size(0);

    dim3 block, grid;
    block.x = 128;
    grid.x = m;

    auto attn = torch::empty_like(values);

    switch(vec_length){
        case 8:
            csrSoftmaxKernel<8, float4, 128><<<grid, block>>>(
                row_indices.data<int>(), row_offsets.data<int>(),
                reinterpret_cast<half *>(values.data<torch::Half>()),
                reinterpret_cast<half *>(attn.data<torch::Half>()),
                scaler
            );
            break;
        case 4:
            csrSoftmaxKernel<4, float2, 128><<<grid, block>>>(
               row_indices.data<int>(), row_offsets.data<int>(),
               reinterpret_cast<half *>(values.data<torch::Half>()),
               reinterpret_cast<half *>(attn.data<torch::Half>()),
               scaler
           );
           break; 
        
        case 2:
            csrSoftmaxKernel<2, float, 128><<<grid, block>>>(
               row_indices.data<int>(), row_offsets.data<int>(),
               reinterpret_cast<half *>(values.data<torch::Half>()),
               reinterpret_cast<half *>(attn.data<torch::Half>()),
               scaler
           );
           break;
    }

    return attn;
}


torch::Tensor batched_csr_softmax_cuda(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor values,
    float scaler,
    int vec_length,
    int batch_size)
{
    int m = row_indices.size(0);
    
    dim3 block, grid;
    block.x = 128;
    grid.x = m;
    grid.y = batch_size;

    auto attn = torch::empty_like(values);

    int stride = values.numel()/batch_size;

    switch(vec_length){
        case 8:
            batchedCsrSoftmaxKernel<8, float4, 128><<<grid, block>>>(
                row_indices.data<int>(), row_offsets.data<int>(),
                reinterpret_cast<half *>(values.data<torch::Half>()),
                stride,
                reinterpret_cast<half *>(attn.data<torch::Half>()),
                stride, scaler
            );
            break;
        case 4:
            batchedCsrSoftmaxKernel<4, float2, 128><<<grid, block>>>(
               row_indices.data<int>(), row_offsets.data<int>(),
               reinterpret_cast<half *>(values.data<torch::Half>()),
               stride,
               reinterpret_cast<half *>(attn.data<torch::Half>()),
               stride,
               scaler
           );
           break; 
        
        case 2:
            batchedCsrSoftmaxKernel<2, float, 128><<<grid, block>>>(
               row_indices.data<int>(), row_offsets.data<int>(),
               reinterpret_cast<half *>(values.data<torch::Half>()),
               stride,
               reinterpret_cast<half *>(attn.data<torch::Half>()),
               stride,
               scaler
           );
           break;
    }

    return attn;
}