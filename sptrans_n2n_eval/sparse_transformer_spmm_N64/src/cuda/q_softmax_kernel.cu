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
__device__ __forceinline__ void vmax(float* a, LoadType b, const float sqrt_dk)
{
    half* b_h = reinterpret_cast<half*>(&b);
    
    #pragma unroll
    for (int i = 0; i < VecLength; i++){
        a[i] = a[i] > __half2float(b_h[i]) * sqrt_dk ? a[i] : __half2float(b_h[i]) * sqrt_dk;
    }
}


template <int VecLength, typename LoadType>
__device__ __forceinline__ void vexpsum(float* a, LoadType b, Array<VecLength> max, const float sqrt_dk)
{
    half* b_h = reinterpret_cast<half*>(&b);

    #pragma unroll
    for (int i = 0; i < VecLength; i++){
        a[i] += __expf(__half2float(b_h[i]) * sqrt_dk - max.data[i]);
    }
}


template <int VecLength, typename LoadType>
__device__ __forceinline__ void vsoftmax_4b(LoadType a, LoadType b, Array<VecLength> max, Array<VecLength> sum, char* out, const float sqrt_dk, const int alignment, const float scale, const bool in_range)
{
    half* a_h = reinterpret_cast<half*>(&a);
    half* b_h = reinterpret_cast<half*>(&b);

    char mask = 15;
    char out_c[VecLength];
    float tempf0;
    float tempf1;
    #pragma unroll
    for (int i = 0; i < VecLength; i++){
        tempf0 = __expf(__half2float(a_h[i]) * sqrt_dk - max.data[i]) / sum.data[i] * scale;
        tempf1 = __expf(__half2float(b_h[i]) * sqrt_dk - max.data[i]) / sum.data[i] * scale;

        if(tempf0 < -8.0)
            tempf0 = -8.0;
        if(tempf0 > 7.0)
            tempf0 = 7.0;

        if(!in_range){
            tempf1 = 0.0;
        }
        else{
            if(tempf1 < -8.0)
                tempf1 = -8.0;
            if(tempf1 > 7.0)
                tempf1 = 7.0;
        }

        out_c[i] = (((char)tempf0) & mask) | (((char)tempf1) << 4);
    }

    #pragma unroll
    for (int i = 0; i < VecLength; i++){
        *(out+i*alignment) = out_c[i];
    }
}


template <int VecLength, typename LoadType>
__device__ __forceinline__ void vsoftmax_8b(LoadType a, Array<VecLength> max, Array<VecLength> sum, char* out, const float sqrt_dk, const int alignment, const float scale)
{
    half* a_h = reinterpret_cast<half*>(&a);
    char out_c[VecLength];
    float tempf;

    #pragma unroll
    for (int i = 0; i < VecLength; i++){
        tempf = __expf(__half2float(a_h[i]) * sqrt_dk - max.data[i]) / sum.data[i];

        //quantization
        tempf *= scale;

        if(tempf < -128.0)
            tempf = -128.0;
        if(tempf > 127.0)
            tempf = 127.0;

        out_c[i] = (char)tempf;
    }

    #pragma unroll
    for (int i = 0; i < VecLength; i++){
        *(out+i*alignment) = out_c[i];
    }
}

template <int VecLength, typename LoadType>
__device__ __forceinline__ void vsoftmax_16b(LoadType a, Array<VecLength> max, Array<VecLength> sum, char* out, const float sqrt_dk, const int alignment, const float scale)
{
    half* a_h = reinterpret_cast<half*>(&a);
    short out_s[VecLength];
    float tempf;

    #pragma unroll
    for (int i = 0; i < VecLength; i++){
        tempf = __expf(__half2float(a_h[i]) * sqrt_dk - max.data[i]) / sum.data[i];

        //quantization
        tempf *= scale;

        if(tempf < -32768.0)
            tempf = -32768.0;
        if(tempf > 32767.0)
            tempf = 32767.0;

        out_s[i] = (short)tempf;
    }

    char* out_c = reinterpret_cast<char*>(out_s);
    #pragma unroll
    for (int i = 0; i < VecLength; i++){
	*(out+i*alignment*2) = out_c[i*2]; 
	*(out+i*alignment*2 + alignment) = out_c[i*2+1]; 
    }
}


template <int VecLength, typename LoadType, int BlockDim>
__device__ void csrSoftmaxKernel_(
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const half* __restrict__ values,
    int* __restrict__ attn,
    const float sqrt_dk,
    const float scale,
    int bits,
    int alignment)
{
    int m_index = blockIdx.x; 
    m_index = __ldg(row_indices + m_index);

    //int row_offset = __ldg(row_offsets + m_index);
    //int nonzeros = __ldg(row_offsets + m_index + 1) - row_offset;

    int row_offset = __ldg(row_offsets + m_index*2);
    int nonzeros = __ldg(row_offsets + m_index*2 + 1) - row_offset;

    // Specialized BlockReduce for a 1D block on type half
    typedef cub::BlockReduce<Array<VecLength>, BlockDim> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    __shared__ Array<VecLength> row_max;
    __shared__ Array<VecLength> row_sum;

    // Private register file that holds the loaded data
    float in_attn[VecLength] = {-1e+10};

    // Pointer to the input attention weight
    const LoadType* values_v = reinterpret_cast<const LoadType *>(values) + row_offset;
    //LoadType* attn_v = reinterpret_cast<LoadType *>(attn) + row_offset;
    int num_items_per_int32 = 32 / bits;
    char* attn_c = reinterpret_cast<char *>(attn + (row_offset*VecLength)/num_items_per_int32);

    // First Run: get the maximum number of the current row
    for (int i = threadIdx.x; i < nonzeros; i += BlockDim){
        // Load data to register file
        vmax<VecLength, LoadType>(in_attn, __ldg(values_v + i), sqrt_dk);
    }

    Array<VecLength> local_max(in_attn);
    Array<VecLength> max_val = BlockReduce(temp_storage).Reduce(local_max, ArrayMaxFunc<VecLength>());

    if (threadIdx.x == 0) row_max = max_val;  // only the result of threadIdx.x is deterministic

    __syncthreads();

    // print_val(blockIdx.x, threadIdx.x, row_max.data[0]);

    #pragma unroll
    for (int i = 0; i < VecLength; i ++){
        in_attn[i] = 0.0f;
    }

    max_val = row_max;  // broadcast from shared memory
    // print_val(blockIdx.x, threadIdx.x, max_val.data[0]);

    // Second Run: Compute the sum
    for (int i = threadIdx.x; i < nonzeros; i += BlockDim){
        vexpsum<VecLength, LoadType>(in_attn, __ldg(values_v + i), max_val, sqrt_dk);
    }

    Array<VecLength> local_sum(in_attn);
    Array<VecLength> sum_val = BlockReduce(temp_storage).Reduce(local_sum, ArraySumFunc<VecLength>());

    if (threadIdx.x == 0) row_sum = sum_val; // only the result of threadIdx.x is deterministic

    __syncthreads();
    sum_val = row_sum;  // broadcast from shared memory

    // print_val(blockIdx.x, threadIdx.x, sum_val.data[0]);

    // Do softmax with quantization
    if(bits == 16){
        for (int i = threadIdx.x; i < nonzeros; i += BlockDim){
            vsoftmax_16b<VecLength, LoadType>(__ldg(values_v + i), max_val, sum_val, attn_c + (i/alignment)*VecLength*alignment*2 + i%alignment, sqrt_dk, alignment, scale);
        }
    }
    else if(bits == 8){
        for (int i = threadIdx.x; i < nonzeros; i += BlockDim){
            vsoftmax_8b<VecLength, LoadType>(__ldg(values_v + i), max_val, sum_val, attn_c + (i/alignment)*VecLength*alignment + i%alignment, sqrt_dk, alignment, scale);
        }
    }
    else if(bits == 4){
        for (int i = threadIdx.x * 2; i < nonzeros; i += (BlockDim * 2)){
            vsoftmax_4b<VecLength, LoadType>(__ldg(values_v + i), __ldg(values_v + i + 1), max_val, sum_val, attn_c + (i/alignment)*VecLength*alignment/2 + (i/2) % (alignment/2), sqrt_dk, alignment/2, scale, (i+1) < nonzeros);
        }
    }
}


template <int VecLength, typename LoadType, int BlockDim>
__global__ void csrSoftmaxKernel(
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const half* __restrict__ values,
    int* __restrict__ attn,
    const float sqrt_dk,
    const float scale,
    int bits,
    int alignment)
{
    csrSoftmaxKernel_<VecLength, LoadType, BlockDim>(
        row_indices, row_offsets, values, attn, sqrt_dk, scale, bits, alignment
    );
}


template <int VecLength, typename LoadType, int BlockDim>
__global__ void batchedCsrSoftmaxKernel(
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const half* __restrict__ values_b,
    int values_stride,
    int* __restrict__ attn_b,
    int attn_stride,
    const float sqrt_dk,
    const float scale,
    int bits,
    int alignment)
{
    int entry_idx = blockIdx.y;
    const half* values = values_b + values_stride * entry_idx;
    int* attn = attn_b + attn_stride * entry_idx;

    csrSoftmaxKernel_<VecLength, LoadType, BlockDim>(
        row_indices, row_offsets, values, attn, sqrt_dk, scale, bits, alignment
    );
}


torch::Tensor csr_softmax_cuda(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor values,
    float sqrt_dk,
    float scale,
    int vec_length,
    int bits)
{
    int m = row_indices.size(0);

    dim3 block, grid;
    block.x = 128;
    grid.x = m;

    int alignment = 1;
    if(bits == 4)
        alignment = 32;    
    else if(bits == 8)
        alignment = 16;
    else if(bits == 16)
        alignment = 16;
    else
        printf("Unsupported precision for softmax!\n");

    //auto attn = torch::empty_like(values);

    int num_items_per_int32 = 32 / bits;

    int num_values = values.numel();
    int num_attn = num_values / num_items_per_int32;

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(values.device());
    auto attn = torch::zeros({num_attn, }, options);

    switch(vec_length){
        case 8:
            csrSoftmaxKernel<8, float4, 128><<<grid, block>>>(
                row_indices.data<int>(), row_offsets.data<int>(),
                reinterpret_cast<half *>(values.data<torch::Half>()),
                reinterpret_cast<int *>(attn.data<int>()),
                sqrt_dk,
                scale,
                bits,
                alignment
            );
            break;

        case 4:
            csrSoftmaxKernel<4, float2, 128><<<grid, block>>>(
               row_indices.data<int>(), row_offsets.data<int>(),
               reinterpret_cast<half *>(values.data<torch::Half>()),
               reinterpret_cast<int *>(attn.data<int>()),
               sqrt_dk,
               scale,
               bits,
               alignment
           );
           break; 
        
        case 2:
            csrSoftmaxKernel<2, float, 128><<<grid, block>>>(
               row_indices.data<int>(), row_offsets.data<int>(),
               reinterpret_cast<half *>(values.data<torch::Half>()),
               reinterpret_cast<int *>(attn.data<int>()),
               sqrt_dk,
               scale,
               bits,
               alignment
           );
           break;

        default:
            printf("Unsupported vec_length for softmax!\n");
    }

    return attn;
}


torch::Tensor batched_csr_softmax_cuda(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor values,
    float sqrt_dk,
    float scale,
    int vec_length,
    int batch_size,
    int bits)
{
    int m = row_indices.size(0);
    
    dim3 block, grid;
    block.x = 128;
    grid.x = m;
    grid.y = batch_size;

    int alignment = 1;
    if(bits == 4)
        alignment = 32;    
    else if(bits == 8)
        alignment = 16;
    else if(bits == 16)
        alignment = 16;
    else
        printf("Unsupported precision for softmax!\n");

    int num_items_per_int32 = 32 / bits;

    //auto attn = torch::empty_like(values);
    int values_stride = values.numel()/batch_size;
    int attn_stride = values_stride / num_items_per_int32;

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(values.device());
    auto attn = torch::zeros({batch_size, attn_stride}, options);


    switch(vec_length){
        case 8:
            batchedCsrSoftmaxKernel<8, float4, 128><<<grid, block>>>(
                row_indices.data<int>(), row_offsets.data<int>(),
                reinterpret_cast<half *>(values.data<torch::Half>()),
                values_stride,
                reinterpret_cast<int *>(attn.data<int>()),
                attn_stride,
                sqrt_dk,
                scale,
                bits,
                alignment
            );
            break;

        case 4:
            batchedCsrSoftmaxKernel<4, float2, 128><<<grid, block>>>(
               row_indices.data<int>(), row_offsets.data<int>(),
               reinterpret_cast<half *>(values.data<torch::Half>()),
               values_stride,
               reinterpret_cast<int *>(attn.data<int>()),
               attn_stride,
               sqrt_dk,
               scale,
               bits,
               alignment
           );
           break; 
        
        case 2:
            batchedCsrSoftmaxKernel<2, float, 128><<<grid, block>>>(
               row_indices.data<int>(), row_offsets.data<int>(),
               reinterpret_cast<half *>(values.data<torch::Half>()),
               values_stride,
               reinterpret_cast<int *>(attn.data<int>()),
               attn_stride,
               sqrt_dk,
               scale,
               bits,
               alignment
           );
           break;

        default:
            printf("Unsupported vec_length for softmax!\n");
    }

    return attn;
}
