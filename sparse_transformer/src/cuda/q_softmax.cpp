#include <torch/extension.h>

torch::Tensor csr_softmax_cuda(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor values,
    float sqrt_dk,
    float scale,
    int vec_length,
    int bits);


torch::Tensor q_csr_softmax(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor values,
    float sqrt_dk,
    float scale,
    int vec_length,
    int bits)
{
    return csr_softmax_cuda(row_indices, row_offsets, values, sqrt_dk, scale, vec_length, bits);
}


torch::Tensor batched_csr_softmax_cuda(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor values,
    float sqrt_dk,
    float scale,
    int vec_length,
    int batch_size,
    int bits);


torch::Tensor q_batched_csr_softmax(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor values,
    float sqrt_dk,
    float scale,
    int vec_length,
    int batch_size,
    int bits)
{
    return batched_csr_softmax_cuda(row_indices, row_offsets, values, sqrt_dk, scale, vec_length, batch_size, bits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("q_csr_softmax", &q_csr_softmax, "Quantized Softmax kernel");
    m.def("q_bcsr_softmax", &q_batched_csr_softmax, "Quantized Batched Softmax kernel");
}
