#include <torch/extension.h>

torch::Tensor csr_softmax_cuda(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor values,
    float scaler,
    int vec_length);


torch::Tensor csr_softmax(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor values,
    float scaler,
    int vec_length)
{
    return csr_softmax_cuda(row_indices, row_offsets, values, scaler, vec_length);
}


torch::Tensor batched_csr_softmax_cuda(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor values,
    float scaler,
    int vec_length,
    int batch_size);


torch::Tensor batched_csr_softmax(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor values,
    float scaler,
    int vec_length,
    int batch_size)
{
    return batched_csr_softmax_cuda(row_indices, row_offsets, values, scaler, vec_length, batch_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("csr_softmax", &csr_softmax, "Custom Softmax kernel");
    m.def("bcsr_softmax", &batched_csr_softmax, "Custom Batched Softmax kernel");
}