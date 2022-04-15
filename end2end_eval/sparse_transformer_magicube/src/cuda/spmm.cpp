#include <torch/extension.h>

torch::Tensor spmm_cuda(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    int vec_length);

torch::Tensor spmm(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    int vec_length)
{
    return spmm_cuda(row_indices, row_offsets, column_indices, values, rhs_matrix, vec_length);
}



torch::Tensor batched_spmm_cuda(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    int vec_length);

torch::Tensor batched_spmm(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    int vec_length)
{
    return batched_spmm_cuda(row_indices, row_offsets, column_indices, values, rhs_matrix, vec_length);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("spmm", &spmm, "Custom SPMM kernel");
    m.def("bspmm", &batched_spmm, "Custom Batched SPMM kernel");
}