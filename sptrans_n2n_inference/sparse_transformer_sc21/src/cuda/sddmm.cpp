#include <torch/extension.h>

torch::Tensor sddmm_cuda(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int vec_length);


torch::Tensor sddmm(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int vec_length)
{
    return sddmm_cuda(row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, vec_length);
}


torch::Tensor batched_sddmm_cuda(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int vec_length);


torch::Tensor batched_sddmm(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int vec_length)
{
    return batched_sddmm_cuda(row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, vec_length);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("sddmm", &sddmm, "Custom SDDMM kernel");
    m.def("bsddmm", &batched_sddmm, "Custom Batched SDDMM kernel  ");
}