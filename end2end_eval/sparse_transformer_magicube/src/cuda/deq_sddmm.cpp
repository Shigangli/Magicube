#include <torch/extension.h>

torch::Tensor deq_sddmm_mma_4b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits,
    float scale);

torch::Tensor deq_sddmm_mma_8b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits,
    float scale);

torch::Tensor batched_deq_sddmm_mma_4b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits,
    float scale);

torch::Tensor batched_deq_sddmm_mma_8b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits,
    float scale);

torch::Tensor sddmm_4b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits,
    float scale)
{
    return deq_sddmm_mma_4b(row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, vec_length, bits, scale);
}

torch::Tensor bsddmm_4b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits,
    float scale)
{
    return batched_deq_sddmm_mma_4b(row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, vec_length, bits, scale);
}

torch::Tensor sddmm_8b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits,
    float scale)
{
    return deq_sddmm_mma_8b(row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, vec_length, bits, scale);
}

torch::Tensor bsddmm_8b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor lhs_matrix,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits,
    float scale)
{
    return batched_deq_sddmm_mma_8b(row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, vec_length, bits, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("sddmm_4b", &sddmm_4b, "Custom SDDMM kernel with 4-bit inputs");
    m.def("sddmm_8b", &sddmm_8b, "Custom SDDMM kernel with 8-bit inputs");
    m.def("bsddmm_4b", &bsddmm_4b, "Custom batched SDDMM kernel with 4-bit inputs");
    m.def("bsddmm_8b", &bsddmm_8b, "Custom batched SDDMM kernel with 8-bit inputs");
}
