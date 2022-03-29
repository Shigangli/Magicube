#include <torch/extension.h>



torch::Tensor batched_deq_spmm_mma_8b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits_lhs,
    int bits_rhs,
    float scale);

torch::Tensor batched_deq_spmm_mma_16b8b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits_lhs,
    int bits_rhs,
    float scale);

torch::Tensor batched_deq_spmm_mma_4b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits_lhs,
    int bits_rhs,
    float scale);

torch::Tensor batched_deq_spmm_mma_8b4b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits_lhs,
    int bits_rhs,
    float scale);


torch::Tensor bspmm_4b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits_lhs,
    int bits_rhs,
    float scale){

    return batched_deq_spmm_mma_4b(
                              row_indices,
                              row_offsets,
                              column_indices,
                              values,
                              rhs_matrix,
                              vec_length,
                              bits_lhs,
                              bits_rhs,
                              scale);
}

torch::Tensor bspmm_8b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits_lhs,
    int bits_rhs,
    float scale){

    return batched_deq_spmm_mma_8b(
                              row_indices,
                              row_offsets,
                              column_indices,
                              values,
                              rhs_matrix,
                              vec_length,
                              bits_lhs,
                              bits_rhs,
                              scale);
}

torch::Tensor bspmm_8b4b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits_lhs,
    int bits_rhs,
    float scale){

    return batched_deq_spmm_mma_8b4b(
                                row_indices,
                                row_offsets,
                                column_indices,
                                values,
                                rhs_matrix,
                                vec_length,
                                bits_lhs,
                                bits_rhs,
                                scale);
}

torch::Tensor bspmm_16b8b(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    int vec_length,
    int bits_lhs,
    int bits_rhs,
    float scale){

    return batched_deq_spmm_mma_16b8b(
                                 row_indices,
                                 row_offsets,
                                 column_indices,
                                 values,
                                 rhs_matrix,
                                 vec_length,
                                 bits_lhs,
                                 bits_rhs,
                                 scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("bspmm_4b", &bspmm_4b, "Custom batched 4-bit SpMM kernel");
    m.def("bspmm_8b", &bspmm_8b, "Custom batched 8-bit SpMM kernel");
    m.def("bspmm_8b4b", &bspmm_8b4b, "Custom batched 8-bit 4-bit SpMM kernel");
    m.def("bspmm_16b8b", &bspmm_16b8b, "Custom batched 16-bit 8-bit SpMM kernel");
}
