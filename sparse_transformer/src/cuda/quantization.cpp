#include <torch/extension.h>

torch::Tensor quantization_cuda(torch::Tensor input_matrix, int bits, float scale);

torch::Tensor quantization(torch::Tensor input_matrix, int bits, float scale)
{
    return quantization_cuda(input_matrix, bits, scale);
}

torch::Tensor batched_quantization_cuda(torch::Tensor input_matrix, int bits, float scale);

torch::Tensor batched_quantization(torch::Tensor input_matrix, int bits, float scale)
{
    return batched_quantization_cuda(input_matrix, bits, scale);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("quantization", &quantization, "Custom symmetric quantization kernel");
    m.def("bquantization", &batched_quantization, "Custom Batched symmetric quantization kernel");
}
