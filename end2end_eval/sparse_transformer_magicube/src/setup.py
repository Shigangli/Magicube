from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(
    name='sptrans',
    version='0.0.1',
    description='Custom library for Sparse Transformer for pytorch',
    author='Shigang Li',
    author_email='shigangli.cs@gmail.com',
    ext_modules=[
        CUDAExtension('sptrans.deq_sddmm', 
                      ['cuda/deq_sddmm.cpp', 'cuda/deq_sddmm_kernel.cu'],
                      extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_80', '-lcusparse', '--ptxas-options=-v', '-lineinfo']}),
        CUDAExtension('sptrans.deq_spmm', 
                      ['cuda/deq_spmm.cpp', 'cuda/deq_spmm_kernel.cu'],
                      extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_80', '-lcusparse', '--ptxas-options=-v', '-lineinfo']}),
        CUDAExtension('sptrans.q_softmax', 
                      ['cuda/q_softmax.cpp', 'cuda/q_softmax_kernel.cu'],
                      extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_80', '-lcusparse', '--ptxas-options=-v', '-lineinfo']}),
        CUDAExtension('sptrans.quantization', 
                      ['cuda/quantization.cpp', 'cuda/quantization_kernel.cu'],
                      extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_80', '-lcusparse', '--ptxas-options=-v', '-lineinfo']}),
        ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch']
)
