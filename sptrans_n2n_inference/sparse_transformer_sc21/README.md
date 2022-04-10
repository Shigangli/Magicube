# Sparse Transformer Inference

This repo provides a pytorch extension that speedup transformer inference with fixed structured sparsity. 

The end-to-end speedup & memory profiling can be obtained with `end_to_end.py`. 
* To profile the execution time of sparse transformer, launch `python3 end_to_end.py --model sparse` with nsight system.
* To profile the execution time of dense transformer, launch `python3 end_to_end.py --model dense` with nsight system.
* To profile the memory of sparse transformer, launch `python3 end_to_end.py --model sparse --mem` with nsight system.
* To profile the memory of dense transformer, launch `python3 end_to_end.py --model dense --mem` with nsight system.

***

#### Dependencies
We generate the sparse mask with `scipy.sparse`. The pytorch version is `1.8.1+cu111`. The memory profiling is based on [`pytorch_memlab`](https://github.com/Stonesjtu/pytorch_memlab), and we annotate our program with `nvtx`. 

To build the custom kernels, please use the `src/install.sh`. As our kernels target on the V100 GPU's tensor core architecture, currently only `sm70` is supported.