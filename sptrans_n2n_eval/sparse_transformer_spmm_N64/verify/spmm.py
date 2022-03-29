import argparse
from os import pardir
import torch
from torch._C import dtype
from sptrans.spmm import spmm
import numpy as np
from static_mask import static_random_mask, csr2dense
from scipy.sparse import csr_matrix


parser = argparse.ArgumentParser(description='SPMM kernel')

parser.add_argument('--seq_len', type=int, default=4096, help='input sequence length')
parser.add_argument('--feature', type=int, default=128, help='feature length')
parser.add_argument('--sparsity', type=float, default=0.9, help='mask sparsity')
parser.add_argument('--vec_length', type=int, default=4, help='vector length')
parser.add_argument('--func', action='store_true', help='do functional verification')

args = parser.parse_args()

m = int(args.seq_len / args.vec_length)
n = args.seq_len

column_indices, row_offsets, row_indices = static_random_mask(m, n, args.sparsity)

nnz = column_indices.numel() * args.vec_length

# Step 4: generate the input matrix
rhs_matrix = torch.randn(size=(args.seq_len, args.feature), dtype=torch.float16, device='cuda')
values = torch.randn(size=(nnz,), dtype=torch.float16, device='cuda')

# Step 5: run the spmm function
output_matrix = spmm(row_indices, row_offsets, column_indices, values, rhs_matrix, args.vec_length)


if args.func:
    value_dense = csr2dense(column_indices, row_offsets, values, m, n, args.vec_length)
    dense_ref = torch.matmul(value_dense, rhs_matrix)
    print(output_matrix)
    print(dense_ref)
    print(torch.max(torch.abs(output_matrix - dense_ref)))