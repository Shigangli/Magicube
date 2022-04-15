import argparse
import imp
from os import pardir
import torch
from torch._C import dtype
from sptrans.spmm import bspmm
import numpy as np
from static_mask import static_random_mask, batched_csr2dense
import nvtx


parser = argparse.ArgumentParser(description='SPMM kernel')

parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--seq_len', type=int, default=4096, help='input sequence length')
parser.add_argument('--feature', type=int, default=128, help='feature length')
parser.add_argument('--sparsity', type=float, default=0.9, help='mask sparsity')
parser.add_argument('--vec_length', type=int, default=8, help='vector length')
parser.add_argument('--func', action='store_true', help='do functional verification')

args = parser.parse_args()

m = int(args.seq_len / args.vec_length)
n = args.seq_len

column_indices, row_offsets, row_indices = static_random_mask(m, n, args.sparsity)

nnz = column_indices.numel() * args.vec_length

# Step 4: generate the input matrices
rhs_matrix = torch.randn(size=(args.batch_size, args.seq_len, args.feature), dtype=torch.float16, device='cuda')
values = torch.randn(size=(args.batch_size, nnz), dtype=torch.float16, device='cuda')

# Step 5: run the batched spmm function


if args.func:
    output_matrix = bspmm(row_indices, row_offsets, column_indices, values, rhs_matrix, args.vec_length)
    value_dense = batched_csr2dense(column_indices, row_offsets, values, m, n, args.vec_length, args.batch_size)
    dense_ref = torch.bmm(value_dense, rhs_matrix)

    print(output_matrix)
    print(dense_ref)
    print(torch.max(torch.abs(output_matrix - dense_ref)))


# profiling
value_dense_fake = torch.randn(size=(args.batch_size, args.seq_len, args.seq_len), dtype=torch.float16, device='cuda')

for i in range(10):
    with nvtx.annotate("torch.bmm"):
        dense_ref = torch.bmm(value_dense_fake, rhs_matrix)

for i in range(10):
    with nvtx.annotate("bspmm"):
        output_matrix = bspmm(row_indices, row_offsets, column_indices, values, rhs_matrix, args.vec_length)