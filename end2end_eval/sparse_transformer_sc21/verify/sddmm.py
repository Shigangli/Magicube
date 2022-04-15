import argparse
from os import pardir
import torch
from torch._C import dtype
from sptrans.sddmm import sddmm
import numpy as np
from static_mask import static_random_mask


# Args
parser = argparse.ArgumentParser(description='SDDMM kernel')

parser.add_argument('--seq_len', type=int, default=4096, help='input sequence length')
parser.add_argument('--feature', type=int, default=128, help='feature length')
parser.add_argument('--sparsity', type=float, default=0.9, help='mask sparsity')
parser.add_argument('--vec_length', type=int, default=1, help='vector length')
parser.add_argument('--func', action='store_true', help='do functional verification')

args = parser.parse_args()

m = int(args.seq_len / args.vec_length)
n = args.seq_len

column_indices, row_offsets, row_indices = static_random_mask(m, n, args.sparsity)

# Step 4: generate the input matrices
lhs_matrix = torch.randn(size=(args.seq_len, args.feature), dtype=torch.float16, device='cuda')
rhs_matrix = torch.randn(size=(args.seq_len, args.feature), dtype=torch.float16, device='cuda')


# Step 5: run the sddmm function
output_val = sddmm(row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, args.vec_length)


if args.func:
    # Do functional verification
    dense_ref = torch.matmul(lhs_matrix, torch.transpose(rhs_matrix, 0, 1))
    column_indices_cpu = column_indices.cpu().detach().numpy()
    row_offsets_cpu = row_offsets.cpu().detach().numpy()
    output_val_cpu = output_val.cpu().detach().numpy()
    dense_ref_cpu = dense_ref.cpu().detach().numpy()

    for m_vec in range(m):
        row_nnz = row_offsets_cpu[m_vec + 1] - row_offsets_cpu[m_vec]
        for j in range(row_nnz):
            for v in range(args.vec_length):
                val = output_val_cpu[(row_offsets_cpu[m_vec] + j) * args.vec_length + v]
                val_ref = dense_ref_cpu[m_vec * args.vec_length + v][column_indices_cpu[row_offsets_cpu[m_vec] + j]]
                if (np.abs(val - val_ref) > 0.05): 
                    print("value: %.4f, value_ref: %.4f" % (val, val_ref))
