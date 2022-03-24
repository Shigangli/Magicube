import argparse
from os import pardir
import torch
from sptrans.softmax import bcsr_softmax
import numpy as np
from static_mask import static_random_mask, batched_csr2dense
from scipy.sparse import csr_matrix
import nvtx

parser = argparse.ArgumentParser(description='SPMM kernel')

parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--seq_len', type=int, default=4096, help='input sequence length')
parser.add_argument('--feature', type=int, default=128, help='feature length')
parser.add_argument('--sparsity', type=float, default=0.9, help='mask sparsity')
parser.add_argument('--vec_length', type=int, default=2, help='vector length')
parser.add_argument('--func', action='store_true', help='do functional verification')

args = parser.parse_args()

m = int(args.seq_len / args.vec_length)
n = args.seq_len

column_indices, row_offsets, row_indices = static_random_mask(m, n, args.sparsity)

nnz = column_indices.numel() * args.vec_length

# Step 4: generate the input matrices
values = torch.randn(size=(args.batch_size, nnz), dtype=torch.float16, device='cuda')
scaler = 1./np.sqrt(args.seq_len)


if args.func:
    attn = bcsr_softmax(row_indices, row_offsets, values, scaler, args.vec_length, args.batch_size)
    value_dense = batched_csr2dense(column_indices, row_offsets, values, m, n, args.vec_length, args.batch_size, -np.inf)
    dense_ref = torch.nn.functional.softmax(value_dense * scaler, dim=-1)

    column_indices_cpu = column_indices.cpu().detach().numpy()
    row_offsets_cpu = row_offsets.cpu().detach().numpy()
    output_val_cpu = attn.cpu().detach().numpy()
    dense_ref_cpu = dense_ref.cpu().detach().numpy()


    for b in range(args.batch_size):
        print("batch %d" % b)
        for m_vec in range(m):
            row_nnz = row_offsets_cpu[m_vec + 1] - row_offsets_cpu[m_vec]
            for j in range(row_nnz):
                for v in range(args.vec_length):
                    val = output_val_cpu[b][(row_offsets_cpu[m_vec] + j) * args.vec_length + v]
                    val_ref = dense_ref_cpu[b][m_vec * args.vec_length + v][column_indices[row_offsets_cpu[m_vec] + j]]
                    if (np.abs(val - val_ref) > 0.01 * val_ref): 
                        print("value: %.4f, value_ref: %.4f" % (val, val_ref))


value_dense_fake = torch.randn(size=(args.batch_size, args.seq_len, args.seq_len), dtype=torch.float16, device='cuda')
for i in range(10):
    with nvtx.annotate("torch.softmax"):
        dense_ref = torch.nn.functional.softmax(value_dense_fake, dim=-1)

for i in range(10):
    with nvtx.annotate("bsoftmax"):
        attn = bcsr_softmax(row_indices, row_offsets, values, scaler, args.vec_length, args.batch_size)



