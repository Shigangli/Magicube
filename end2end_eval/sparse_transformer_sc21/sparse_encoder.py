from numpy.lib.function_base import vectorize
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import nvtx
from attention import MultiheadAttention as MultiheadAttention_
from spattention import spMultiheadAttention
from verify.static_mask import static_random_mask
from torch.cuda.amp import autocast
from cudaprofile import start, stop

import argparse

parser = argparse.ArgumentParser(description='Sparse Multihead Attention')
# Hyper-parameter of the self-attention module
parser.add_argument('--embed_dim', type=int, default=2048, help='The embedding dimension. We have head_dim * num_heads = embed_dim')
parser.add_argument('--num_heads', type=int, default=8, help='The number of attention heads')
# Hyper-parameter of the input size
parser.add_argument('--bs', type=int, default=1, help='batch size')
parser.add_argument('--seq_len', type=int, default=2048, help='sequence length')
# For sparse mask
parser.add_argument('--sparsity', type=float, default=0.95, help='mask sparsity')
parser.add_argument('--vec_length', type=int, default=8, help='vector length')
args = parser.parse_args()


# Construct the multihead self-attention module
torch_attn = MultiheadAttention_(embed_dim=args.embed_dim, num_heads=args.num_heads, dropout=0).cuda()

sp_attn = spMultiheadAttention(embed_dim=args.embed_dim, num_heads=args.num_heads, dropout=0).cuda()


torch_attn.eval().half()
sp_attn.eval().half()

# Initialize the inputs
query = torch.randn(size=(args.seq_len, args.bs, args.embed_dim), dtype=torch.float16, device='cuda')

# Generate the mask
m = int(args.seq_len / args.vec_length)
n = args.seq_len
column_indices, row_offsets, row_indices = static_random_mask(m, n, args.sparsity)

# with autocast():
# Warmup
for i in range(10):
    out = torch_attn(query, query, query, need_weights=False) #, attn_mask=mask)

# profile:
start()
for i in range(5):
    with nvtx.annotate("MultiheadAttention"):
        out = torch_attn(query, query, query, need_weights=False)
stop()

for i in range(10):
    out = sp_attn(query, query, query, need_weights=False, row_indices=row_indices, row_offsets=row_offsets, column_indices=column_indices, vec_length=args.vec_length)

start()
for i in range(5):
    with nvtx.annotate("spMultiheadAttention"):
        out = sp_attn(query, query, query, need_weights=False, row_indices=row_indices, row_offsets=row_offsets, column_indices=column_indices, vec_length=args.vec_length)

stop()
