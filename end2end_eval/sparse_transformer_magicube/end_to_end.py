from numpy.lib.function_base import select, vectorize
from torch._C import dtype
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import nvtx
from torch.nn.modules import dropout
from attention import MultiheadAttention as MultiheadAttention_
from spattention import spMultiheadAttention
#from verify.static_mask import static_random_mask
from verify.static_mask import static_random_mask_aligned
from torch.cuda.amp import autocast
from cudaprofile import start, stop
from pytorch_memlab import LineProfiler, profile
import math

import argparse
import time

parser = argparse.ArgumentParser(description='Sparse Multihead Attention')
# Hyper-parameter of the self-attention module
parser.add_argument('--vocab_size', type=int, default=257, help='The number of vocabularies in the dataset')
parser.add_argument('--num_class', type=int, default=2, help='Number of classes in the model')
parser.add_argument('--mlp_dim', type=int, default=1024, help='The embedding dimension. We have head_dim * num_heads = embed_dim')
parser.add_argument('--embed_dim', type=int, default=256, help='the key and value dimension')
parser.add_argument('--num_heads', type=int, default=4, help='The number of attention heads')
# Hyper-parameter of the input size
parser.add_argument('--bs', type=int, default=2, help='batch size')
parser.add_argument('--lhs_pre', type=int, default=8, help='precision of lhs matrix')
parser.add_argument('--rhs_pre', type=int, default=8, help='precision of rhs matrix')
parser.add_argument('--seq_len', type=int, default=4096, help='sequence length')
# For sparse mask
parser.add_argument('--sparsity', type=float, default=0.9, help='mask sparsity')
parser.add_argument('--vec_length', type=int, default=8, help='vector length')
parser.add_argument('--model', choices=['sparse', 'dense', 'both'], default='sparse', help='which model to launch')
parser.add_argument('--mem', action='store_true', help="If set, the peak memory usage will be reported")
args = parser.parse_args()

def profile_(model):
    if args.mem and (model == args.model or args.model == 'both'):
        return profile
    else:
        return lambda a: a


class SparseTransformerBlock(nn.Module):
    def __init__(self, embed_dim, mlp_dim, num_heads, sparsity, seq_len):
        super(SparseTransformerBlock, self).__init__()

        m = int(seq_len / args.vec_length)
        n = seq_len
        mma_k_dim = 16
        if args.rhs_pre == 8:
            mma_k_dim = 16
        if args.rhs_pre == 4:
            mma_k_dim = 32

        self.self_attention = spMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0).cuda()
        #self.column_indices, self.row_offsets, self.row_indices = static_random_mask(m, n, sparsity)
        self.column_indices, self.column_indices_shuffle, self.row_offsets, self.row_indices, self.aligned_num_item = static_random_mask_aligned(m, n, sparsity, mma_k_dim);

        self.layer_norm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)

    def forward(self, x):
        with nvtx.annotate("Layer Norm 1"):
            out = self.layer_norm1(x)
        with nvtx.annotate("Self Attention"):
            out = self.self_attention(
                out, out, out, need_weights=False, row_indices=self.row_indices, 
                row_offsets=self.row_offsets, column_indices=self.column_indices, 
                vec_length=args.vec_length, lhs_pre=args.lhs_pre, rhs_pre=args.rhs_pre)
        with nvtx.annotate("Residual 1"):
            out = out[0] + x
        with nvtx.annotate("Layer Norm 2"):
            out = self.layer_norm2(out)
        with nvtx.annotate("Linear 1"):
            out = self.linear1(out)
        with nvtx.annotate("Linear 2"):
            out = self.linear2(out)

        return out + x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, mlp_dim, num_heads):
        super(TransformerBlock, self).__init__()

        self.self_attention = MultiheadAttention_(embed_dim=embed_dim, num_heads=num_heads, dropout=0).cuda()
        self.layer_norm1 = nn.LayerNorm(normalized_shape=embed_dim).cuda()
        self.layer_norm2 = nn.LayerNorm(normalized_shape=embed_dim).cuda()
        self.linear1 = nn.Linear(embed_dim, mlp_dim).cuda()
        self.linear2 = nn.Linear(mlp_dim, embed_dim).cuda()

    def forward(self, x):
        with nvtx.annotate("Layer Norm 1"):
            out = self.layer_norm1(x)
        with nvtx.annotate("Self Attention"):
            out = self.self_attention(out, out, out, need_weights=False)
        with nvtx.annotate("Residual 1"):
            out = out[0] + x
        with nvtx.annotate("Layer Norm 2"):
            out = self.layer_norm2(out)
        with nvtx.annotate("Linear 1"):
            out = self.linear1(out)
        with nvtx.annotate("Linear 2"):
            out = self.linear2(out)

        return out + x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SparseTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, sparsity, seq_len, mlp_dim, num_class):
        super(SparseTransformerEncoder, self).__init__()
        self.encoder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=0.)
        
        self.transformer_encoder0 = SparseTransformerBlock(embed_dim=embed_dim, num_heads=num_heads, sparsity=sparsity, seq_len=seq_len, mlp_dim=mlp_dim)
        self.transformer_encoder1 = SparseTransformerBlock(embed_dim=embed_dim, num_heads=num_heads, sparsity=sparsity, seq_len=seq_len, mlp_dim=mlp_dim)
        self.transformer_encoder2 = SparseTransformerBlock(embed_dim=embed_dim, num_heads=num_heads, sparsity=sparsity, seq_len=seq_len, mlp_dim=mlp_dim)
        self.transformer_encoder3 = SparseTransformerBlock(embed_dim=embed_dim, num_heads=num_heads, sparsity=sparsity, seq_len=seq_len, mlp_dim=mlp_dim)
        self.embed_dim = embed_dim
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.linear = nn.Linear(embed_dim, num_class)

    @profile_('sparse')
    def forward(self, x):
        out = self.encoder(x) * np.sqrt(self.embed_dim)
        out = torch.transpose(out, 0, 1)
        out = self.pos_encoder(out)
        out = self.transformer_encoder0(out)
        out = self.transformer_encoder1(out)
        out = self.transformer_encoder2(out)
        out = self.transformer_encoder3(out)
        out = self.layer_norm(out)
        out = self.linear(out)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, seq_len, mlp_dim, num_class):
        super(TransformerEncoder, self).__init__()
        self.encoder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=0.)
        self.transformer_encoder0 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)
        self.transformer_encoder1 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)
        self.transformer_encoder2 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)
        self.transformer_encoder3 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)
        self.embed_dim = embed_dim
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.linear = nn.Linear(embed_dim, num_class)

    @profile_('dense')
    def forward(self, x):
        out = self.encoder(x) * np.sqrt(self.embed_dim)
        out = torch.transpose(out, 0, 1)
        out = self.pos_encoder(out)
        out = self.transformer_encoder0(out)
        out = self.transformer_encoder1(out)
        out = self.transformer_encoder2(out)
        out = self.transformer_encoder3(out)
        out = self.layer_norm(out)
        out = self.linear(out)
        return out


def sparse_profile():
    spTrans = SparseTransformerEncoder(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        sparsity=args.sparsity,
        seq_len=args.seq_len,
        mlp_dim=args.mlp_dim,
        num_class=args.mlp_dim
    )
    spTrans.cuda().eval().half()

    x = torch.randint(low=0, high=args.vocab_size, size=(args.bs, args.seq_len), dtype=torch.int32, device='cuda')
    if args.mem:
        out = spTrans(x)
    else:
        # warm up
        for i in range(10):
            out = spTrans(x)

        start_time = time.time()
        # timer
        for i in range(32):
            out = spTrans(x)
        end_time = time.time()
        print("Magicube (lhs_pre[%d], rhs_pre[%d], batch_size[%d]) average runtime %.4f milliseconds" % (args.lhs_pre, args.rhs_pre, args.bs, (end_time-start_time)/32.0*1000.0))

        # profile
        for i in range(10):
            with nvtx.annotate("Sparse Encoder"):
                out = spTrans(x)
    
def dense_prof():
    denseTrans = TransformerEncoder(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        mlp_dim=args.mlp_dim,
        num_class=args.mlp_dim
    )
    denseTrans.cuda().eval().half()

    x = torch.randint(low=0, high=args.vocab_size, size=(args.bs, args.seq_len), dtype=torch.int32, device='cuda')
    if args.mem:
        out = denseTrans(x)
    else:
        # warm up
        for i in range(10):
            out = denseTrans(x)

        start_time = time.time()
        # timer
        for i in range(32):
            out = denseTrans(x)
        end_time = time.time()
        print("Dense average runtime: [%.6f] seconds" % ((end_time-start_time)/32.0))

        # profile
        for i in range(10):
            with nvtx.annotate("Dense Encoder"):
                out = denseTrans(x)
    

if args.model == "sparse":
    sparse_profile()
elif args.model == "dense":
    dense_prof()
else:
    sparse_profile()
    dense_prof()
