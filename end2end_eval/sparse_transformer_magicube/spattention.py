import torch
from typing import Tuple, Optional
import warnings
import nvtx

from sptrans.quantization import bquantization
#from sptrans.sddmm import bsddmm
#from sptrans.spmm import bspmm
#from sptrans.softmax import bcsr_softmax

from sptrans.deq_sddmm import bsddmm_8b
from sptrans.deq_sddmm import bsddmm_4b
from sptrans.deq_spmm import bspmm_8b
from sptrans.deq_spmm import bspmm_16b8b
from sptrans.deq_spmm import bspmm_4b
from sptrans.deq_spmm import bspmm_8b4b
from sptrans.q_softmax import q_bcsr_softmax


def sp_multi_head_attention_forward(
    query: torch.Tensor,                                # query (seq_len, bs, embed_dim)
    key: torch.Tensor,                                  # key   (seq_len, bs, embed_dim)
    value: torch.Tensor,                                # value (seq_len, bs, embed_dim)
    embed_dim_to_check: int,                            # This equals to embed_dim
    num_heads: int,                                     # number of attention heads
    in_proj_weight: torch.Tensor,                       # Project the input q, k, v, (3 * embed_dim, embed_dim)
    in_proj_bias: torch.Tensor,                         # Bias for the above projection. (3 * embed_dim)
    bias_k: Optional[torch.Tensor],                     # bias for key, (1, 1, embed_dim)
    bias_v: Optional[torch.Tensor],                     # bias for value, (1, 1, embed_dim)
    add_zero_attn: bool,                                # TODO
    dropout_p: float,                                   # probability of an element to be zeros
    out_proj_weight: torch.Tensor,                      # output projection weight (embed_dim, embed_dim)
    out_proj_bias: torch.Tensor,                        # output projection bias (embed_dim)
    training: bool = True,                              # training or inference
    key_padding_mask: Optional[torch.Tensor] = None,    # TODO:
    need_weights: bool = True,                          # If True, returns the attention weight
    row_indices: Optional[torch.Tensor] = None,         # indices to the csr mask rows
    row_offsets: Optional[torch.Tensor] = None,         # the row indices of the csr mask
    column_indices: Optional[torch.Tensor] = None,      # the column indices of the csr mask
    vec_length: Optional[int] = 2,                      # the vector length of column vector sparsity
    lhs_pre: Optional[int] = 8,                         # the precision of lhs matrix
    rhs_pre: Optional[int] = 8,                         # the precision of rhs matrix
    use_separate_proj_weight: bool = False,             # if True, the q/k/v_proj_weight will be used rather than in_proj_weight
    q_proj_weight: Optional[torch.Tensor] = None,
    k_proj_weight: Optional[torch.Tensor] = None,
    v_proj_weight: Optional[torch.Tensor] = None,
    static_k: Optional[torch.Tensor] = None,            # TODO
    static_v: Optional[torch.Tensor] = None,            # TODO
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    scale_qkv = 36.0
    #scale_sfmx = 255.0
    scale_sfmx = 32.0

    # Get problem size
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    # Get feature dimension of each head
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

    # This computes the 1/sqrt{d} in the attention score before softmax
    scaling = float(head_dim) ** -0.5

    with nvtx.annotate("sp Input Linear Projection"):
    # If using the in_proj_weight for projection
        if not use_separate_proj_weight:
            # For self-attention whose query, key, and value are the same
            if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
                q, k, v = torch.nn.functional.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

            elif key is value or torch.equal(key, value):
                # encoder-decoder attention
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = 0
                _end = embed_dim
                _w = in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                q = torch.nn.functional.linear(query, _w, _b)

                if key is None:
                    assert value is None
                    k = None
                    v = None
                else:

                    # This is inline in_proj function with in_proj_weight and in_proj_bias
                    _b = in_proj_bias
                    _start = embed_dim
                    _end = None
                    _w = in_proj_weight[_start:, :]
                    if _b is not None:
                        _b = _b[_start:]
                    k, v = torch.nn.functional.linear(key, _w, _b).chunk(2, dim=-1)

            else:
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = 0
                _end = embed_dim
                _w = in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                q = torch.nn.funtional.linear(query, _w, _b)

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = embed_dim * 2
                _w = in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                k = torch.nn.functional.linear(key, _w, _b)

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim * 2
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                v = torch.nn.functional.linear(value, _w, _b)
        else:
            q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
            len1, len2 = q_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == query.size(-1)

            k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
            len1, len2 = k_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == key.size(-1)

            v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
            len1, len2 = v_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == value.size(-1)

            if in_proj_bias is not None:
                q = torch.nn.functional.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
                k = torch.nn.functional.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)])
                v = torch.nn.functional.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
            else:
                q = torch.nn.functional.linear(query, q_proj_weight_non_opt, in_proj_bias)
                k = torch.nn.functional.linear(key, k_proj_weight_non_opt, in_proj_bias)
                v = torch.nn.functional.linear(value, v_proj_weight_non_opt, in_proj_bias)

    batch_size = bsz * num_heads
    with nvtx.annotate("sp Input transpose"):
        # Transpose the batch*head and sequence length dimensions
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if static_k is not None:
            assert static_k.size(0) == bsz * num_heads
            assert static_k.size(2) == head_dim
            k = static_k

        if static_v is not None:
            assert static_v.size(0) == bsz * num_heads
            assert static_v.size(2) == head_dim
            v = static_v

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if add_zero_attn:
            src_len += 1
            k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.nn.functional.pad(key_padding_mask, (0, 1))

    with nvtx.annotate("QKV quantization"):
        #q_abs_max = torch.max(torch.abs(q))
        #k_abs_max = torch.max(torch.abs(k))
        #v_abs_max = torch.max(torch.abs(v))
        #q = bquantization(q, rhs_pre, scale_qkv/q_abs_max*q_abs_max)
        #k = bquantization(k, rhs_pre, scale_qkv/q_abs_max*q_abs_max)
        #v = bquantization(v, rhs_pre, scale_qkv/q_abs_max*q_abs_max)
        q = bquantization(q, rhs_pre, scale_qkv)
        k = bquantization(k, rhs_pre, scale_qkv)
        v = bquantization(v, rhs_pre, scale_qkv)
    
    # batched matrix multiplication
    with nvtx.annotate("sp QK^T"):
        if rhs_pre == 8:
            attn_output_weights = bsddmm_8b(row_indices, row_offsets, column_indices, q, k, vec_length, rhs_pre, scale_qkv*scale_qkv)
        if rhs_pre == 4:
            attn_output_weights = bsddmm_4b(row_indices, row_offsets, column_indices, q, k, vec_length, rhs_pre, scale_qkv*scale_qkv)
    
    with nvtx.annotate("sp Softmax"):
        attn_output_weights = q_bcsr_softmax(row_indices, row_offsets, attn_output_weights, scaling, scale_sfmx, vec_length, batch_size, lhs_pre)
    
    # Apply dropout
    #with nvtx.annotate("sp dropout"):
    #    attn_output_weights = torch.nn.functional.dropout(attn_output_weights, p=dropout_p, training=training)
    
    # batch multiplication with the value
    with nvtx.annotate("sp AV"):
        if lhs_pre == 8 and rhs_pre == 8:
            attn_output = bspmm_8b(row_indices, row_offsets, column_indices, attn_output_weights, v, vec_length, lhs_pre, rhs_pre, scale_qkv*scale_sfmx)
        if lhs_pre == 16 and rhs_pre == 8:
            attn_output = bspmm_16b8b(row_indices, row_offsets, column_indices, attn_output_weights, v, vec_length, lhs_pre, rhs_pre, scale_qkv*scale_sfmx)
        if lhs_pre == 8 and rhs_pre == 4:
            attn_output = bspmm_8b4b(row_indices, row_offsets, column_indices, attn_output_weights, v, vec_length, lhs_pre, rhs_pre, scale_qkv*scale_sfmx)
        if lhs_pre == 4 and rhs_pre == 4:
            attn_output = bspmm_4b(row_indices, row_offsets, column_indices, attn_output_weights, v, vec_length, lhs_pre, rhs_pre, scale_qkv*scale_sfmx)
    
    # transpose the output and concatenate the heads
    with nvtx.annotate("sp Output transpose"):
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

    with nvtx.annotate("sp Output Projection"):
        attn_output = torch.nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None
    

class spMultiheadAttention(torch.nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(spMultiheadAttention, self).__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True, row_indices: Optional[torch.Tensor] = None, row_offsets: Optional[torch.Tensor] = None,
                column_indices: Optional[torch.Tensor] = None, vec_length: int = 2, lhs_pre: int = 8, rhs_pre: int = 8) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return sp_multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            row_indices=row_indices, row_offsets=row_offsets, column_indices=column_indices,
            vec_length=vec_length,
            lhs_pre=lhs_pre,
            rhs_pre=rhs_pre)
