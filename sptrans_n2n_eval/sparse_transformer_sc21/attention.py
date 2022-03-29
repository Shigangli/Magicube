import torch
from typing import Tuple, Optional
import warnings
import nvtx


def multi_head_attention_forward(
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
    attn_mask: Optional[torch.Tensor] = None,           # a boolTensor with shape (seq_len, seq_len) or (bs * num_head, seq_len, seq_len)
    use_separate_proj_weight: bool = False,             # if True, the q/k/v_proj_weight will be used rather than in_proj_weight
    q_proj_weight: Optional[torch.Tensor] = None,
    k_proj_weight: Optional[torch.Tensor] = None,
    v_proj_weight: Optional[torch.Tensor] = None,
    static_k: Optional[torch.Tensor] = None,            # TODO
    static_v: Optional[torch.Tensor] = None,            # TODO
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    # print("my own implementation")
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

    with nvtx.annotate("Input Linear Projection"):
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

    with nvtx.annotate("Query Scaling"):
        # Scale q at the very begining
        q = q * scaling

    with nvtx.annotate("Preprocessing attention mask"):
        # If a sparse attention mask is applied
        if attn_mask is not None:
            # Check the attn_mask
            assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
            if attn_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)

            # This handles the 2D mask shared across the heads and samples in the batch
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            
            # This processes the 3D mask
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        if bias_k is not None and bias_v is not None:
            if static_k is None and static_v is None:
                k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
                v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
                if attn_mask is not None:
                    attn_mask = torch.nn.functional.pad(attn_mask, (0, 1))
                if key_padding_mask is not None:
                    key_padding_mask = torch.nn.functional.pad(key_padding_mask, (0, 1))
            else:
                assert static_k is None, "bias cannot be added to static key."
                assert static_v is None, "bias cannot be added to static value."
        else:
            assert bias_k is None
            assert bias_v is None

    with nvtx.annotate("Input transpose"):
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
            if attn_mask is not None:
                attn_mask = torch.nn.functional.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = torch.nn.functional.pad(key_padding_mask, (0, 1))

    # batched matrix multiplication
    with nvtx.annotate("QK^T"):
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    with nvtx.annotate("Apply attention mask"):
        if attn_mask is not None:
            # binary mask that fills the attention weight
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    # print(attn_output_weights.dtype)

    # Apply softmax
    with nvtx.annotate("Softmax"):
        attn_output_weights = torch.nn.functional.softmax(attn_output_weights, dim=-1)

    # Apply dropout
    with nvtx.annotate("dropout"):
        attn_output_weights = torch.nn.functional.dropout(attn_output_weights, p=dropout_p, training=training)

    # print(attn_output_weights.dtype)

    # batch multiplication with the value
    with nvtx.annotate("AV"):
        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]

    # transpose the output and concatenate the heads
    with nvtx.annotate("Output transpose"):
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

    with nvtx.annotate("Output Projection"):
        attn_output = torch.nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttention(torch.nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim)


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self._qkv_same_embed_dim:
            # In this case, separate projection weight must be used.
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
