import torch
import lego_ops

lego_ops.load_ft_torch()
from flash_attn.flash_attn_interface import maybe_contiguous


def _flash_attn_varlen_forward(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    return_softmax=False,
    block_table=None,
    leftpad_k=None,
    seqused_k=None,
):
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    # out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flash_attn_cuda.varlen_fwd(
    #     q,
    #     k,
    #     v,
    #     None,
    #     cu_seqlens_q,
    #     cu_seqlens_k,
    #     seqused_k,
    #     leftpad_k,
    #     block_table,
    #     alibi_slopes,
    #     max_seqlen_q,
    #     max_seqlen_k,
    #     dropout_p,
    #     softmax_scale,
    #     False,
    #     causal,
    #     window_size[0],
    #     window_size[1],
    #     softcap,
    #     return_softmax,
    #     None,
    # )
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = torch.ops.Extern.Flash2Attn_forward_ext1(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, False, causal, False,
        -1, return_softmax, False, None)
    return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state


def _flash_attn_varlen_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    deterministic,
    rng_state=None,
):
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    # (
    #     dq,
    #     dk,
    #     dv,
    #     softmax_d,
    # ) = flash_attn_cuda.varlen_bwd(
    #     dout,
    #     q,
    #     k,
    #     v,
    #     out,
    #     softmax_lse,
    #     dq,
    #     dk,
    #     dv,
    #     cu_seqlens_q,
    #     cu_seqlens_k,
    #     alibi_slopes,
    #     max_seqlen_q,
    #     max_seqlen_k,
    #     dropout_p,
    #     softmax_scale,
    #     False,
    #     causal,
    #     window_size[0],
    #     window_size[1],
    #     softcap,
    #     deterministic,
    #     None,
    #     rng_state,
    # )
    dq, dk, dv, softmax_d = torch.ops.Extern.Flash2Attn_backward_ext1(dout, q, k, v, out, softmax_lse, dq, dk, dv,
                                                                      cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                                                                      max_seqlen_k, dropout_p, softmax_scale, False,
                                                                      causal, False, deterministic, -1, None, rng_state)
    return dq, dk, dv, softmax_d
