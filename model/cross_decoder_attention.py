from __future__ import annotations

from typing import List

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase


def _delimiter_token_ids(tokenizer: PreTrainedTokenizerBase, delimiter: str) -> List[int]:
    base = tokenizer("a b", add_special_tokens=False).input_ids
    with_delim = tokenizer(f"a{delimiter} b", add_special_tokens=False).input_ids
    ids = sorted(set(with_delim) - set(base))
    if not ids:
        ids = list(dict.fromkeys(tokenizer(delimiter, add_special_tokens=False).input_ids))
    return ids


def colon_token_ids(tokenizer: PreTrainedTokenizerBase) -> List[int]:
    return _delimiter_token_ids(tokenizer, ":")


def comma_token_ids(tokenizer: PreTrainedTokenizerBase) -> List[int]:
    return _delimiter_token_ids(tokenizer, ",")


class CrossDecoderAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        *,
        gamma: float,
    ):
        super().__init__()
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.gamma = float(gamma)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.ctx_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.o_proj.weight)

    def forward(
        self,
        main_hidden: torch.Tensor,
        aux_hidden: torch.Tensor,
        main_align: torch.Tensor,
        aux_align: torch.Tensor,
        *,
        gamma_scale: float = 1.0,
    ) -> torch.Tensor:
        B, Lm, D = main_hidden.shape
        La = aux_hidden.shape[1]
        H, hd = self.n_heads, self.head_dim

        q = self.q_proj(main_hidden).view(B, Lm, H, hd).transpose(1, 2)
        k = self.k_proj(aux_hidden).view(B, La, H, hd).transpose(1, 2)
        v = self.v_proj(aux_hidden).view(B, La, H, hd).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (hd ** 0.5)

        align = (main_align.unsqueeze(-1) == aux_align.unsqueeze(1))
        align = align & (main_align.unsqueeze(-1) >= 0) & (aux_align.unsqueeze(1) >= 0)
        valid_row = align.any(dim=-1)

        mask = align.unsqueeze(1)
        scores = scores.masked_fill(~mask, float("-inf"))
        scores = torch.where(
            valid_row.unsqueeze(1).unsqueeze(-1),
            scores,
            torch.zeros_like(scores),
        )

        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(B, Lm, D)
        ctx = self.dropout(self.o_proj(self.ctx_norm(ctx)))
        ctx = ctx * valid_row.unsqueeze(-1).to(ctx.dtype)

        coeff = self.gamma * float(gamma_scale)
        return main_hidden + coeff * ctx
