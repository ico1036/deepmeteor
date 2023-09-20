from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from deepmeteor.modules.mlp import MLP

def scaled_dot_product_attention(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 attn_mask: Tensor | None,
                                 num_heads: int,
                                 dropout_p: float = 0,
):
    if num_heads > 1:
        N, T, E = query.size()
        S = key.size(1)

        H = num_heads
        D = E // H

        query = query.view(N, T, H, D).transpose(1, 2) # (N, H, T, D)
        key = key.view(N, S, H, D).transpose(1, 2) # (N, H, S, D)
        value = value.view(N, S, H, D).transpose(1, 2) # (N, H, S, D)
    output = F.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False
    )
    if num_heads > 1:
        N, H, T, E = output.size()
        # (N, H, T, E) -> (N, T, H, E) -> (N, T, H * E)
        output = output.transpose(1, 2).contiguous().view(N, T, -1)
    return output


def make_attn_mask(target_data_mask: Tensor,
                   source_data_mask: Tensor
) -> Tensor:
    # (N, T, 1) * (N, 1, S) -> (N, T, S)
    return target_data_mask.unsqueeze(2) * source_data_mask.unsqueeze(1)


class ScaledDotProductAttention(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout_p: float
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.project_q = nn.Linear(embed_dim, embed_dim)
        self.project_kv = nn.Linear(embed_dim, 2 * embed_dim)

    def forward(self,
                target: Tensor,
                source: Tensor,
                source_data_mask: Optional[Tensor],
    ) -> Tensor:
        query = self.project_q(target)
        key, value = self.project_kv(source).split(self.embed_dim, dim=2)
        if source_data_mask is None:
            attn_mask = None
        else:
            attn_mask = source_data_mask.unsqueeze(1).repeat(1, target.size(1), 1)

        output = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            num_heads=self.num_heads,
            dropout_p=self.dropout_p,
        )
        return output


class CrossAttentionModule(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 # mlp
                 activation: str,
                 widening_factor: int,
                 # common
                 dropout_p: float
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.layer_norm_target = nn.LayerNorm(embed_dim)
        self.layer_norm_source = nn.LayerNorm(embed_dim)
        self.attn = ScaledDotProductAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_p=dropout_p
        )
        self.pre_norm_mlp = MLP(
            input_dim=embed_dim,
            activation=activation,
            widening_factor=widening_factor,
            pre_norm=True
        )

    def forward(self,
                target: Tensor,
                source: Tensor,
                source_data_mask: Optional[Tensor],
    ) -> Tensor:
        residual = self.attn(
            target=self.layer_norm_target(target),
            source=self.layer_norm_source(source),
            source_data_mask=source_data_mask
        )
        output = target + residual

        residual = self.pre_norm_mlp(output)
        output = output + residual
        return output


class SelfAttentionModule(CrossAttentionModule):

    def forward(self, input: Tensor, data_mask: Optional[Tensor]) -> Tensor:
        return super().forward(input, input, data_mask)
