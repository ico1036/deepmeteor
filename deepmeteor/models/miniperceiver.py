from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from deepmeteor.models.base import ModelBase
from deepmeteor.models.base import ModelConfigBase
from deepmeteor.data.dataset import Batch
from deepmeteor.modules.attention import scaled_dot_product_attention
from deepmeteor.modules.reshape import Permute


class MiniCrossAttentionModule(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.layer_norm_target = nn.LayerNorm(input_dim)
        self.layer_norm_source = nn.LayerNorm(input_dim)

    def forward(self,
                target: Tensor,
                source: Tensor,
                source_data_mask: Optional[Tensor],
    ) -> Tensor:
        """
        Args:
            latent: shape (N, T, E)
            input: shape (N, S, E)
            input_mask: shape (N, S)
        Returns:
            output: shape (N, T, E)
        """

        target = self.layer_norm_target(target)
        source = self.layer_norm_source(source)

        if source_data_mask is None:
            attn_mask = None
        else:
            attn_mask = source_data_mask.unsqueeze(1).repeat(1, target.size(1), 1)

        output = scaled_dot_product_attention(
            query=target,
            key=source,
            value=source,
            attn_mask=attn_mask,
            num_heads=1
        )

        return output


class MiniSelfAttentionModule(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 num_heads: int,
                 activation: str,
                 dropout_p: float,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(latent_dim)
        self.linear = nn.Linear(latent_dim, latent_dim)
        self.activation = getattr(nn, activation)()

        self.num_heads = num_heads
        self.dropout_p = dropout_p

    def forward(self, latent: Tensor):
        z = self.layer_norm(latent)
        z = scaled_dot_product_attention(
            query=z,
            key=z,
            value=z,
            attn_mask=None,
            num_heads=self.num_heads,
        )
        z = F.dropout(latent, p=self.dropout_p)

        z = self.linear(z)
        z = self.activation(z)
        z = F.dropout(latent, p=self.dropout_p)
        return z



class Encoder(nn.Module):
    latent: nn.Parameter

    def __init__(self,
                 latent_len: int,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 activation: str,
                 dropout_p: float,
    ) -> None:
        """
        Args:
            latent_len:
            embed_dim:
            num_heads:
            num_layers:
            dropout_p:
        """
        super().__init__()
        self.register_parameter(
            name='latent',
            param=self.init_latent(latent_len, embed_dim)
        )
        self.num_layers = num_layers

        self.cross_attn = MiniCrossAttentionModule(input_dim=embed_dim)

        self.self_attn = MiniSelfAttentionModule(
            latent_dim=embed_dim,
            num_heads=num_heads,
            activation=activation,
            dropout_p=dropout_p
        )

    @staticmethod
    def init_latent(length: int,
                    num_features: int,
                    std: float = 0.02
    ) -> nn.Parameter:
        latent = torch.empty(length, num_features)
        nn.init.trunc_normal_(latent, std=std)
        return nn.Parameter(latent)

    def forward(self, input: Tensor, data_mask: Tensor):
        latent = self.latent.unsqueeze(0).repeat(input.size(0), 1, 1)
        z = self.cross_attn(target=latent, source=input, source_data_mask=data_mask)
        for _ in range(self.num_layers):
            z = self.self_attn(z)
        # FIXME
        return z


@dataclass
class MiniPerceiverConfig(ModelConfigBase):
    name: str = 'MiniPerceiver' # TODO note
    num_heads: int = 2
    num_layers: int = 1
    activation: str = 'ReLU'
    latent_len: int = 8
    dropout_p: float = 0

    def __post_init__(self):
        super().__post_init__()
        assert self.embed_dim % self.num_heads == 0

    def build(self):
        return MiniPerceiver(self)


class MiniPerceiver(ModelBase):
    config: MiniPerceiverConfig


    def __init__(self, config: MiniPerceiverConfig):
        super().__init__(config)
        self.input_embedding = self.build_input_embedding(config)
        self.encoder = self.build_encoder(config)
        self.decoder = self.build_decoder(config)
        self.regression_head = self.build_regression_head(config)

    @classmethod
    def build_encoder(cls, config: MiniPerceiverConfig) -> nn.Module:
        return Encoder(
            latent_len=config.latent_len,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout_p=config.dropout_p,
            num_layers=config.num_layers,
            activation=config.activation,
        )

    @classmethod
    def build_decoder(cls, config: MiniPerceiverConfig) -> nn.Module:
        if config.weight_dim == 0:
            decoder = nn.Sequential(
                Permute((0, 2, 1)),
                nn.AdaptiveAvgPool1d(output_size=1),
                nn.Flatten(),
            )
        else:
            decoder = MiniCrossAttentionModule(input_dim=config.embed_dim)
        return decoder

    def forward(self,
                cont: Tensor,
                pdgid: Tensor,
                charge: Tensor,
                data_mask: Tensor,
    ) -> Tensor:
        input = self.input_embedding(cont=cont, pdgid=pdgid, charge=charge)
        latent = self.encoder(input=input, data_mask=data_mask)
        if self.config.weight_dim == 0:
            latent = self.decoder(latent)
        else:
            latent = self.decoder(target=input, source=latent,
                             source_data_mask=None)
        return self.regression_head(latent)
