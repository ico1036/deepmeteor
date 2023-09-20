from dataclasses import dataclass
import torch
import torch.nn as nn
from torch import Tensor
from deepmeteor.models.base import ModelBase
from deepmeteor.models.base import ModelConfigBase
from deepmeteor.modules.attention import CrossAttentionModule, SelfAttentionModule
from deepmeteor.modules.reshape import Permute


class Encoder(nn.Module):
    latent: nn.Parameter

    def __init__(self,
                 latent_len: int,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 activation: str,
                 widening_factor: int,
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

        self.cross_attn = CrossAttentionModule(
            embed_dim=embed_dim,
            num_heads=1,
            activation=activation,
            widening_factor=widening_factor,
            dropout_p=dropout_p,
        )

        self.self_attn = SelfAttentionModule(
            embed_dim=embed_dim,
            num_heads=num_heads,
            activation=activation,
            widening_factor=widening_factor,
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

    def forward(self, input: Tensor, data_mask: Tensor) -> Tensor:
        latent = self.latent.unsqueeze(0).repeat(input.size(0), 1, 1)
        z = self.cross_attn(
            target=latent,
            source=input,
            source_data_mask=data_mask)
        for _ in range(self.num_layers):
            z = self.self_attn(input=z, data_mask=None)
        # FIXME
        return z


@dataclass
class PerceiverConfig(ModelConfigBase):
    name: str = 'Perceiver' # TODO note
    num_heads: int = 2
    num_layers: int = 1
    activation: str = 'ReLU'
    widening_factor: int = 1
    latent_len: int = 8
    dropout_p: float = 0

    def __post_init__(self):
        super().__post_init__()
        assert self.embed_dim % self.num_heads == 0

    def build(self):
        return Perceiver(self)


class Perceiver(ModelBase):
    config: PerceiverConfig


    def __init__(self, config: PerceiverConfig):
        super().__init__(config)
        self.input_embedding = self.build_input_embedding(config)
        self.encoder = self.build_encoder(config)
        self.decoder = self.build_decoder(config)
        self.regression_head = self.build_regression_head(config)

    @classmethod
    def build_encoder(cls, config: PerceiverConfig) -> nn.Module:
        return Encoder(
            latent_len=config.latent_len,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout_p=config.dropout_p,
            num_layers=config.num_layers,
            activation=config.activation,
            widening_factor=config.widening_factor,
        )

    @classmethod
    def build_decoder(cls, config: PerceiverConfig) -> nn.Module:
        if config.weight_dim == 0:
            decoder = nn.Sequential(
                Permute((0, 2, 1)),
                nn.AdaptiveAvgPool1d(output_size=1),
                nn.Flatten(),
            )
        else:
            decoder = CrossAttentionModule(
                embed_dim=config.embed_dim,
                num_heads=1,
                activation=config.activation,
                widening_factor=config.widening_factor,
                dropout_p=config.dropout_p,
            )
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
