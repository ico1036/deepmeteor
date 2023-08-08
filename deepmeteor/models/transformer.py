from dataclasses import dataclass
import torch
import torch.nn as nn
from torch import Tensor
from deepmeteor.models.base import ModelBase
from deepmeteor.models.base import ModelConfigBase
from deepmeteor.data.dataset import MeteorDataset
from deepmeteor.data.dataset import Batch
from deepmeteor.modules import ScatterMean


class SelfAttention(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 dropout_prob: float
    ) -> None:
        super().__init__()
        self.qkv_projection = nn.Linear(embed_dim, 3 * embed_dim)
        self.attention_dropout = nn.Dropout(dropout_prob)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.output_dropout = nn.Dropout(dropout_prob)

    def forward(self, input: Tensor, attn_mask: Tensor) -> Tensor:
        # dimensions
        ## N: batch size, L: sequence length, E: embedding dimension
        N, L, E = input.size()
        ## the number of heads
        H = attn_mask.size(1)
        ## depth
        D = E // H

        # query, key, value
        q, k, v = self.qkv_projection(input).split(E, dim=2)
        ## (N, L, E=H*D) -> (N, L, H D) -> (N, H, L, D)
        q, k, v = [each.view(N, L, H, D).transpose(1, 2) for each in [q, k, v]]

        # attention weight matrix
        ## (N, H, L, D) @ (N, H, D, L) -> (N, H, L, L)
        a = q @ k.transpose(2, 3)
        ## scaling
        a = D**-0.5 * a
        a = a.masked_fill(attn_mask, float('-inf'))
        a = a.softmax(dim=-1)
        a = self.attention_dropout(a)

        # output
        o = a @ v
        o = o.transpose(1, 2).contiguous().view(N, L, E)
        o = self.output_projection(o)
        o = self.output_dropout(o)
        return o

class Feedforward(nn.Sequential):

    def __init__(self,
                 input_dim: int,
                 widening_factor: int = 4,
                 dropout_prob: float = 0.0
    ) -> None:
        hidden_dim = widening_factor * input_dim
        super().__init__(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout_prob)
        )



class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 dropout_prob: float
    ) -> None:
        super().__init__()

        self.norm_1 = nn.LayerNorm(embed_dim)
        self.self_attention = SelfAttention(embed_dim, dropout_prob)

        self.feedforward = Feedforward(input_dim=embed_dim,
                                       dropout_prob=dropout_prob)
        self.norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        x = x + self.self_attention(self.norm_1(x), attn_mask)
        x = x + self.feedforward(self.norm_2(x))
        return x


class TransformerEncoder(nn.Module):
    num_heads: int

    def __init__(self,
                 embed_dim: int,
                 dropout_prob: float,
                 num_layers: int,
                 num_heads: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim=embed_dim,
                                    dropout_prob=dropout_prob)
            for _ in range(num_layers)
        ])
        self.num_heads = num_heads

    def forward(self, input: Tensor, data_mask: Tensor):
        """
        Args:
            input:
            data_mask: the boolean tensor mask for input
        """
        attn_mask = self.make_self_attention_mask(
            pad_mask=data_mask.logical_not(),
            num_heads=self.num_heads)

        x = input
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x

    @staticmethod
    def make_self_attention_mask(pad_mask: Tensor,
                                 num_heads: int,
    ) -> Tensor:
        r"""
        """
        # N: batch size, L: sequence length, H: the number of heads
        length = pad_mask.size(1)

        # pad_mask=(N, L) -unsqueeze-> (N, 1, L) -expand-> (N, L, L)
        attn_mask = pad_mask.unsqueeze(1).expand(-1, length, -1)
        # -expand-> (N, 1, L, L) -repeat-> (N, H, L, L)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)
        return attn_mask


@dataclass
class TransformerConfig(ModelConfigBase):
    name: str = 'Transformer' # TODO note
    cont_embed_dim: int = 16
    pdgid_embed_dim: int = 8
    charge_embed_dim: int = 8
    num_heads: int = 2
    dropout_prob: int = 0
    num_layers: int = 2


    def build(self):
        return Transformer(self)

    def __post_init__(self):
        if self.embed_dim % self.num_heads != 0:
            raise RuntimeError

    @property
    def embed_dim(self) -> int:
        return (self.cont_embed_dim + self.pdgid_embed_dim
                + self.charge_embed_dim)


class Transformer(ModelBase):

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.cont_embedding = nn.Linear(
            in_features=MeteorDataset.cont_num_features,
            out_features=config.cont_embed_dim,
        )

        self.pdgid_embedding = nn.Embedding(
            num_embeddings=MeteorDataset.pdgid_num_embeddings,
            embedding_dim=config.pdgid_embed_dim,
            padding_idx=0,
        )

        self.charge_embedding = nn.Embedding(
            num_embeddings=MeteorDataset.charge_num_embeddings,
            embedding_dim=config.charge_embed_dim,
            padding_idx=0
        )

        self.event_encoder = TransformerEncoder(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout_prob=config.dropout_prob,
            num_layers=config.num_layers,
        )

        self.aggregator = ScatterMean()

        self.regression_head = nn.Sequential(
            nn.BatchNorm1d(config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, MeteorDataset.target_num_features)
        )

    def forward(self,
                puppi_cands_cont: Tensor,
                puppi_cands_pdgid: Tensor,
                puppi_cands_charge: Tensor,
                puppi_cands_data_mask: Tensor,
                puppi_cands_length: Tensor
    ) -> Tensor:
        z = [
            self.cont_embedding(puppi_cands_cont),
            self.pdgid_embedding(puppi_cands_pdgid),
            self.charge_embedding(puppi_cands_charge),
        ]
        z = torch.cat(z, dim=2)

        z = self.event_encoder(z, puppi_cands_data_mask)

        z = self.aggregator(z, puppi_cands_data_mask, puppi_cands_length)

        # reconstructed met
        output = self.regression_head(z)

        return output

    # TODO rename
    def run(self, batch: Batch):
        return self(
            puppi_cands_cont=batch.puppi_cands_cont,
            puppi_cands_pdgid=batch.puppi_cands_pdgid,
            puppi_cands_charge=batch.puppi_cands_charge,
            puppi_cands_data_mask=batch.puppi_cands_data_mask,
            puppi_cands_length=batch.puppi_cands_length
        )
