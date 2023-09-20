import torch
import torch.nn as nn
from torch import Tensor
from deepmeteor.data.dataset import MeteorDataset


class PUPPIEmbedding(nn.Module):

    def __init__(self,
                 cont_embed_dim: int,
                 pdgid_embed_dim: int,
                 charge_embed_dim: int,
    ) -> None:
        super().__init__()
        self.cont_embedding = nn.Linear(
            in_features=MeteorDataset.cont_num_features,
            out_features=cont_embed_dim,
        )

        self.pdgid_embedding = nn.Embedding(
            num_embeddings=MeteorDataset.pdgid_num_embeddings,
            embedding_dim=pdgid_embed_dim,
            padding_idx=0,
        )

        self.charge_embedding = nn.Embedding(
            num_embeddings=MeteorDataset.charge_num_embeddings,
            embedding_dim=charge_embed_dim,
            padding_idx=0
        )

    def forward(self,
                cont: Tensor,
                pdgid: Tensor,
                charge: Tensor,
    ) -> Tensor:
        z = [
            self.cont_embedding(cont),
            self.pdgid_embedding(pdgid),
            self.charge_embedding(charge),
        ]
        # TODO cat, sum,
        z = torch.cat(z, dim=2)
        return z
