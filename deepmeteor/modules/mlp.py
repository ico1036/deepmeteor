import torch.nn as nn

class MLP(nn.Sequential):

    def __init__(self,
                 input_dim: int,
                 widening_factor: int = 4,
                 output_dim: int | None = None,
                 dropout_prob: float = 0.0,
                 activation: str = 'GELU',
                 pre_norm: bool = True,
    ) -> None:
        hidden_dim = widening_factor * input_dim
        output_dim = output_dim or input_dim

        layers = []

        if pre_norm:
            layers += [
                nn.LayerNorm(input_dim)
            ]

        layers += [
            nn.Linear(input_dim, hidden_dim),
            getattr(nn, activation)(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout_prob)
        ]

        super().__init__(*layers)
