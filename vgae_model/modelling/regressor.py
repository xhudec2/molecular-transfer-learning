# https://github.com/davidbuterez/multi-fidelity-gnns-for-drug-discovery-and-quantum-mechanics/blob/main/multifidelity_gnn/src/graph_models.py
import torch.nn as nn
import torch
from torch import Tensor


class VGAERegressionHead(nn.Module):
    def __init__(
        self,
        graph_latent_dim: int = 256,
        use_batch_norm: bool = True,
        linear_output_size: int = 1,
    ):
        super().__init__()
        self.use_batch_norm = use_batch_norm

        self.linear_output1 = nn.Linear(graph_latent_dim, graph_latent_dim)

        if self.use_batch_norm:
            self.bn3 = nn.BatchNorm1d(graph_latent_dim)

        self.linear_output2 = nn.Linear(graph_latent_dim, linear_output_size)

    def forward(
        self,
        graph_embeddings: Tensor,
    ) -> Tensor:
        if self.use_batch_norm:
            predictions = self.bn3(self.linear_output1(graph_embeddings)).relu()
        predictions = torch.flatten(self.linear_output2(predictions))
        return predictions
