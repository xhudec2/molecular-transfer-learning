# https://github.com/davidbuterez/multi-fidelity-gnns-for-drug-discovery-and-quantum-mechanics/blob/main/multifidelity_gnn/src/graph_models.py
import torch_geometric  # type: ignore
from torch_geometric.nn import GCNConv, BatchNorm  # type: ignore
import torch.nn as nn
import lightning as pl
from torch import Tensor


class VariationalGCNEncoder(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        use_batch_norm: bool,
        out_channels: int,
        num_layers: int,
    ):
        super(VariationalGCNEncoder, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.num_layers = num_layers

        modules: list[nn.Module | tuple[nn.Module, str]] = []

        for i in range(self.num_layers):
            if i == 0:
                modules.append(
                    (
                        GCNConv(in_channels, intermediate_dim, cached=False),
                        "x, edge_index -> x",
                    )
                )
            else:
                modules.append(
                    (
                        GCNConv(intermediate_dim, intermediate_dim, cached=False),
                        "x, edge_index -> x",
                    )
                )

            if self.use_batch_norm:
                modules.append(BatchNorm(intermediate_dim))
            modules.append(nn.ReLU(inplace=True))

        self.convs = torch_geometric.nn.Sequential("x, edge_index", modules)

        self.conv_mu = GCNConv(intermediate_dim, out_channels, cached=False)
        self.conv_logstd = GCNConv(intermediate_dim, out_channels, cached=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        x = self.convs(x, edge_index)

        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
