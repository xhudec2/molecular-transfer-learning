# https://github.com/davidbuterez/multi-fidelity-gnns-for-drug-discovery-and-quantum-mechanics/blob/main/multifidelity_gnn/src/graph_models.py
from torch_geometric.nn.models import VGAE
import torch.nn as nn
import torch
from torch_geometric.utils import to_dense_batch
from pretraining.modelling.gcn import VariationalGCNEncoder
from pretraining.modelling.set_transformer import SetTransformer


class VGAEBackbone(nn.Module):
    def __init__(
        self,
        num_features: int,
        gnn_intermediate_dim: int = 256,
        node_latent_dim: int = 50,
        graph_latent_dim: int = 256,
        max_num_atoms_in_mol: int = 124,
        num_layers: int = 3,
        set_transformer_hidden_dim: int = 1024,
        set_transformer_num_heads: int = 16,
        set_transformer_num_sabs: int = 2,
        set_transformer_dropout: float = 0.0,
        use_batch_norm: bool = True,
        linear_output_size: int = 1,
    ):
        super().__init__()
        self.num_features = num_features

        self.node_latent_dim = node_latent_dim
        self.gnn_intermediate_dim = gnn_intermediate_dim
        self.use_batch_norm = use_batch_norm
        self.num_layers = num_layers
        self.max_num_atoms_in_mol = max_num_atoms_in_mol
        self.set_transformer_hidden_dim = set_transformer_hidden_dim
        self.set_transformer_num_heads = set_transformer_num_heads
        self.set_transformer_num_sabs = set_transformer_num_sabs
        self.set_transformer_dropout = set_transformer_dropout

        self.graph_latent_dim = graph_latent_dim
        self.linear_output_size = linear_output_size

        gnn_args = dict(
            in_channels=num_features,
            out_channels=node_latent_dim,
            num_layers=self.num_layers,
            intermediate_dim=self.gnn_intermediate_dim,
            use_batch_norm=self.use_batch_norm,
        )

        # VGAE
        self.gnn_model = VGAE(VariationalGCNEncoder(**gnn_args))

        # aggr
        self.st = SetTransformer(
            dim_input=node_latent_dim,
            num_outputs=32,
            dim_output=self.graph_latent_dim,
            num_inds=None,
            ln=True,
            dim_hidden=self.set_transformer_hidden_dim,
            num_heads=self.set_transformer_num_heads,
            num_sabs=self.set_transformer_num_sabs,
            dropout=self.set_transformer_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ):
        # 1. Obtain node embeddings
        z = self.gnn_model.encode(x, edge_index)

        # 2. Readout layer
        # Due to batching in PyTorch Geometric, the node embeddings must be regrouped into their original graphs
        # Details: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
        graph_embeddings, _ = to_dense_batch(
            z, batch, fill_value=0, max_num_nodes=self.max_num_atoms_in_mol
        )
        graph_embeddings = self.st(graph_embeddings)
        graph_embeddings = graph_embeddings.mean(dim=1)
        return z, graph_embeddings
