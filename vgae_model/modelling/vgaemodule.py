# https://github.com/davidbuterez/multi-fidelity-gnns-for-drug-discovery-and-quantum-mechanics/blob/main/multifidelity_gnn/src/graph_models.py
import lightning as pl
import torch
import torch.nn.functional as F
from vgae_model.modelling import VGAERegressionHead, VGAEBackbone
from torchmetrics.regression import (
    MeanSquaredError,
    MeanAbsoluteError,
    R2Score,
)
from torchmetrics import MetricCollection
from copy import deepcopy


class VGAEModule(pl.LightningModule):
    def __init__(
        self,
        num_features: int,
        gnn_intermediate_dim: int = 256,
        node_latent_dim: int = 50,
        graph_latent_dim: int = 256,
        max_num_atoms_in_mol: int = 124,
        num_layers: int = 3,
        set_transformer_hidden_dim: int = 64,
        set_transformer_num_heads: int = 16,
        set_transformer_num_sabs: int = 2,
        set_transformer_dropout: float = 0.0,
        lr: float = 0.001,
        batch_size: int = 32,
        monitor_loss: str = "val/total_loss",
        use_batch_norm: bool = True,
        linear_output_size: int = 1,
        num_pretrain_epochs: int = 150,
    ):
        super().__init__()
        self.num_features = num_features
        self.lr = lr
        self.batch_size = batch_size
        self.monitor_loss = monitor_loss
        self.use_batch_norm = use_batch_norm
        self.num_pretrain_epochs = num_pretrain_epochs

        metrics = {
            "rmse": MeanSquaredError(squared=False),
            "mae": MeanAbsoluteError(),
            "r2": R2Score(),
        }
        self.train_metrics = MetricCollection(deepcopy(metrics), prefix="train/")
        self.val_metrics = MetricCollection(deepcopy(metrics), prefix="val/")
        self.test_metrics = MetricCollection(deepcopy(metrics), prefix="test/")
        self.num_called_test = 1

        self.backbone = VGAEBackbone(
            num_features=num_features,
            node_latent_dim=node_latent_dim,
            gnn_intermediate_dim=gnn_intermediate_dim,
            use_batch_norm=use_batch_norm,
            num_layers=num_layers,
            max_num_atoms_in_mol=max_num_atoms_in_mol,
            set_transformer_hidden_dim=set_transformer_hidden_dim,
            set_transformer_num_heads=set_transformer_num_heads,
            set_transformer_num_sabs=set_transformer_num_sabs,
            set_transformer_dropout=set_transformer_dropout,
            graph_latent_dim=graph_latent_dim,
            linear_output_size=linear_output_size,
        )
        self.regressor = VGAERegressionHead(
            graph_latent_dim, use_batch_norm, linear_output_size
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ):
        z, graph_embeddings = self.backbone(x, edge_index, batch)
        if self.current_epoch < self.num_pretrain_epochs:
            return z, graph_embeddings, None

        predictions = self.regressor(graph_embeddings)
        return z, graph_embeddings, predictions

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": opt,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=0.75, patience=15
            ),
            "monitor": self.monitor_loss,
        }

    def _batch_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor = None,
        batch_mapping: torch.Tensor | None = None,
    ):
        num_nodes = x.shape[0]

        z, graph_embeddings, predictions = self.forward(x, edge_index, batch_mapping)

        vgae_loss = self.backbone.gnn_model.recon_loss(z, edge_index)
        vgae_loss = vgae_loss + (1 / num_nodes) * self.backbone.gnn_model.kl_loss()

        if self.current_epoch < self.num_pretrain_epochs:
            task_loss = 0
        else:
            task_loss = F.mse_loss(torch.flatten(predictions), torch.flatten(y.float()))

        total_loss = vgae_loss + task_loss
        return total_loss, vgae_loss, task_loss, z, graph_embeddings, predictions

    def _step(self, batch: torch.Tensor):
        x, edge_index, y, batch_mapping = (
            batch.x,
            batch.edge_index,
            batch.y,
            batch.batch,
        )

        (
            total_loss,
            vgae_loss,
            task_loss,
            _,
            _,
            predictions,
        ) = self._batch_loss(x, edge_index, y, batch_mapping)

        return total_loss, vgae_loss, task_loss, predictions

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        train_total_loss, vgae_loss, task_loss, predictions = self._step(batch)

        if self.current_epoch >= self.num_pretrain_epochs:
            self.train_metrics.update(predictions, batch.y)
            self.log_dict(self.train_metrics, batch_size=self.batch_size, on_epoch=True)
            self.log("train/task_loss", task_loss, batch_size=self.batch_size)

        self.log("train/total_loss", train_total_loss, batch_size=self.batch_size)
        self.log("train/vgae_loss", vgae_loss, batch_size=self.batch_size)
        return train_total_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        val_total_loss, vgae_loss, task_loss, predictions = self._step(batch)

        if self.current_epoch >= self.num_pretrain_epochs:
            self.val_metrics.update(predictions, batch.y)
            self.log_dict(self.val_metrics, batch_size=self.batch_size)
            self.log("val/task_loss", task_loss, batch_size=self.batch_size)

        self.log("val/total_loss", val_total_loss, batch_size=self.batch_size)
        self.log("val/vgae_loss", vgae_loss, batch_size=self.batch_size)

        return val_total_loss

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        test_total_loss, vgae_loss, task_loss, predictions = self._step(batch)

        self.log("test/total_loss", test_total_loss, batch_size=self.batch_size)
        self.log("test/vgae_loss", vgae_loss, batch_size=self.batch_size)
        self.log("test/task_loss", task_loss, batch_size=self.batch_size)
        self.test_metrics.update(predictions, batch.y)
        self.log_dict(self.test_metrics, batch_size=self.batch_size)

        return test_total_loss
