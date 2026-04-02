"""Hierarchical Multi-Scale GPS Model with MinCutPool.

Architecture:
  FeatureEncoder  (node + edge feature embedding)
  Fine-scale GPS layers  (local GatedGCN + global Mamba, for N_fine layers)
  MinCutPoolLayer          (differentiable soft-assignment: atoms → supernodes)
  Coarse-scale GPS layers (local GatedGCN + global Mamba, for N_coarse layers)
  GNNHead                  (global mean-pool → MLP → output)

The auxiliary MinCut + Orthogonality losses are accumulated during forward and
stored on ``self.pool_loss`` so that the training loop can add them (with the
configured weight) to the task loss.

Registration key: ``'HierarchicalGPSModel'``
"""
import torch
import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import new_layer_config, BatchNorm1dNode
from torch_geometric.graphgym.register import register_network

from graphgps.encoder.ER_edge_encoder import EREdgeEncoder
from graphgps.layer.gps_layer import GPSLayer
from graphgps.layer.mincut_pool_layer import MinCutPoolLayer


# ---------------------------------------------------------------------------
# Feature encoder (shared with GPSModel)
# ---------------------------------------------------------------------------

class FeatureEncoder(nn.Module):
    """Encode node and edge features into the hidden dimension.

    Identical to the FeatureEncoder in ``gps_model.py``; reproduced here to
    keep this module self-contained.
    """

    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            cfg.gnn.dim_edge = (
                16 if 'PNA' in cfg.gt.layer_type else cfg.gnn.dim_inner
            )
            if cfg.dataset.edge_encoder_name == 'ER':
                self.edge_encoder = EREdgeEncoder(cfg.gnn.dim_edge)
            elif cfg.dataset.edge_encoder_name.endswith('+ER'):
                EdgeEncoder = register.edge_encoder_dict[
                    cfg.dataset.edge_encoder_name[:-3]]
                self.edge_encoder = EdgeEncoder(
                    cfg.gnn.dim_edge - cfg.posenc_ERE.dim_pe)
                self.edge_encoder_er = EREdgeEncoder(
                    cfg.posenc_ERE.dim_pe, use_edge_attr=True)
            else:
                EdgeEncoder = register.edge_encoder_dict[
                    cfg.dataset.edge_encoder_name]
                self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


# ---------------------------------------------------------------------------
# Helper: build a single GPSLayer from global config
# ---------------------------------------------------------------------------

def _build_gps_layer(dim_h, local_gnn_type, global_model_type):
    return GPSLayer(
        dim_h=dim_h,
        local_gnn_type=local_gnn_type,
        global_model_type=global_model_type,
        num_heads=cfg.gt.n_heads,
        pna_degrees=cfg.gt.pna_degrees,
        equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
        dropout=cfg.gt.dropout,
        attn_dropout=cfg.gt.attn_dropout,
        layer_norm=cfg.gt.layer_norm,
        batch_norm=cfg.gt.batch_norm,
        bigbird_cfg=cfg.gt.bigbird,
    )


# ---------------------------------------------------------------------------
# Hierarchical GPS Model
# ---------------------------------------------------------------------------

@register_network('HierarchicalGPSModel')
class HierarchicalGPSModel(nn.Module):
    """Two-level hierarchical GPS model for graph classification.

    A :class:`MinCutPoolLayer` is inserted after ``cfg.gt.hier_pool_after_layer``
    GPS layers.  The remaining GPS layers operate on the coarser graph of
    ``cfg.gt.hier_num_clusters`` supernodes.

    The auxiliary pool loss (mc_loss + o_loss) is stored on ``self.pool_loss``
    after each forward call and should be added to the task loss in the
    training loop (see :mod:`graphgps.train.custom_train`).

    If ``cfg.gt.hier_pool_after_layer == 0``, the model degrades to the
    standard flat ``GPSModel`` without any pooling.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

        # --- Feature encoder -----------------------------------------------
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, (
            "The inner and hidden dims must match.")

        # Parse layer type (e.g. "CustomGatedGCN+Mamba_Hybrid_Degree_Noise")
        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except ValueError:
            raise ValueError(
                f"Unexpected layer type: {cfg.gt.layer_type}. "
                "Expected format: '<local_type>+<global_type>'.")

        dim_h = cfg.gt.dim_hidden
        total_layers = cfg.gt.layers
        pool_after = cfg.gt.hier_pool_after_layer  # 0 means no pooling

        if pool_after < 0 or pool_after > total_layers:
            raise ValueError(
                f"cfg.gt.hier_pool_after_layer={pool_after} must be in "
                f"[0, cfg.gt.layers={total_layers}].")

        # --- Fine-scale GPS layers -----------------------------------------
        self.fine_layers = nn.ModuleList([
            _build_gps_layer(dim_h, local_gnn_type, global_model_type)
            for _ in range(pool_after)
        ])

        # --- MinCutPool (only when pool_after > 0) -------------------------
        self.pool_layer = None
        if pool_after > 0:
            self.pool_layer = MinCutPoolLayer(
                dim_h=dim_h,
                num_clusters=cfg.gt.hier_num_clusters,
            )

        # --- Coarse-scale GPS layers ---------------------------------------
        n_coarse = total_layers - pool_after
        self.coarse_layers = nn.ModuleList([
            _build_gps_layer(dim_h, local_gnn_type, global_model_type)
            for _ in range(n_coarse)
        ])

        # --- Prediction head -----------------------------------------------
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

        # Auxiliary pool loss (set during forward, read in training loop)
        self.pool_loss = None

    # -----------------------------------------------------------------------

    def forward(self, batch):
        # Feature encoding
        batch = self.encoder(batch)
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)

        # Fine-scale processing
        for layer in self.fine_layers:
            batch = layer(batch)

        # Hierarchical pooling (if enabled)
        self.pool_loss = None
        if self.pool_layer is not None:
            batch, pool_loss = self.pool_layer(batch)
            self.pool_loss = pool_loss

        # Coarse-scale processing
        for layer in self.coarse_layers:
            batch = layer(batch)

        # Prediction head: global mean pool → MLP → (pred, true)
        return self.post_mp(batch)
