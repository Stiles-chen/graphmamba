"""MinCut Pooling layer for hierarchical graph processing.

Projects a fine-scale PyG Batch into a coarser batch using differentiable
soft-assignment pooling (MinCutPool).  Returns the coarse batch together with
the auxiliary pool losses (MinCut loss + Orthogonality loss) that should be
weighted and added to the downstream task loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Batch
from torch_geometric.nn.dense import mincut_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj


class MinCutPoolLayer(nn.Module):
    """Hierarchical MinCut pooling layer.

    Converts a fine-scale PyG sparse Batch into a coarser Batch by:
      1. Projecting node features to soft cluster assignments (S) via an MLP.
      2. Applying PyG's ``mincut_pool`` to obtain pooled features & adjacency.
      3. Rebuilding a sparse Batch from the pooled dense tensors.

    The returned auxiliary losses (mc_loss + o_loss) should be accumulated and
    added (with weighting ``cfg.gt.hier_pool_loss_weight``) to the task loss in
    the training loop.

    Args:
        dim_h (int): Node feature dimension (shared for input and output).
        num_clusters (int): Number of supernodes K in the coarser graph.
    """

    def __init__(self, dim_h: int, num_clusters: int):
        super().__init__()
        self.num_clusters = num_clusters

        # Soft-assignment MLP: node features → cluster logits  [N, K]
        self.assignment_net = nn.Sequential(
            nn.Linear(dim_h, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, num_clusters),
        )

        # Project scalar adjacency weight to dim_h edge features for downstream
        # GatedGCN which requires edge_attr of dimension dim_h.
        self.edge_proj = nn.Linear(1, dim_h)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dense_adj_to_sparse(self, adj_pool: torch.Tensor):
        """Convert a batched dense adjacency to global sparse edge_index/weight.

        Args:
            adj_pool: ``[B, K, K]`` non-negative soft pooled adjacency.

        Returns:
            edge_index: ``[2, E]`` with global node indices in ``[0, B*K)``.
            edge_weight: ``[E]`` corresponding non-negative weights.
        """
        B, K, _ = adj_pool.shape
        device = adj_pool.device
        offsets = torch.arange(B, device=device) * K  # [B]

        # Keep all strictly positive entries (near-zero weights bring no info)
        b_idx, row_idx, col_idx = (adj_pool > 0).nonzero(as_tuple=True)
        if b_idx.numel() == 0:
            # Degenerate: return empty tensors
            empty_ei = torch.zeros(2, 0, dtype=torch.long, device=device)
            empty_ew = torch.zeros(0, device=device)
            return empty_ei, empty_ew

        global_row = row_idx + offsets[b_idx]
        global_col = col_idx + offsets[b_idx]
        edge_index = torch.stack([global_row, global_col], dim=0)  # [2, E]
        edge_weight = adj_pool[b_idx, row_idx, col_idx]
        return edge_index, edge_weight

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch):
        """Pool a fine-scale batch to a coarser batch.

        Args:
            batch: PyG :class:`~torch_geometric.data.Batch` with at minimum
                ``.x``, ``.edge_index``, ``.edge_attr``, ``.batch``, and
                ``.split`` (train / val / test string).  If ``.y`` is present
                it is forwarded unchanged to the coarse batch.

        Returns:
            coarse_batch: PyG Batch with ``num_clusters`` supernodes per graph.
            pool_loss: scalar ``mc_loss + o_loss`` (to be added to task loss).
        """
        x = batch.x             # [N_total, D]
        edge_index = batch.edge_index
        batch_vec = batch.batch  # [N_total]

        # 1. Dense conversion -----------------------------------------------
        x_dense, node_mask = to_dense_batch(x, batch_vec)  # [B, N_max, D]
        B, N_max, D = x_dense.shape

        adj_dense = to_dense_adj(
            edge_index, batch_vec, max_num_nodes=N_max
        )  # [B, N_max, N_max]

        # 2. Soft cluster assignment ----------------------------------------
        s = self.assignment_net(x_dense)  # [B, N_max, K]

        # 3. MinCutPool -------------------------------------------------------
        x_pool, adj_pool, mc_loss, o_loss = mincut_pool(
            x_dense, adj_dense, s, mask=node_mask
        )  # x_pool: [B, K, D], adj_pool: [B, K, K]

        # 4. Rebuild sparse coarse batch ------------------------------------
        K = self.num_clusters
        x_new = x_pool.reshape(B * K, D)  # [B*K, D]
        batch_new = torch.arange(B, device=x.device).repeat_interleave(K)

        # Convert pooled adjacency (non-negative values only) to sparse format.
        # Values ≤ 0 carry no structural information; threshold at 0 is the
        # natural choice after F.relu and keeps the coarse graph sparse.
        adj_pool_nn = F.relu(adj_pool)  # ensure non-negative
        edge_index_new, edge_weight_new = self._dense_adj_to_sparse(adj_pool_nn)

        # Project scalar edge weights → dim_h for downstream GatedGCN
        if edge_weight_new.numel() > 0:
            edge_attr_new = self.edge_proj(edge_weight_new.unsqueeze(-1))
        else:
            edge_attr_new = torch.zeros(0, D, device=x.device)

        # Assemble new Batch object
        coarse_batch = Batch()
        coarse_batch.x = x_new
        coarse_batch.edge_index = edge_index_new
        coarse_batch.edge_attr = edge_attr_new
        coarse_batch.batch = batch_new
        coarse_batch.split = batch.split
        # Propagate graph labels (required by the prediction head)
        if hasattr(batch, 'y') and batch.y is not None:
            coarse_batch.y = batch.y

        return coarse_batch, mc_loss + o_loss
