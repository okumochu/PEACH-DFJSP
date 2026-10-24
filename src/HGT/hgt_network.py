import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import HGTConv, Linear, MLP
from torch_geometric.nn.aggr import AttentionalAggregation as WeightedPooling
from torch_geometric.nn.norm import LayerNorm as PygLayerNorm
from torch_geometric.data import Batch


class HGTFeatureExtractor(nn.Module):
    """
    HGT feature extractor with explicit job nodes and PT nodes (no global node).
    Returns node embeddings for each head's weighted pooling.
    
    Node types: op, job, machine, pt
    Edge types: 
        - op -> op (precedes)
        - op -> job (belongs_to)
        - pt <-> op (connects_op, has_pt)
        - pt <-> machine (connects_machine, has_pt)
    """

    def __init__(
        self,
        op_feature_dim: int,
        job_feature_dim: int,
        machine_feature_dim: int,
        pt_feature_dim: int,
        hidden_dim: int,
        num_hgt_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_hgt_layers = num_hgt_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # All nodes use same dimension
        hgt_out_dim = hidden_dim * num_heads

        # Store dimensions for each node type
        self.node_dims = {
            'op': hgt_out_dim,
            'job': hgt_out_dim,
            'machine': hgt_out_dim,
            'pt': hgt_out_dim,
        }

        # Embeddings
        self.op_embedding = Linear(op_feature_dim, hgt_out_dim, bias=True)
        self.job_embedding = Linear(job_feature_dim, hgt_out_dim, bias=True)
        self.machine_embedding = Linear(machine_feature_dim, hgt_out_dim, bias=True)
        self.pt_embedding = Linear(pt_feature_dim, hgt_out_dim, bias=True)

        # Metadata: PT nodes connect op and machine (replacing direct op<->machine edges)
        self.metadata = (
            ["op", "job", "machine", "pt"],
            [
                ("op", "precedes", "op"),
                ("op", "belongs_to", "job"),
                ("pt", "connects_op", "op"),
                ("op", "has_pt", "pt"),
                ("pt", "connects_machine", "machine"),
                ("machine", "has_pt", "pt"),
            ],
        )

        self.hgt_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.residual_dropout = nn.Dropout(dropout)
        # Gradient checkpointing of the per-layer HGTConv call (set by HGT from
        # config.use_grad_checkpoint). Trades a recompute in backward for not
        # retaining the dense per-edge attention/scatter activations.
        self.use_grad_checkpoint = False

        # Build HGT layers with uniform dimensions
        for layer_idx in range(num_hgt_layers):
            hgt_layer = HGTConv(
                in_channels=hgt_out_dim,  # All nodes same dim
                out_channels=hgt_out_dim,
                metadata=self.metadata,
                heads=num_heads,
            )
            self.hgt_layers.append(hgt_layer)
            self.layer_norms.append(nn.ModuleDict({
                nt: PygLayerNorm(hgt_out_dim, mode='node') for nt in self.metadata[0]
            }))


    def forward(self, graph):
        """
        Args:
            graph: List of HeteroData graphs

        Returns:
            (fea_job, fea_pt, flat_node_embeddings, batch_vectors)
            - fea_job: [batch_size, max_jobs, hidden_dim]
            - fea_pt: [batch_size, max_pts, hidden_dim]
            - flat_node_embeddings: per-type node embedding tensors
            - batch_vectors: per-type graph indices used for weighted pooling
        """
        num_jobs_per_graph = [g["job"].x.size(0) for g in graph]
        num_pts_per_graph = [g["pt"].x.size(0) for g in graph]

        batched_graph = Batch.from_data_list(graph)

        batch_vectors = {
            ntype: batched_graph[ntype].batch for ntype in self.metadata[0]
        }

        op_x = self.op_embedding(batched_graph["op"].x)
        job_x = self.job_embedding(batched_graph["job"].x)
        machine_x = self.machine_embedding(batched_graph["machine"].x)
        pt_x = self.pt_embedding(batched_graph["pt"].x)

        x_dict = {"op": op_x, "job": job_x, "machine": machine_x, "pt": pt_x}

        for layer_idx, hgt_layer in enumerate(self.hgt_layers):
            residual_dict = {k: v for k, v in x_dict.items()}
            if self.use_grad_checkpoint and torch.is_grad_enabled():
                out_dict = checkpoint(
                    hgt_layer, x_dict, batched_graph.edge_index_dict,
                    use_reentrant=False, preserve_rng_state=True)
            else:
                out_dict = hgt_layer(x_dict, batched_graph.edge_index_dict)

            x_dict = {}
            for node_type, residual in residual_dict.items():
                hgt_output = out_dict.get(node_type, torch.zeros_like(residual))
                combined = hgt_output + residual
                x_dict[node_type] = self.layer_norms[layer_idx][node_type](
                    self.residual_dropout(combined)
                )

        final_op_embeddings = x_dict["op"]
        final_job_embeddings = x_dict["job"]
        final_machine_embeddings = x_dict["machine"]
        final_pt_embeddings = x_dict["pt"]

        # Unbatch embeddings (only job and pt, as op and machine are not used)
        job_splits = torch.split(final_job_embeddings, num_jobs_per_graph)
        pt_splits = torch.split(final_pt_embeddings, num_pts_per_graph)

        # Padding the node embeddings to the max size (only job and pt)
        batch_size = len(graph)
        device = final_job_embeddings.device
        hidden_dim = final_job_embeddings.size(-1)

        job_node_embedding = torch.zeros(batch_size, max(num_jobs_per_graph), hidden_dim, device=device)
        pt_node_embedding = torch.zeros(batch_size, max(num_pts_per_graph), hidden_dim, device=device)

        for i, (job_feat, pt_feat) in enumerate(zip(job_splits, pt_splits)):
            job_node_embedding[i, : job_feat.size(0), :] = job_feat
            pt_node_embedding[i, : pt_feat.size(0), :] = pt_feat

        # Both heads do their own weighted pooling; expose flat per-node
        # embeddings and graph-batch indices for them to consume.
        flat_node_embeddings = {
            'op': final_op_embeddings,
            'job': final_job_embeddings,
            'machine': final_machine_embeddings,
            'pt': final_pt_embeddings,
        }

        return (
            job_node_embedding,
            pt_node_embedding,
            flat_node_embeddings,
            batch_vectors,
        )


class HGTFeatureExtractorNoPT(nn.Module):
    """HGT feature extractor for the graph schema without PT nodes.

    Node types: op, job, machine
    Edge types:
        - op -> op (precedes)
        - op -> job (belongs_to)
        - op -> machine (processed_by)
        - machine -> op (processes)
    """

    def __init__(
        self,
        op_feature_dim: int,
        job_feature_dim: int,
        machine_feature_dim: int,
        hidden_dim: int,
        num_hgt_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        hgt_out_dim = hidden_dim * num_heads
        self.metadata = (
            ["op", "job", "machine"],
            [
                ("op", "precedes", "op"),
                ("op", "belongs_to", "job"),
                ("op", "processed_by", "machine"),
                ("machine", "processes", "op"),
            ],
        )

        self.op_embedding = Linear(op_feature_dim, hgt_out_dim, bias=True)
        self.job_embedding = Linear(job_feature_dim, hgt_out_dim, bias=True)
        self.machine_embedding = Linear(machine_feature_dim, hgt_out_dim, bias=True)

        self.hgt_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.residual_dropout = nn.Dropout(dropout)
        self.use_grad_checkpoint = False  # set by HGT from config.use_grad_checkpoint
        for _ in range(num_hgt_layers):
            self.hgt_layers.append(HGTConv(
                in_channels=hgt_out_dim,
                out_channels=hgt_out_dim,
                metadata=self.metadata,
                heads=num_heads,
            ))
            self.layer_norms.append(nn.ModuleDict({
                nt: PygLayerNorm(hgt_out_dim, mode='node')
                for nt in self.metadata[0]
            }))

    def forward(self, graph):
        num_ops_per_graph = [g["op"].x.size(0) for g in graph]
        num_jobs_per_graph = [g["job"].x.size(0) for g in graph]
        num_machines_per_graph = [g["machine"].x.size(0) for g in graph]

        batched_graph = Batch.from_data_list(graph)
        batch_vectors = {
            ntype: batched_graph[ntype].batch for ntype in self.metadata[0]
        }

        x_dict = {
            "op": self.op_embedding(batched_graph["op"].x),
            "job": self.job_embedding(batched_graph["job"].x),
            "machine": self.machine_embedding(batched_graph["machine"].x),
        }

        for layer_idx, hgt_layer in enumerate(self.hgt_layers):
            residual_dict = {k: v for k, v in x_dict.items()}
            if self.use_grad_checkpoint and torch.is_grad_enabled():
                out_dict = checkpoint(
                    hgt_layer, x_dict, batched_graph.edge_index_dict,
                    use_reentrant=False, preserve_rng_state=True)
            else:
                out_dict = hgt_layer(x_dict, batched_graph.edge_index_dict)

            x_dict = {}
            for node_type, residual in residual_dict.items():
                hgt_output = out_dict.get(node_type, torch.zeros_like(residual))
                combined = hgt_output + residual
                x_dict[node_type] = self.layer_norms[layer_idx][node_type](
                    self.residual_dropout(combined)
                )

        final_op_embeddings = x_dict["op"]
        final_job_embeddings = x_dict["job"]
        final_machine_embeddings = x_dict["machine"]

        op_splits = torch.split(final_op_embeddings, num_ops_per_graph)
        job_splits = torch.split(final_job_embeddings, num_jobs_per_graph)
        machine_splits = torch.split(final_machine_embeddings, num_machines_per_graph)

        batch_size = len(graph)
        device = final_op_embeddings.device
        hidden_dim = final_op_embeddings.size(-1)

        op_node_embedding = torch.zeros(
            batch_size, max(num_ops_per_graph), hidden_dim, device=device)
        job_node_embedding = torch.zeros(
            batch_size, max(num_jobs_per_graph), hidden_dim, device=device)
        machine_node_embedding = torch.zeros(
            batch_size, max(num_machines_per_graph), hidden_dim, device=device)

        for i, (op_feat, job_feat, machine_feat) in enumerate(
                zip(op_splits, job_splits, machine_splits)):
            op_node_embedding[i, : op_feat.size(0), :] = op_feat
            job_node_embedding[i, : job_feat.size(0), :] = job_feat
            machine_node_embedding[i, : machine_feat.size(0), :] = machine_feat

        flat_node_embeddings = {
            'op': final_op_embeddings,
            'job': final_job_embeddings,
            'machine': final_machine_embeddings,
        }

        return (
            op_node_embedding,
            job_node_embedding,
            machine_node_embedding,
            flat_node_embeddings,
            batch_vectors,
        )



class HGT(nn.Module):
    """
    HGT-based network with PT nodes for unified graph-based policy.
    
    Each head has its own per-node-type weighted pools (no shared/mean
    pooling). The two ModuleDicts are independent so actor and critic can
    learn different salience patterns.

    Actor input (per (job, machine) candidate):
        [candidate_job, candidate_pt] + 4 actor weighted pools (broadcast)
        = 6 * embedding_dim → actor MLP → scalar logit per candidate.

    Critic input (per state):
        4 critic weighted pools + a weighted pool over J*M of cat(job_emb, pt_emb)
        with invalid candidates masked
        = 4*d + 2*d = 6 * embedding_dim → critic MLP → V.

    Both heads use a progressive-halving MLP: input -> input/2 -> ... -> bottleneck -> 1
    """

    def __init__(self, config):
        super().__init__()
        device = torch.device(config.device)

        self.use_pt_nodes = bool(getattr(config, "use_pt_nodes", True))
        self.op_feature_dim = getattr(config, "op_feature_dim", 9)
        self.job_feature_dim = getattr(config, "job_feature_dim", 7)
        default_machine_dim = 6 if self.use_pt_nodes else 7
        self.machine_feature_dim = getattr(
            config, "machine_feature_dim", default_machine_dim)
        self.pt_feature_dim = getattr(config, "pt_feature_dim", 2)
        self.hidden_dim = config.hidden_dim
        self.num_hgt_layers = config.num_hgt_layers
        self.num_heads = config.num_heads
        self.dropout = config.dropout

        self.embedding_output_dim = self.hidden_dim * self.num_heads
        if self.use_pt_nodes:
            self.feature_extractor = HGTFeatureExtractor(
                op_feature_dim=self.op_feature_dim,
                job_feature_dim=self.job_feature_dim,
                machine_feature_dim=self.machine_feature_dim,
                pt_feature_dim=self.pt_feature_dim,
                hidden_dim=self.hidden_dim,
                num_hgt_layers=self.num_hgt_layers,
                num_heads=self.num_heads,
                dropout=self.dropout,
            ).to(device)
        else:
            self.feature_extractor = HGTFeatureExtractorNoPT(
                op_feature_dim=self.op_feature_dim,
                job_feature_dim=self.job_feature_dim,
                machine_feature_dim=self.machine_feature_dim,
                hidden_dim=self.hidden_dim,
                num_hgt_layers=self.num_hgt_layers,
                num_heads=self.num_heads,
                dropout=self.dropout,
            ).to(device)

        # Opt-in gradient checkpointing of the HGTConv layers (memory<->compute
        # trade for the dense PT-edge attention; exact gradients). Off by default.
        self.feature_extractor.use_grad_checkpoint = bool(
            getattr(config, "use_grad_checkpoint", False))

        d = self.embedding_output_dim
        self.node_types = self.feature_extractor.metadata[0]

        self.actor_node_pools = nn.ModuleDict({
            nt: WeightedPooling(gate_nn=nn.Linear(d, 1))
            for nt in self.node_types
        }).to(device)
        self.critic_node_pools = nn.ModuleDict({
            nt: WeightedPooling(gate_nn=nn.Linear(d, 1))
            for nt in self.node_types
        }).to(device)
        if self.use_pt_nodes:
            # Candidate-axis gate over cat(job_emb, pt_emb) per (j, m). Critic only.
            self.critic_candidate_gate = nn.Linear(2 * d, 1).to(device)

        actor_input_dim = 6 * d
        actor_channels = getattr(config, 'actor_channels', None) or [actor_input_dim, 512, 256, 128, 64, 32, 1]
        self.actor = MLP(
            channel_list=actor_channels,
            act=torch.tanh,
            dropout=0.0,
            norm=None,
            bias=True,
        ).to(device)

        critic_input_dim = 6 * d if self.use_pt_nodes else 3 * d
        critic_channels = getattr(config, 'critic_channels', None) or [critic_input_dim, 512, 256, 128, 64, 32, 1]
        # ---- PopArt state ------------------------------------------
        # Buffers, so the existing policy.state_dict() save/load and the
        # existing policy -> policy_old buffer copy in PPO.update() carry
        # them with no extra checkpoint code. mu/sigma are the EFFECTIVE
        # (bias-corrected) statistics used by every path below; _ema_* are
        # the raw accumulators. With use_popart False they stay (0, 1) and
        # every PopArt path is an exact identity.
        self.use_popart = bool(getattr(config, 'use_popart', False))
        self.popart_beta = float(getattr(config, 'popart_beta', 0.1))
        # Set by load_state_dict: a loaded head carries real information and
        # must be preserved across the first rescale; a cold one must not be.
        self._popart_head_is_warm = False
        self.register_buffer('popart_mu', torch.zeros(1, device=device))
        self.register_buffer('popart_sigma', torch.ones(1, device=device))
        self.register_buffer('popart_ema_mu', torch.zeros(1, device=device))
        self.register_buffer('popart_ema_var', torch.zeros(1, device=device))
        self.register_buffer('popart_steps', torch.zeros(1, device=device))

        self.critic = MLP(
            channel_list=critic_channels,
            act=torch.tanh,
            dropout=0.0,
            norm=None,
            bias=True,
        ).to(device)

    def forward(
        self,
        candidate,
        candidate_mask,
        candidate_pt_indices,
        graph,
    ):
        """
        Forward pass with PT nodes for unified graph-based policy.
        
        Args:
            candidate: Candidate operations [sz_b, J]
            candidate_mask: Candidate pair mask [sz_b, J, M]
            candidate_pt_indices: [sz_b, J, M] indices into PT nodes
            graph: HeteroData graph for HGT processing
            
        Returns:
            pi: Policy [sz_b, J*M]
            v: Value [sz_b, 1]
        """
        if not self.use_pt_nodes:
            return self._forward_no_pt(
                candidate=candidate,
                candidate_mask=candidate_mask,
                graph=graph,
            )

        # Extract features. Pure feature extractor — no global pooling here;
        # actor and critic each compute their own weighted pools below.
        (
            job_node_embedding,
            pt_node_embedding,
            flat_node_embeddings,
            batch_vectors,
        ) = self.feature_extractor(graph)

        actor_weighted_pools = {
            nt: self.actor_node_pools[nt](
                flat_node_embeddings[nt], batch_vectors[nt]
            )
            for nt in self.node_types
        }

        # Derive dimensions from available tensors
        sz_b = job_node_embedding.size(0)  # batch size
        J = candidate.size(1)  # number of candidate slots (may include pre-allocated)
        M = candidate_mask.size(2)  # number of machines (from mask shape [sz_b, J, M])
        d = job_node_embedding.size(-1)  # hidden dimension

        # Pad job embedding to candidate dimension if graph has fewer job nodes
        # (pre-allocated slots exist in candidate but not in graph)
        if job_node_embedding.size(1) < J:
            pad = torch.zeros(sz_b, J - job_node_embedding.size(1), d,
                              device=job_node_embedding.device)
            job_node_embedding = torch.cat([job_node_embedding, pad], dim=1)

        # 1. Candidate Job Embedding
        # fea_job is [sz_b, J, d]. Candidate column j corresponds to job j.
        fea_job_serialized = (
            job_node_embedding.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        )
        
        # 2. Candidate PT Embedding
        # candidate_pt_indices is [sz_b, J, M] containing indices into fea_pt
        # We need to gather PT embeddings for each (job, machine) pair
        # Handle -1 indices (incompatible pairs) by clamping to 0 and later masking
        pt_indices_flat = candidate_pt_indices.reshape(sz_b, J * M)  # [sz_b, J*M]
        pt_indices_safe = pt_indices_flat.clamp(min=0)  # Clamp -1 to 0 for gather
        pt_indices_expanded = pt_indices_safe.unsqueeze(-1).expand(-1, -1, d)  # [sz_b, J*M, d]
        
        fea_pt_serialized = torch.gather(pt_node_embedding, 1, pt_indices_expanded)  # [sz_b, J*M, d]
        
        # Zero out incompatible pairs (where original index was -1)
        invalid_mask = (pt_indices_flat < 0).unsqueeze(-1).expand(-1, -1, d)
        fea_pt_serialized = fea_pt_serialized.masked_fill(invalid_mask, 0.0)

        # 3-6. Actor's per-type weighted pools, broadcast to every candidate.
        global_op_serialized = (
            actor_weighted_pools['op'].unsqueeze(1).expand(-1, M * J, -1)
        )
        global_job_serialized = (
            actor_weighted_pools['job'].unsqueeze(1).expand(-1, M * J, -1)
        )
        global_machine_serialized = (
            actor_weighted_pools['machine'].unsqueeze(1).expand(-1, M * J, -1)
        )
        global_pt_serialized = (
            actor_weighted_pools['pt'].unsqueeze(1).expand(-1, M * J, -1)
        )

        candidate_feature = torch.cat(
            (
                fea_job_serialized,
                fea_pt_serialized,
                global_op_serialized,
                global_job_serialized,
                global_machine_serialized,
                global_pt_serialized,
            ),
            dim=-1,
        )

        candidate_scores = self.actor(candidate_feature).squeeze(-1)

        mask = candidate_mask.reshape(sz_b, -1).bool()
        candidate_scores[mask] = float("-inf")

        all_masked = torch.isinf(candidate_scores).all(dim=1)
        if all_masked.any():
            candidate_scores[all_masked] = 0.0

        pi = F.softmax(candidate_scores, dim=1)

        v = self._critic_forward(
            flat_node_embeddings, batch_vectors,
            fea_job_serialized, fea_pt_serialized, mask, all_masked,
        )

        return pi, v

    def _forward_no_pt(self, candidate, candidate_mask, graph):
        (
            op_node_embedding,
            job_node_embedding,
            machine_node_embedding,
            flat_node_embeddings,
            batch_vectors,
        ) = self.feature_extractor(graph)

        actor_weighted_pools = {
            nt: self.actor_node_pools[nt](
                flat_node_embeddings[nt], batch_vectors[nt]
            )
            for nt in self.node_types
        }
        critic_weighted_pools = [
            self.critic_node_pools[nt](
                flat_node_embeddings[nt], batch_vectors[nt]
            )
            for nt in self.node_types
        ]

        sz_b = job_node_embedding.size(0)
        J = candidate.size(1)
        M = candidate_mask.size(2)
        d = job_node_embedding.size(-1)

        if job_node_embedding.size(1) < J:
            pad = torch.zeros(sz_b, J - job_node_embedding.size(1), d,
                              device=job_node_embedding.device)
            job_node_embedding = torch.cat([job_node_embedding, pad], dim=1)
        if machine_node_embedding.size(1) < M:
            pad = torch.zeros(sz_b, M - machine_node_embedding.size(1), d,
                              device=machine_node_embedding.device)
            machine_node_embedding = torch.cat([machine_node_embedding, pad], dim=1)

        num_ops_available = op_node_embedding.size(1)
        cand_gather = candidate.clamp(min=0, max=max(num_ops_available - 1, 0))
        cand_gather_exp = cand_gather.unsqueeze(-1).expand(-1, -1, d)
        op_emb_cand = torch.gather(op_node_embedding, 1, cand_gather_exp)
        invalid_cand_mask = (candidate < 0).unsqueeze(-1).expand(-1, -1, d)
        op_emb_cand = op_emb_cand.masked_fill(invalid_cand_mask, 0.0)

        fea_op_serialized = (
            op_emb_cand.unsqueeze(2).expand(-1, -1, M, -1)
            .reshape(sz_b, J * M, d)
        )
        fea_job_serialized = (
            job_node_embedding.unsqueeze(2).expand(-1, -1, M, -1)
            .reshape(sz_b, J * M, d)
        )
        fea_mch_serialized = (
            machine_node_embedding[:, :M].unsqueeze(1)
            .expand(-1, J, -1, -1)
            .reshape(sz_b, J * M, d)
        )

        global_op_serialized = (
            actor_weighted_pools['op'].unsqueeze(1).expand(-1, J * M, -1)
        )
        global_job_serialized = (
            actor_weighted_pools['job'].unsqueeze(1).expand(-1, J * M, -1)
        )
        global_machine_serialized = (
            actor_weighted_pools['machine'].unsqueeze(1).expand(-1, J * M, -1)
        )

        candidate_feature = torch.cat(
            (
                fea_op_serialized,
                fea_job_serialized,
                fea_mch_serialized,
                global_op_serialized,
                global_job_serialized,
                global_machine_serialized,
            ),
            dim=-1,
        )
        candidate_scores = self.actor(candidate_feature).squeeze(-1)

        mask = candidate_mask.reshape(sz_b, -1).bool()
        candidate_scores[mask] = float("-inf")
        all_masked = torch.isinf(candidate_scores).all(dim=1)
        if all_masked.any():
            candidate_scores[all_masked] = 0.0

        pi = F.softmax(candidate_scores, dim=1)
        v = self.popart_denormalize(
            self.critic(torch.cat(critic_weighted_pools, dim=-1)))
        return pi, v

    POPART_BUFFER_KEYS = ('popart_mu', 'popart_sigma', 'popart_ema_mu',
                          'popart_ema_var', 'popart_steps')

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Load weights, reconciling checkpoints from before PopArt.

        Two problems are handled here rather than at each call site,
        because there are five of them (model_utils.load_model and its
        siblings, _load_resume_checkpoint x3, HGTTrainer.load_model) and
        missing one reintroduces the bug silently.

        1. A checkpoint written before PopArt has none of these buffers,
           and a strict load would reject it. Every artifact of every
           finished run is such a checkpoint, so they are filled with the
           identity statistics (mu=0, sigma=1), under which
           popart_denormalize is an exact no-op and the raw-unit head
           loaded from that checkpoint is immediately correct.

        2. The reverse direction is the dangerous one. A PopArt model and
           a non-PopArt model have IDENTICAL key sets, so loading a
           PopArt-trained checkpoint into a use_popart=False model
           succeeds silently, the statistics are ignored, and the critic
           head - which is in normalized units - is read as if it were in
           real units. Measured error: ~100% of the value scale. Nothing
           raises, the greedy actor still returns plausible schedules, and
           only value-based diagnostics look wrong. Refuse it loudly.
        """
        import torch as _torch
        sd = state_dict
        have = [k for k in self.POPART_BUFFER_KEYS if k in sd]
        if have:
            def _scalar(key, default):
                v = sd.get(key)
                if v is None:
                    return default
                try:
                    return float(_torch.as_tensor(v).reshape(-1)[0])
                except Exception:
                    return default
            mu = _scalar('popart_mu', 0.0)
            sigma = _scalar('popart_sigma', 1.0)
            steps = _scalar('popart_steps', 0.0)
            trained_with_popart = (steps > 0.0) or (sigma != 1.0) or (mu != 0.0)
            if trained_with_popart and not self.use_popart:
                raise RuntimeError(
                    'refusing to load a PopArt-trained checkpoint into a '
                    'model with use_popart=False (checkpoint statistics: '
                    f'mu={mu:.6g}, sigma={sigma:.6g}, steps={steps:.0f}). '
                    'The critic head of such a checkpoint is in NORMALIZED '
                    'units; without the denormalizer its predictions are '
                    'wrong by roughly the full value scale, and nothing '
                    'downstream would raise. Set use_popart=True for this '
                    'model, or use a checkpoint trained without PopArt.')
        else:
            sd = dict(sd)
            for key in self.POPART_BUFFER_KEYS:
                sd[key] = getattr(self, key).detach().clone()
        result = super().load_state_dict(sd, strict=strict, **kwargs)
        self._popart_head_is_warm = True
        return result

    def popart_denormalize(self, v_norm):
        """Map the critic's normalized output back to real objective units:
        v = sigma * v_norm + mu. Identity when PopArt is off, so GAE,
        bootstrapping, value targets and logging all keep seeing real units
        and none of them needs a PopArt branch."""
        if not self.use_popart:
            return v_norm
        return v_norm * self.popart_sigma + self.popart_mu

    def popart_normalize(self, y):
        """Inverse of popart_denormalize: real units -> critic units."""
        if not self.use_popart:
            return y
        return (y - self.popart_mu) / self.popart_sigma

    @torch.no_grad()
    def popart_update_stats(self, targets):
        """Fold a batch of REAL-UNIT value targets into the running stats,
        then rescale the critic's last layer so the denormalized function
        it computes is UNCHANGED.

        That invariant is the whole point: the target distribution can move
        (a curriculum changing problem size in either direction), yet
        everything the critic already learned stays valid instead of being
        thrown away.

        Call EXACTLY ONCE per PPO update, before any gradient step of it.
        """
        if not self.use_popart or targets.numel() == 0:
            return
        # float64 throughout: the batch statistics of this project's value
        # targets sit in the thousands, and E[y^2] - E[y]^2 in float32
        # cancels catastrophically there. torch.var is a stable two-pass, so
        # the variance is taken directly rather than reconstructed from a
        # second moment.
        y = targets.detach().double().reshape(-1)
        batch_mu = y.mean()
        batch_var = y.var(unbiased=False) if y.numel() > 1 else torch.zeros_like(batch_mu)
        if not (torch.isfinite(batch_mu) and torch.isfinite(batch_var)):
            return

        b = self.popart_beta
        self.popart_ema_mu.mul_(1 - b).add_(b * batch_mu.to(self.popart_ema_mu.dtype))
        self.popart_ema_var.mul_(1 - b).add_(b * batch_var.to(self.popart_ema_var.dtype))
        self.popart_steps.add_(1)

        # Bias correction, same idea as Adam. Without it the first updates
        # are dragged toward the (0, 1) init, so the very first rescale
        # would be wrong by a large factor.
        corr = torch.clamp(1.0 - (1.0 - b) ** self.popart_steps, min=1e-8)
        mu_new = self.popart_ema_mu / corr
        var_new = torch.clamp(self.popart_ema_var / corr, min=0.0)
        # Floor sigma RELATIVE to |mu| as well as absolutely. A tiny absolute
        # floor is the worst possible fallback for a near-constant target
        # stream in the thousands: it would rescale the head by mu/1e-4.
        sigma_floor = torch.clamp(mu_new.abs() * 1e-6, min=1e-4)
        sigma_new = torch.clamp(var_new.sqrt(), max=1e6)
        sigma_new = torch.maximum(sigma_new, sigma_floor)

        mu_old = self.popart_mu.clone()
        sigma_old = self.popart_sigma.clone()

        if self.popart_steps.item() <= 1 and not self._popart_head_is_warm:
            # Cold start. Preserving the output here would faithfully preserve
            # a MEANINGLESS one: an untrained head predicts ~0 real units
            # while targets are in the thousands, so preservation parks the
            # bias at -mu/sigma and Adam needs thousands of updates to walk it
            # back. Predicting the mean (normalized 0) is strictly better and
            # is what the head would learn first anyway. Only done cold; after
            # any load the head means something and is preserved.
            if self.critic.lins[-1].bias is not None:
                self.critic.lins[-1].bias.data.zero_()
            self.popart_mu.copy_(mu_new)
            self.popart_sigma.copy_(sigma_new)
            return

        # Output-preserving rescale of the LAST critic layer only.
        # Require: sigma_new*(W'x + b') + mu_new == sigma_old*(Wx + b) + mu_old
        #   =>  W' = W * sigma_old / sigma_new
        #       b' = (sigma_old * b + mu_old - mu_new) / sigma_new
        head = self.critic.lins[-1]
        head.weight.data.mul_(sigma_old / sigma_new)
        if head.bias is not None:
            head.bias.data.copy_(
                (sigma_old * head.bias.data + mu_old - mu_new) / sigma_new)

        self.popart_mu.copy_(mu_new)
        self.popart_sigma.copy_(sigma_new)

    def _critic_forward(
        self,
        flat_node_embeddings,
        batch_vectors,
        fea_job_serialized,
        fea_pt_serialized,
        mask,
        all_masked,
    ):
        """Critic head — shared by all HGT subclasses.

        Combines 4 per-node-type weighted pools over flat node embeddings
        with one weighted pool over cat(job_emb, pt_emb) per (j, m) (with
        invalid pairs masked out so they don't dilute the candidate pool).
        """
        critic_weighted_pools = [
            self.critic_node_pools[nt](
                flat_node_embeddings[nt], batch_vectors[nt]
            )
            for nt in self.node_types
        ]

        candidate_local = torch.cat((fea_job_serialized, fea_pt_serialized), dim=-1)
        gate_logits = self.critic_candidate_gate(candidate_local).squeeze(-1)
        gate_logits = gate_logits.masked_fill(mask, float("-inf"))
        # Done states have every candidate masked; uniform weights keep the
        # softmax finite and the V output is irrelevant in that case anyway.
        if all_masked.any():
            gate_logits[all_masked] = 0.0
        pooling_weights = F.softmax(gate_logits, dim=1)
        candidate_weighted_pool = torch.einsum("bn,bnd->bd", pooling_weights, candidate_local)

        critic_context = torch.cat(
            critic_weighted_pools + [candidate_weighted_pool], dim=-1
        )
        return self.popart_denormalize(self.critic(critic_context))


__all__ = ["HGT", "HGTFeatureExtractor", "HGTFeatureExtractorNoPT"]
