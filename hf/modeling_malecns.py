"""The MaleCNS v1.0 fruit-fly connectome as a loadable PyTorch object.

Not a language model and not a biological simulation: this is the anatomical wiring diagram
(neurons, directed synaptic connections, synapse counts, coarse annotations) packaged so that
`AutoModel.from_pretrained(..., trust_remote_code=True)` returns the graph as tensors.

    graph.edge_index      int32 [2, E]   (source, destination) as contiguous neuron indices
    graph.synapse_count   int32 [E]      exact number of synaptic contacts per connection
    graph.edge_weight_bf16 bf16 [E]     derived convenience copy of the counts (exact up to 256, then rounded)
    graph.node_id         int64 [N]      MaleCNS body ids
    neuron.region         int8  [N]      index into config.region_labels
    neuron.superclass     int8  [N]      index into config.superclass_labels
    neuron.nt_class       int8  [N]      index into config.nt_labels (consensus neurotransmitter)
    neuron.nt_confidence  bf16  [N]

`forward(state)` performs one linear propagation step `W @ state` with rows = destination,
i.e. the summed synapse-weighted input every neuron receives from the given presynaptic activity.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .configuration_malecns import MaleCNSConfig


class _Buffers(nn.Module):
    def __init__(self, **tensors):
        super().__init__()
        for k, v in tensors.items():
            self.register_buffer(k, v)


class MaleCNSConnectome(PreTrainedModel):
    config_class = MaleCNSConfig
    base_model_prefix = "malecns"
    supports_gradient_checkpointing = False

    def __init__(self, config: MaleCNSConfig):
        super().__init__(config)
        N, E = config.num_neurons, config.num_edges
        self.graph = _Buffers(
            edge_index=torch.zeros(2, E, dtype=torch.int32),
            synapse_count=torch.zeros(E, dtype=torch.int32),
            edge_weight_bf16=torch.zeros(E, dtype=torch.bfloat16),
            node_id=torch.zeros(N, dtype=torch.int64),
        )
        self.neuron = _Buffers(
            region=torch.zeros(N, dtype=torch.int8),
            superclass=torch.zeros(N, dtype=torch.int8),
            nt_class=torch.zeros(N, dtype=torch.int8),
            nt_confidence=torch.zeros(N, dtype=torch.bfloat16),
        )
        self.post_init()

    def _init_weights(self, module):  # nothing is learned
        pass

    # ---- basic accessors -----------------------------------------------------------------------
    @property
    def num_neurons(self) -> int:
        return self.config.num_neurons

    @property
    def src(self) -> torch.Tensor:
        return self.graph.edge_index[0].long()

    @property
    def dst(self) -> torch.Tensor:
        return self.graph.edge_index[1].long()

    def in_degree(self) -> torch.Tensor:
        return torch.bincount(self.dst, minlength=self.num_neurons)

    def out_degree(self) -> torch.Tensor:
        return torch.bincount(self.src, minlength=self.num_neurons)

    def region_mask(self, *names: str) -> torch.Tensor:
        idx = [self.config.region_labels.index(n) for n in names]
        return torch.isin(self.neuron.region, torch.tensor(idx, dtype=torch.int8))

    def superclass_mask(self, *names: str) -> torch.Tensor:
        idx = [self.config.superclass_labels.index(n) for n in names]
        return torch.isin(self.neuron.superclass, torch.tensor(idx, dtype=torch.int8))

    def subset_mask(self, name: str) -> torch.Tensor:
        """Named neuron subsets from config.subsets (full_cns, central_brain, optic_lobes, vnc, cb_sensory, ...)."""
        spec = self.config.subsets[name]
        if spec == "all":
            return torch.ones(self.num_neurons, dtype=torch.bool)
        return self.superclass_mask(*spec)

    # ---- the matrix ----------------------------------------------------------------------------
    def edge_values(self, exact: bool = True, normalize: Optional[str] = None) -> torch.Tensor:
        """Per-edge values in fp32. `exact` uses the int32 counts; otherwise the bf16 copy.
        normalize: None | 'log1p' | 'in_degree' (divide by the destination's summed incoming synapses)."""
        v = (self.graph.synapse_count if exact else self.graph.edge_weight_bf16).float()
        if normalize == "log1p":
            v = torch.log1p(v)
        elif normalize == "in_degree":
            tot = torch.zeros(self.num_neurons, dtype=torch.float32).index_add_(0, self.dst, v)
            v = v / tot.clamp(min=1)[self.dst]
        elif normalize is not None:
            raise ValueError(normalize)
        return v

    def sparse_weight(self, exact: bool = True, normalize: Optional[str] = None, min_synapses: int = 1,
                      nodes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sparse COO [N, N] with rows = destination, columns = source. `nodes` (bool mask or index
        tensor) restricts to the induced subgraph and relabels to 0..k-1."""
        src, dst, v = self.src, self.dst, self.edge_values(exact, normalize)
        keep = self.graph.synapse_count >= min_synapses
        n = self.num_neurons
        if nodes is not None:
            mask = nodes if nodes.dtype == torch.bool else torch.zeros(n, dtype=torch.bool).index_fill_(0, nodes.long(), True)
            new = torch.full((n,), -1, dtype=torch.long)
            new[mask] = torch.arange(int(mask.sum()))
            keep &= mask[src] & mask[dst]
            src, dst, n = new[src], new[dst], int(mask.sum())
        return torch.sparse_coo_tensor(torch.stack([dst[keep], src[keep]]), v[keep], (n, n)).coalesce()

    def subgraph(self, nodes: torch.Tensor, min_synapses: int = 1) -> dict:
        """Induced subgraph as plain tensors: edge_index [2,E'] (relabelled), synapse_count [E'], node_id [k]."""
        n = self.num_neurons
        mask = nodes if nodes.dtype == torch.bool else torch.zeros(n, dtype=torch.bool).index_fill_(0, nodes.long(), True)
        new = torch.full((n,), -1, dtype=torch.long)
        new[mask] = torch.arange(int(mask.sum()))
        keep = mask[self.src] & mask[self.dst] & (self.graph.synapse_count >= min_synapses)
        return {"edge_index": torch.stack([new[self.src[keep]], new[self.dst[keep]]]),
                "synapse_count": self.graph.synapse_count[keep], "node_id": self.graph.node_id[mask]}

    def forward(self, state: torch.Tensor, exact: bool = True, normalize: Optional[str] = None) -> torch.Tensor:
        """One propagation step: incoming[b, i] = sum_j W_ij state[b, j]. state: [B, N] -> [B, N], fp32."""
        W = self.sparse_weight(exact, normalize)
        return torch.sparse.mm(W, state.float().T).T
