"""The MaleCNS v1.0 fruit-fly connectome as a loadable PyTorch object.

Not a language model and not a biological simulation. This is the release's connectivity table
(segment-to-segment anatomical connection weights = synaptic contact counts) plus its neuron
annotations and neurotransmitter predictions, mirrored losslessly into safetensors:

    neuron_id          int64   [N]   MaleCNS body id of neuron index i
    edge_src           int32   [E]   presynaptic neuron index, in the release file's row order
    edge_dst           int32   [E]   postsynaptic neuron index
    synapse_count      int32   [E]   the release's `weight` column, untouched
    neuron_status      int8    [N]   config.status_labels   (proofreading status from the annotations)
    neuron_superclass  int8    [N]   config.superclass_labels
    nt_neuron_index    int32   [M]   rows of the neurotransmitter table, as neuron indices
    nt_class           int8    [M]   config.nt_labels       (the table's consensus_nt)
    nt_confidence      float64 [M]   the table's predicted_nt_confidence

Nothing is signed, scaled, normalized, rounded, or initialized here. `forward(state)` is one linear
propagation step `W @ state` with rows = destination, provided as a convenience; any neuron model,
sign convention, or normalization is a downstream modeling choice.
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import PreTrainedModel

from .configuration_malecns import MaleCNSConfig


class MaleCNSConnectome(PreTrainedModel):
    config_class = MaleCNSConfig
    base_model_prefix = "malecns"
    supports_gradient_checkpointing = False

    def __init__(self, config: MaleCNSConfig):
        super().__init__(config)
        N, E, M = config.num_neurons, config.num_edges, config.num_nt_rows
        self.register_buffer("neuron_id", torch.zeros(N, dtype=torch.int64))
        self.register_buffer("edge_src", torch.zeros(E, dtype=torch.int32))
        self.register_buffer("edge_dst", torch.zeros(E, dtype=torch.int32))
        self.register_buffer("synapse_count", torch.zeros(E, dtype=torch.int32))
        self.register_buffer("neuron_status", torch.zeros(N, dtype=torch.int8))
        self.register_buffer("neuron_superclass", torch.zeros(N, dtype=torch.int8))
        self.register_buffer("nt_neuron_index", torch.zeros(M, dtype=torch.int32))
        self.register_buffer("nt_class", torch.zeros(M, dtype=torch.int8))
        self.register_buffer("nt_confidence", torch.zeros(M, dtype=torch.float64))
        self.post_init()

    def _init_weights(self, module):  # nothing is learned
        pass

    # ---- accessors -----------------------------------------------------------------------------
    @property
    def num_neurons(self) -> int:
        return self.config.num_neurons

    def in_degree(self) -> torch.Tensor:
        return torch.bincount(self.edge_dst.long(), minlength=self.num_neurons)

    def out_degree(self) -> torch.Tensor:
        return torch.bincount(self.edge_src.long(), minlength=self.num_neurons)

    def _label_mask(self, tensor: torch.Tensor, labels: list[str], names) -> torch.Tensor:
        idx = torch.tensor([labels.index(n) for n in names], dtype=tensor.dtype)
        return torch.isin(tensor, idx)

    def status_mask(self, *names: str) -> torch.Tensor:
        return self._label_mask(self.neuron_status, self.config.status_labels, names)

    def superclass_mask(self, *names: str) -> torch.Tensor:
        return self._label_mask(self.neuron_superclass, self.config.superclass_labels, names)

    def subset_mask(self, name: str) -> torch.Tensor:
        """Named subsets from config.subsets (lists of release superclasses; 'all' = every neuron)."""
        spec = self.config.subsets[name]
        return torch.ones(self.num_neurons, dtype=torch.bool) if spec == "all" else self.superclass_mask(*spec)

    def neuron_nt(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-neuron (nt_class, nt_confidence) scattered onto the N neurons; neurons without a
        prediction row get class index of 'unclear' and confidence 0."""
        cls = torch.full((self.num_neurons,), self.config.nt_labels.index("unclear"), dtype=torch.int8)
        conf = torch.zeros(self.num_neurons, dtype=torch.float64)
        i = self.nt_neuron_index.long()
        cls[i], conf[i] = self.nt_class, self.nt_confidence
        return cls, conf

    # ---- the graph -----------------------------------------------------------------------------
    def _restrict(self, nodes: Optional[torch.Tensor], min_synapses: int):
        src, dst, cnt = self.edge_src.long(), self.edge_dst.long(), self.synapse_count
        keep = cnt >= min_synapses
        n = self.num_neurons
        if nodes is not None:
            mask = nodes if nodes.dtype == torch.bool else torch.zeros(n, dtype=torch.bool).index_fill_(0, nodes.long(), True)
            new = torch.full((n,), -1, dtype=torch.long)
            new[mask] = torch.arange(int(mask.sum()))
            keep &= mask[src] & mask[dst]
            src, dst, n = new[src], new[dst], int(mask.sum())
            ids = self.neuron_id[mask]
        else:
            ids = self.neuron_id
        return src[keep], dst[keep], cnt[keep], n, ids

    def subgraph(self, nodes: torch.Tensor, min_synapses: int = 1) -> dict:
        """Induced subgraph as plain tensors: edge_src/edge_dst (relabelled 0..k-1), synapse_count, neuron_id."""
        src, dst, cnt, n, ids = self._restrict(nodes, min_synapses)
        return {"edge_src": src.int(), "edge_dst": dst.int(), "synapse_count": cnt, "neuron_id": ids}

    def sparse_weight(self, nodes: Optional[torch.Tensor] = None, min_synapses: int = 1,
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Sparse COO [n, n] of raw synapse counts, rows = destination, columns = source."""
        src, dst, cnt, n, _ = self._restrict(nodes, min_synapses)
        return torch.sparse_coo_tensor(torch.stack([dst, src]), cnt.to(dtype), (n, n)).coalesce()

    def forward(self, state: torch.Tensor, min_synapses: int = 1) -> torch.Tensor:
        """One propagation step: incoming[b, i] = sum_j synapse_count_ij * state[b, j]. state [B, N] -> [B, N]."""
        return torch.sparse.mm(self.sparse_weight(min_synapses=min_synapses), state.float().T).T
