"""FlyGPT: a character-level language model whose recurrent core is a real subgraph of the
fruit-fly connectome (MaleCNS v1.0). Hugging Face `transformers` implementation; self-contained.

Dynamics (one scalar state per neuron, README §7 of the FlyGPT spec):

    proposal_i = tanh( sum_j W_ij h_j / sqrt(in_degree_i) + external_input_i + bias_i )
    h_i_new    = (1 - leak_i) * h_i + leak_i * proposal_i

The connectome is stored in `model.safetensors` as integer tensors (`graph.*`); only the learned
per-edge values and the adapters are floating point (bf16 on disk). The sparse recurrent matmul is
rebuilt in fp32 at runtime (rows = destination, columns = source).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.utils import ModelOutput

from .configuration_flygpt import FlyGPTConfig


@dataclass
class FlyGPTOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    state: Optional[torch.FloatTensor] = None   # [B, N] neuron states after the last character


class FlyGraph(nn.Module):
    """The anatomy. Integer buffers only; never trained."""

    def __init__(self, num_neurons: int, num_edges: int, num_input: int, num_output: int):
        super().__init__()
        self.register_buffer("edge_index", torch.zeros(2, num_edges, dtype=torch.int32))   # [source, destination]
        self.register_buffer("synapse_count", torch.zeros(num_edges, dtype=torch.int32))   # MaleCNS synaptic contacts
        self.register_buffer("node_id", torch.zeros(num_neurons, dtype=torch.int64))       # MaleCNS body ids
        self.register_buffer("input_nodes", torch.zeros(num_input, dtype=torch.int64))
        self.register_buffer("output_nodes", torch.zeros(num_output, dtype=torch.int64))


class FlyRecurrentCore(nn.Module):
    """The learned state: one value per real edge, plus per-neuron bias and leak."""

    def __init__(self, num_neurons: int, num_edges: int, leak_init: float, learned_leak: bool):
        super().__init__()
        self.edge_values = nn.Parameter(torch.zeros(num_edges))
        self.bias = nn.Parameter(torch.zeros(num_neurons))
        self.raw_leak = nn.Parameter(torch.full((num_neurons,), math.log(leak_init / (1 - leak_init))),
                                     requires_grad=learned_leak)


class FlyGPTPreTrainedModel(PreTrainedModel):
    config_class = FlyGPTConfig
    base_model_prefix = "flygpt"
    _is_stateful = True
    _supports_cache_class = False
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, FlyRecurrentCore):
            nn.init.normal_(module.edge_values, std=self.config.init_scale)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=1.0)


class FlyGPTForCausalLM(FlyGPTPreTrainedModel, GenerationMixin):
    def __init__(self, config: FlyGPTConfig):
        super().__init__(config)
        c = config
        self.graph = FlyGraph(c.num_neurons, c.num_edges, c.num_input_neurons, c.num_output_neurons)
        self.recurrent = FlyRecurrentCore(c.num_neurons, c.num_edges, c.leak_init, c.learned_leak)
        self.embed = nn.Embedding(c.vocab_size, c.embedding_dim)
        self.input_proj = nn.Linear(c.embedding_dim, c.num_input_neurons)
        self.lm_head = nn.Linear(c.num_output_neurons, c.vocab_size)
        self.post_init()

    # ---- sparse recurrent matrix -------------------------------------------------------------
    @property
    def num_neurons(self) -> int:
        return self.config.num_neurons

    def edge_scale(self) -> torch.Tensor:
        """1/sqrt(in_degree) per edge (degree normalization), or ones."""
        dst = self.graph.edge_index[1].long()
        if not self.config.degree_normalization:
            return torch.ones_like(dst, dtype=torch.float32)
        in_deg = torch.bincount(dst, minlength=self.num_neurons).clamp(min=1).float()
        return 1.0 / in_deg[dst].sqrt()

    def sparse_weight(self) -> torch.Tensor:
        src, dst = self.graph.edge_index[0].long(), self.graph.edge_index[1].long()
        values = self.recurrent.edge_values.float() * self.edge_scale()
        return torch.sparse_coo_tensor(torch.stack([dst, src]), values, (self.num_neurons, self.num_neurons))

    def dense_weight(self) -> torch.Tensor:
        """Convenience for analysis; [N, N] with rows = destination. Never used in the forward pass."""
        return self.sparse_weight().to_dense()

    @property
    def leak(self) -> torch.Tensor:
        return torch.sigmoid(self.recurrent.raw_leak.float())

    # ---- dynamics ------------------------------------------------------------------------------
    def init_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.num_neurons, device=device or self.recurrent.edge_values.device)

    def drive(self, x: torch.Tensor) -> torch.Tensor:
        d = torch.zeros(x.shape[0], self.num_neurons, device=x.device, dtype=torch.float32)
        d[:, self.graph.input_nodes] = self.input_proj(self.embed(x)).float()
        return d

    def step(self, state: torch.Tensor, x: torch.Tensor, W: torch.Tensor | None = None) -> torch.Tensor:
        W = self.sparse_weight() if W is None else W
        drive, leak, bias = self.drive(x), self.leak, self.recurrent.bias.float()
        for _ in range(self.config.microsteps):
            incoming = torch.sparse.mm(W, state.float().T).T
            proposal = torch.tanh(incoming + drive + bias)
            state = (1 - leak) * state + leak * proposal
        return state

    def logits_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.lm_head(state[:, self.graph.output_nodes].to(self.lm_head.weight.dtype)).float()

    def forward(self, input_ids: torch.LongTensor, state: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None, use_cache: Optional[bool] = None,
                return_dict: Optional[bool] = None, **kwargs) -> FlyGPTOutput:
        B, T = input_ids.shape
        state = self.init_state(B, input_ids.device) if state is None else state
        W = self.sparse_weight()
        outs = []
        for t in range(T):
            state = self.step(state, input_ids[:, t], W)
            outs.append(self.logits_from_state(state))
        logits = torch.stack(outs, 1)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
        return FlyGPTOutput(loss=loss, logits=logits, state=state)

    # ---- generation: carry the neuron state instead of a KV cache ------------------------------
    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False  # stateful recurrent model: no KV cache, the neuron state is carried in `state`

    def prepare_inputs_for_generation(self, input_ids, state=None, **kwargs):
        if state is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "state": state}

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, **kwargs):
        model_kwargs["state"] = outputs.state
        return model_kwargs
