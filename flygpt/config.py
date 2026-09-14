"""The launch config schema (plan.md §13). One yaml, no grid.

Everything not expressible here is a follow-up experiment, not a launch option.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    name: str = "tiny_shakespeare"
    path: str = "data/shakespeare/input.txt"
    split: float = 0.90
    max_chars: int | None = None       # dev-only: truncate corpus for the 1k overfit gate


@dataclass
class GraphConfig:
    source: str = "malecns-v1.0"
    edges_path: str = "data/fly/edges.parquet"
    neurons_path: str = "data/fly/neurons.parquet"
    region_filter: str = "central_brain"
    target_neurons: int = 5000
    min_synapses: int = 3
    selector: str = "directed_core"
    retain_largest_scc: bool = True


@dataclass
class InterfaceConfig:
    input_nodes: int = 256
    input_rule: str = "top_out_degree"
    output_nodes: int = 512
    output_rule: str = "top_in_degree"


@dataclass
class ModelConfig:
    state_dim_per_neuron: int = 1
    activation: str = "tanh"
    learned_leak: bool = True
    leak_init: float = 0.5
    degree_normalization: bool = True
    embed_dim: int = 32
    init_scale: float = 0.1
    backend: str = "cuda"              # "cuda": fused kernels (flygpt/kernels.py) when available; "torch": sparse COO path


@dataclass
class SequenceConfig:
    context: int = 64
    microsteps: int = 2


@dataclass
class TrainingConfig:
    optimizer: str = "adamw"
    recurrent_lr: float = 3e-4
    adapter_lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    sparse_precision: str = "fp32"
    adapter_precision: str = "bf16"
    batch_size: int = 32
    steps: int = 20000
    eval_every: int = 250
    eval_batches: int = 20


@dataclass
class ClaimRule:
    all_paired_diffs_same_sign: bool = True
    min_mean_diff_nats: float = 0.05


@dataclass
class EvaluationConfig:
    seeds_dev: list[int] = field(default_factory=lambda: [1, 2, 3])
    seeds_claim: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    paired_seeds: bool = True
    claim_rule: ClaimRule = field(default_factory=ClaimRule)
    prompts: list[str] = field(default_factory=lambda: ["ROMEO:", "KING:", "JULIET:", "First Citizen:"])
    generation_fractions: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    generation_temperature: float = 0.8
    generation_chars: int = 300


@dataclass
class GateConfig:
    """Path-length gate (plan.md §6). Thresholds are engineering choices; log them."""
    min_reachable_fraction: float = 0.95
    max_p90_chars: int = 3             # p90 I->O path must fit in this many characters at `microsteps`


@dataclass
class Config:
    project: str = "flygpt-v0"
    graph_name: str = "cb5k"           # directory under graphs/ where artifacts live
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    controls: list[str] = field(default_factory=lambda: ["real", "degree_preserving"])
    interface: InterfaceConfig = field(default_factory=InterfaceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    gate: GateConfig = field(default_factory=GateConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        ev = dict(d.get("evaluation", {}))
        if "claim_rule" in ev:
            ev["claim_rule"] = ClaimRule(**ev["claim_rule"])
        return cls(
            project=d.get("project", "flygpt-v0"),
            graph_name=d.get("graph_name", "cb5k"),
            dataset=DatasetConfig(**d.get("dataset", {})),
            graph=GraphConfig(**d.get("graph", {})),
            controls=list(d.get("controls", ["real", "degree_preserving"])),
            interface=InterfaceConfig(**d.get("interface", {})),
            model=ModelConfig(**d.get("model", {})),
            sequence=SequenceConfig(**d.get("sequence", {})),
            training=TrainingConfig(**d.get("training", {})),
            evaluation=EvaluationConfig(**ev),
            gate=GateConfig(**d.get("gate", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def graph_dir(self) -> Path:
        return Path("graphs") / self.graph_name
