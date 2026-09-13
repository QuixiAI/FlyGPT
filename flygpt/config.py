"""YAML -> dataclass config. One file fully describes a run."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


@dataclass
class GraphConfig:
    source: str = "synthetic"      # "synthetic" | "fly"
    n_neurons: int = 300
    edges_path: str = "data/fly/edges.parquet"
    neurons_path: str = "data/fly/neurons.parquet"
    synthetic_density: float = 0.05
    scramble: bool = False
    n_input: int = 32
    n_output: int = 64
    io_rule: str = "random"        # "random" | "high_degree"
    seed: int = 0


@dataclass
class ModelConfig:
    embed_dim: int = 32
    microsteps: int = 1
    leak: float = 1.0              # 1.0 = no residual; <1.0 blends old state
    init_scale: float = 0.1


@dataclass
class DataConfig:
    task: str = "shakespeare"      # "shakespeare" | "delayed_copy" | "repeat"
    path: str = "data/shakespeare/input.txt"
    max_chars: int | None = None   # truncate the corpus (for overfit runs)
    seq_len: int = 64
    batch_size: int = 32
    val_fraction: float = 0.1


@dataclass
class TrainConfig:
    steps: int = 2000
    lr_recurrent: float = 3e-4
    lr_adapters: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    bf16: bool = True
    eval_every: int = 200
    sample_every: int = 200
    seed: int = 0


@dataclass
class Config:
    name: str = "run"
    model_type: str = "fly"        # "fly" | "rnn" | "gpt"
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            name=d.get("name", "run"),
            model_type=d.get("model_type", "fly"),
            graph=GraphConfig(**d.get("graph", {})),
            model=ModelConfig(**d.get("model", {})),
            data=DataConfig(**d.get("data", {})),
            train=TrainConfig(**d.get("train", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> dict:
        return asdict(self)
