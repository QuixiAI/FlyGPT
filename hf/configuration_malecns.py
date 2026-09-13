"""MaleCNS v1.0 connectome as a Hugging Face artifact (loaded with trust_remote_code=True)."""
from transformers import PretrainedConfig


class MaleCNSConfig(PretrainedConfig):
    model_type = "malecns"

    def __init__(
        self,
        num_neurons: int = 0,
        num_edges: int = 0,
        num_self_loops: int = 0,
        min_synapses: int = 1,
        release: str = "MaleCNS v1.0",
        source_files: dict | None = None,
        region_labels: list[str] | None = None,
        superclass_labels: list[str] | None = None,
        nt_labels: list[str] | None = None,
        subsets: dict | None = None,
        stats: dict | None = None,
        **kwargs,
    ):
        self.num_neurons = num_neurons
        self.num_edges = num_edges
        self.num_self_loops = num_self_loops
        self.min_synapses = min_synapses
        self.release = release
        self.source_files = source_files or {}
        self.region_labels = region_labels or []
        self.superclass_labels = superclass_labels or []
        self.nt_labels = nt_labels or []
        self.subsets = subsets or {}
        self.stats = stats or {}
        super().__init__(**kwargs)
