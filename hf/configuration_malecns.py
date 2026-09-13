"""MaleCNS v1.0 connectome as a Hugging Face artifact (loaded with trust_remote_code=True)."""
from transformers import PretrainedConfig


class MaleCNSConfig(PretrainedConfig):
    model_type = "malecns"

    def __init__(
        self,
        num_neurons: int = 0,
        num_edges: int = 0,
        num_nt_rows: int = 0,
        release: str = "MaleCNS v1.0",
        source_files: dict | None = None,
        status_labels: list[str] | None = None,
        superclass_labels: list[str] | None = None,
        nt_labels: list[str] | None = None,
        subsets: dict | None = None,
        stats: dict | None = None,
        **kwargs,
    ):
        self.num_neurons = num_neurons
        self.num_edges = num_edges
        self.num_nt_rows = num_nt_rows
        self.release = release
        self.source_files = source_files or {}
        self.status_labels = status_labels or []
        self.superclass_labels = superclass_labels or []
        self.nt_labels = nt_labels or []
        self.subsets = subsets or {}
        self.stats = stats or {}
        super().__init__(**kwargs)
