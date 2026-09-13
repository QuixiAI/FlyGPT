"""FlyGPT configuration for Hugging Face `transformers` (loaded with trust_remote_code=True)."""
from transformers import PretrainedConfig


class FlyGPTConfig(PretrainedConfig):
    model_type = "flygpt"

    def __init__(
        self,
        vocab_size: int = 65,
        num_neurons: int = 0,
        num_edges: int = 0,
        embedding_dim: int = 32,
        num_input_neurons: int = 256,
        num_output_neurons: int = 512,
        microsteps: int = 2,
        activation: str = "tanh",
        learned_leak: bool = True,
        leak_init: float = 0.5,
        degree_normalization: bool = True,
        init_scale: float = 0.1,
        graph: dict | None = None,
        training_state: dict | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.num_neurons = num_neurons
        self.num_edges = num_edges
        self.embedding_dim = embedding_dim
        self.num_input_neurons = num_input_neurons
        self.num_output_neurons = num_output_neurons
        self.microsteps = microsteps
        self.activation = activation
        self.learned_leak = learned_leak
        self.leak_init = leak_init
        self.degree_normalization = degree_normalization
        self.init_scale = init_scale
        self.graph = graph or {}                    # source, hash, region filter, stats, provenance
        self.training_state = training_state or {}  # "init" or the run that produced the values
        super().__init__(**kwargs)
