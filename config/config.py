from dataclasses import dataclass

@dataclass
class SEEDStoryConfig:
    image_dim: int = 4096
    query_dim: int = 1024
    num_queries: int = 64
    max_length: int = 2048
    sink_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout_rate: float = 0.1
    layer_scale_init_value: float = 1e-5
    groups: int = 32
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    image_size: int = 224