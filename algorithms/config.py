import os
import uuid
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union


@dataclass
class TrainConfig:
    # wandb params
    project: str = "CORL"
    group: str = "Pogema-vector-obs"
    name: str = "DT"
    # model params
    embedding_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    seq_len: int = 20
    episode_len: int = 200
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    max_action: float = 1.0
    # training params
    env_name: str = "POMAF"
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 128
    update_steps: int = 100_000
    warmup_steps: int = 10_000
    reward_scale: float = 1
    num_workers: int = 1
    # evaluation params
    target_returns: Tuple[float, ...] = (1.0, 2.0)
    eval_episodes: int = 100
    eval_every: int = 5000
    # general params
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    train_seed: int = 10
    eval_seed: int = 42
    device: str = "cuda"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
