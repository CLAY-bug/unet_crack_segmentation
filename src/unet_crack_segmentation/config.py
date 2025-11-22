# 用dataclass把yaml转成强类型配置，训练代码更干净
from dataclasses import dataclass
from typing import List, Optional
import yaml

@dataclass
class DataConfig:
    root: str
    train_images_dir: str
    train_masks_dir: str
    val_images_dir: str
    val_masks_dir: str
    image_size: List[int]
    num_workers: int
    batch_size: int

@dataclass
class ModelConfig:
    name: str
    in_channels: int
    num_classes: int

@dataclass
class TrainConfig:
    epochs: int
    lr: float
    weight_decay: float
    resume: Optional[str]

@dataclass
class LossConfig:
    type: str

@dataclass
class LoggingConfig:
    log_dir: str
    save_dir: str
    save_interval: int
    val_interval: int

@dataclass
class Config:
    experiment_name: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    loss: LossConfig
    logging: LoggingConfig

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    return Config(
        experiment_name=cfg_dict["experiment_name"],
        data=DataConfig(**cfg_dict["data"]),
        model=ModelConfig(**cfg_dict["model"]),
        train=TrainConfig(**cfg_dict["train"]),
        loss=LossConfig(**cfg_dict["loss"]),
        logging=LoggingConfig(**cfg_dict["logging"]),
    )
