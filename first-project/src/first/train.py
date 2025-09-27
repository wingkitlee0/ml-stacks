import os
from datetime import datetime
from typing import Optional, Union

import yaml

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

from functools import cached_property
from typing import Annotated

import lightning as L
import ray
import torch
from pydantic import BaseModel, Discriminator, Field
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayDeepSpeedStrategy,
    RayFSDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
)
from ray.train.torch import TorchTrainer

from first.model.dummy import (
    AdvancedModelConfig,
    CustomModelConfig,
    DummyLightningModule,
    DummyModelConfig,
)


def get_datetime_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# Define a discriminated union for model configs
ModelConfig = Annotated[
    Union[DummyModelConfig, AdvancedModelConfig, CustomModelConfig],
    Discriminator("model_type"),
]


def create_model(config: ModelConfig) -> L.LightningModule:
    return DummyLightningModule(
        config.input_size,
        config.output_size,
        config.lr,
    )


class TrainerConfig(BaseModel):
    accelerator: str = "cpu"
    strategy: Optional[str] = None
    devices: int = 1
    max_epochs: int = 10

    @cached_property
    def get_strategy(self):
        if self.strategy is None:
            return None
        elif self.strategy == "ddp":
            return RayDDPStrategy()
        elif self.strategy == "fsdp":
            return RayFSDPStrategy()
        elif self.strategy == "deepspeed":
            return RayDeepSpeedStrategy()
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")


class TrainLoopConfig(BaseModel):
    """Configuration for the training loop."""

    model: ModelConfig
    trainer: TrainerConfig


def load_config_dict_from_yaml(yaml_path: str) -> dict:
    """Load training configuration as dictionary from a YAML file."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def create_dataset(name: str) -> ray.data.Dataset:
    return ray.data.from_items(
        [{"x": torch.randn(10), "y": torch.randn(10)} for _ in range(100)]
    )


def train_func(config_dict):
    print(config_dict)
    # Parse the config into our Pydantic model
    train_config = TrainLoopConfig(**config_dict)

    model = create_model(train_config.model)

    trainer_config: TrainerConfig = train_config.trainer
    trainer = L.Trainer(
        accelerator=trainer_config.accelerator,
        strategy=trainer_config.get_strategy,
        devices=trainer_config.devices,
        logger=True,
        callbacks=[
            RayTrainReportCallback(),
        ],
        plugins=[
            RayLightningEnvironment(),
        ],
        enable_progress_bar=False,
        enable_checkpointing=False,
        max_epochs=trainer_config.max_epochs,
    )

    train_ds = create_dataset("train")
    val_ds = create_dataset("val")

    train_dataloader = train_ds.iter_torch_batches(batch_size=10)
    val_dataloader = val_ds.iter_torch_batches(batch_size=10)

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


def load_and_validate_config_dict(config_path: Optional[str] = None):
    if config_path is not None:
        print(f"Loading configuration from {config_path}")
        config_dict = load_config_dict_from_yaml(config_path)

        TrainLoopConfig.model_validate(config_dict)
        print("✅ Configuration validation passed")

        return config_dict

    print("Using default configuration")
    train_config = TrainLoopConfig(
        model=DummyModelConfig(
            input_size=10,
            output_size=10,
            lr=0.001,
        ),
        trainer=TrainerConfig(
            accelerator="cpu",
            strategy="ddp",
            devices=2,
            max_epochs=10,
        ),
    )
    # Convert to dict for TorchTrainer
    config_dict = train_config.model_dump()
    return config_dict


def main(config_path: Optional[str] = None):
    datetime_str = get_datetime_str()

    config_dict = load_and_validate_config_dict(config_path)

    trainer = TorchTrainer(
        train_func,
        # Use the validated dictionary
        train_loop_config=config_dict,
        scaling_config=ScalingConfig(num_workers=2),
        run_config=RunConfig(
            name=f"first_{datetime_str}",
            storage_path="/tmp/ray_checkpoints/",
            checkpoint_config=CheckpointConfig(),
        ),
    )

    result = trainer.fit()

    print(result)


if __name__ == "__main__":
    main()
