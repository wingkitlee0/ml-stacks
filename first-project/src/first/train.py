import os
from datetime import datetime

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import lightning as L
import ray
import torch
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayDeepSpeedStrategy,
    RayFSDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
)
from ray.train.torch import TorchTrainer

from first.model.dummy import DummyLightningModule, DummyModelConfig


def get_datetime_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_model(config: DummyModelConfig) -> L.LightningModule:
    return DummyLightningModule(
        config.input_size,
        config.output_size,
        config.lr,
    )


@dataclass
class TrainerConfig:
    accelerator: str = "cpu"
    strategy: Optional[str] = None
    devices: int = 1

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


def create_dataset(name: str) -> ray.data.Dataset:
    return ray.data.from_items(
        [{"x": torch.randn(10), "y": torch.randn(10)} for _ in range(100)]
    )


def train_func(config):
    print(config)
    model = create_model(config["model"])

    trainer_config: TrainerConfig = config["trainer"]
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


def main():
    datetime_str = get_datetime_str()

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "model": DummyModelConfig(
                input_size=10,
                output_size=10,
                lr=0.001,
            ),
            "trainer": TrainerConfig(
                accelerator="cpu",
                strategy="ddp",
                devices=2,
            ),
        },
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
