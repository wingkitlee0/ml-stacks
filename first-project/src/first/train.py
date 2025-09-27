import argparse
import logging
import os
from datetime import datetime
from typing import Optional

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"  # noqa: E402


import lightning as L
import ray
import torch
from lightning.pytorch.profilers import PyTorchProfiler
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import (
    RayLightningEnvironment,
    RayTrainReportCallback,
)
from ray.train.torch import TorchTrainer

from first.configs import TrainLoopConfig, load_and_validate_config_dict
from first.data import create_train_val_datasets
from first.model import create_model


def get_datetime_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_dataset(name: str) -> ray.data.Dataset:
    """Create a dataset for the given name."""
    num_rows = 100 if name == "train" else 10

    return ray.data.from_items(
        [{"x": torch.randn(10), "y": torch.randn(10)} for _ in range(num_rows)]
    )


def train_func(config_dict):
    train_config = TrainLoopConfig.model_validate(config_dict)

    model = create_model(train_config.model)

    trainer_config = train_config.trainer
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
        profiler=PyTorchProfiler(),
        enable_progress_bar=False,
        enable_checkpointing=False,
        max_epochs=trainer_config.max_epochs,
    )

    train_ds, val_ds = create_train_val_datasets()

    train_dataloader = train_ds.iter_torch_batches(batch_size=10)
    val_dataloader = val_ds.iter_torch_batches(batch_size=10)

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


def main(config_path: Optional[str] = None, datetime_str: Optional[str] = None):
    datetime_str = datetime_str or get_datetime_str()

    config_dict = load_and_validate_config_dict(config_path)

    print(config_dict)
    print(TrainLoopConfig.model_validate(config_dict))

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
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
    parser = argparse.ArgumentParser(description="Run training with YAML configuration")
    parser.add_argument(
        "--config", "-c", type=str, help="Path to YAML configuration file", default=None
    )
    parser.add_argument(
        "--datetime",
        "-dt",
        type=str,
        help="Datetime string",
        default=get_datetime_str(),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    main(args.config, args.datetime)
