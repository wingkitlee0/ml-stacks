from typing import Annotated, Optional, Union

import yaml
from pydantic import BaseModel, Discriminator, Tag

from first.model.advanced import AdvancedModelConfig
from first.model.custom import CustomModelConfig
from first.model.dummy import DummyModelConfig
from first.utils.configs import TrainerConfig


def load_config_dict_from_yaml(yaml_path: str) -> dict:
    """Load training configuration as dictionary from a YAML file."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def get_discriminator_value(v: object) -> str | None:
    if v is None:
        return None
    if isinstance(v, dict):
        return v.get("model_type")
    return getattr(v, "model_type", None)


class TrainLoopConfig(BaseModel):
    """Configuration for the training loop."""

    model: Annotated[
        Union[
            Annotated[DummyModelConfig, Tag("dummy")],
            Annotated[AdvancedModelConfig, Tag("advanced")],
            Annotated[CustomModelConfig, Tag("custom")],
        ],
        Discriminator(get_discriminator_value),
    ]
    trainer: TrainerConfig


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
