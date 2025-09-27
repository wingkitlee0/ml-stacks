import lightning as L

from .advanced import AdvancedModelConfig
from .custom import CustomModelConfig
from .dummy import DummyLightningModule, DummyModel, DummyModelConfig

__all__ = [
    "DummyModelConfig",
    "DummyLightningModule",
    "DummyModel",
    "AdvancedModelConfig",
    "CustomModelConfig",
]


def create_model(
    config: DummyModelConfig | AdvancedModelConfig | CustomModelConfig,
) -> L.LightningModule:
    match config:
        case DummyModelConfig():
            return DummyLightningModule(
                config.input_size,
                config.output_size,
                config.lr,
            )
        case AdvancedModelConfig():
            raise NotImplementedError("Advanced model config not implemented")
        case CustomModelConfig():
            raise NotImplementedError("Custom model config not implemented")
        case _:
            raise ValueError(f"Invalid model config: {config}")
