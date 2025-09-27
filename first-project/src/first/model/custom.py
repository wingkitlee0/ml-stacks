from typing import Literal

from pydantic import BaseModel


class CustomModelConfig(BaseModel):
    """Custom model configuration with custom parameters."""

    model_type: Literal["custom"] = "custom"
    custom_param1: str = "default"
    custom_param2: int = 42
