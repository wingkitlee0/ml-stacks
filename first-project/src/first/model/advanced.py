from typing import Literal

from pydantic import BaseModel


class AdvancedModelConfig(BaseModel):
    """Advanced model configuration with additional parameters."""

    model_type: Literal["advanced"] = "advanced"
    hidden_size: int = 64
    dropout_rate: float = 0.1
    activation: str = "relu"
