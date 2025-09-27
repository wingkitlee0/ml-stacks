from typing import Literal

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pydantic import BaseModel


class DummyModelConfig(BaseModel):
    """Base model configuration."""

    model_type: Literal["dummy"] = "dummy"
    input_size: int
    output_size: int
    lr: float


class DummyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.layer(x)


class DummyLightningModule(L.LightningModule):
    def __init__(self, input_size, output_size, lr):
        super().__init__()
        self.model = DummyModel(input_size, output_size)
        self.lr = lr
        self.save_hyperparameters()

    @property
    def name(self):
        return "DummyLightningModule"

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
