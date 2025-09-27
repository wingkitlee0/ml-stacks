from functools import cached_property
from typing import Optional

from pydantic import BaseModel
from ray.train.lightning import RayDDPStrategy, RayDeepSpeedStrategy, RayFSDPStrategy


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
