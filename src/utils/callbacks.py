"""Training callback utilities."""

from __future__ import annotations

from lightning.pytorch.callbacks import ModelCheckpoint


class UniqueStateModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint variant with a stable, configurable callback state key."""

    def __init__(self, state_key_suffix: str, **kwargs: object) -> None:
        self.state_key_suffix = state_key_suffix
        super().__init__(**kwargs)

    @property
    def state_key(self) -> str:
        return f"{super().state_key}-{self.state_key_suffix}"
