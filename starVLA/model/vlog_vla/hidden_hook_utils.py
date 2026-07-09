from __future__ import annotations


class HiddenTokenHook:
    """Context manager capturing module forward-pre-hook inputs."""

    def __init__(self, module) -> None:
        self.module = module
        self.last_inputs = None
        self.handle = None

    def __enter__(self) -> "HiddenTokenHook":
        self.handle = self.module.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()

    def _hook(self, module, inputs) -> None:
        self.last_inputs = inputs

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
