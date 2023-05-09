from .BinarySwap import SimulateSwap
import torch


class ModuleWrapperBitSwap(torch.nn.Module):
    def __init__(
        self, module_to_wrap: torch.nn.Module, name: str, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.name = f"wrapped_{name}"
        self.module_to_wrap = module_to_wrap
        self.swap = SimulateSwap()
        self.bitswap_coefficient = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bitswap_coefficient:
            input = self.swap(input=input)
        return self.module_to_wrap.forward(input)
