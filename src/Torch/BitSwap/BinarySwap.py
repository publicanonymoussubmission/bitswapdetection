import torch


class SimulateSwap(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def sample_bit_int32(self, x: torch.Tensor) -> torch.Tensor:
        x = (torch.rand(size=x.size()) * 31).to(x.device)
        y = torch.pow(
            torch.tensor(2, dtype=torch.int32, device=x.device),
            x.view(torch.int32),
        )
        return y.view(torch.int32)

    def get_sign(self, x: torch.Tensor) -> torch.Tensor:
        y = -2 * (torch.clip(input=x, min=0, max=1).view(torch.float32) - 0.5)
        return y.view(torch.int32)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        target_type = input.dtype
        bitcast_to_int32 = input.view(torch.int32)
        magnitude = self.sample_bit_int32(x=input)
        sign_mult = self.get_sign(x=bitcast_to_int32 & magnitude)
        bitcast_to_int32 = bitcast_to_int32 + sign_mult * magnitude
        bitcast_to_float = bitcast_to_int32.view(target_type)
        return bitcast_to_float
