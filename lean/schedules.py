import abc
from typing import NamedTuple, Mapping
import math
import torch


class SinRBFSchedule(torch.nn.Module):
    def __init__(
        self,
        steps: int,
        base: str="linear",
    ):
        super().__init__()
        gamma = steps * torch.ones(steps)
        coefficient = torch.randn(steps) * 1e-2
        self.gamma = torch.nn.Parameter(gamma)
        self.coefficient = torch.nn.Parameter(coefficient)
        self.base = base
        
    def forward(
        self,
        time: float,
    ):
        mu = torch.linspace(0, 1, len(self.gamma))
        linear = time
        time = time - mu
        time = torch.nn.functional.softplus(self.gamma) * (time ** 2)
        time = torch.exp(-time)
        time = (time * self.coefficient).sum()
        time = torch.sin(linear * (math.pi)) * time
        if self.base == "linear":
            time = time + linear
        elif self.base == "ones":
            time = time + 1
        elif self.base == "zeros":
            time = time
        return time

class MeanFieldSinRBFSchedule(torch.nn.Module):
    def __init__(
        self,
        steps: int,
        base: str="linear",
        log_sigma: float=-5,
    ):
        super().__init__()
        gamma = steps * torch.ones(steps)
        coefficient = torch.randn(steps) * 1e-3
        self.gamma_mu = torch.nn.Parameter(gamma)
        self.coefficient_mu = torch.nn.Parameter(coefficient)
        self.gamma_log_sigma = torch.nn.Parameter(torch.ones_like(gamma) * log_sigma)
        # self.gamma_log_sigma = torch.ones_like(gamma) * log_sigma
        self.coefficient_log_sigma = torch.nn.Parameter(torch.ones_like(coefficient) * log_sigma)
        # self.coefficient_log_sigma = torch.ones_like(coefficient) * log_sigma
        self.base = base

    @property
    def gamma(self):
        return torch.distributions.Normal(
            self.gamma_mu,
            self.gamma_log_sigma.exp(),
        )
        
    
    @property
    def coefficient(self):
        return torch.distributions.Normal(
            self.coefficient_mu,
            self.coefficient_log_sigma.exp(),
        )
    
    def forward(
        self,
        time: float,
        gamma: torch.Tensor,
        coefficient: torch.Tensor,
    ):
        mu = torch.linspace(0, 1, len(gamma))
        linear = time
        time = time - mu
        time = torch.nn.functional.softplus(gamma) * (time ** 2)
        time = torch.exp(-time)
        time = (time * coefficient).sum()
        time = torch.sin(linear * (math.pi)) * time
        if self.base == "linear":
            time = time + linear
        elif self.base == "ones":
            time = time + 1
        elif self.base == "zeros":
            time = time
        return time