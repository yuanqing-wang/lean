import abc
from typing import NamedTuple, Mapping
import math
import jax
from jax import numpy as jnp

class Schedule(NamedTuple):
    params: Mapping

    @abc.abstractmethod
    def __call__(self, time: float) -> float:
        raise NotImplementedError
    
    @classmethod
    @abc.abstractmethod
    def init(cls, **kwargs) -> 'Schedule':
        raise NotImplementedError

class SinRBFSchedule(Schedule):
    params: Mapping

    def __call__(self, time: float) -> float:
        gamma = self.params['gamma']
        mu = jnp.linspace(0, 1, len(gamma))
        linear = time
        time = time - mu
        time = jax.nn.softplus(gamma) * (time ** 2)
        time = jnp.exp(-time)
        time = (time * self.params['coefficient']).sum()
        time = jnp.sin(linear * (math.pi)) * time
        if self.base == "linear":
            time = time + linear
        elif self.base == "ones":
            time = time + 1
        elif self.base == "zeros":
            time = time
        return time
    
    @classmethod
    def init(cls, key:jax.random.PRNGKey, steps: int, base: str="linear") -> 'Schedule': 
        gamma = (1 / steps) * jnp.ones(steps)
        coefficient = jax.random.normal(key, (steps,)) * 1e-1
        initialized = cls({'gamma': gamma, 'coefficient': coefficient})
        initialized.__class__.base = base
        return initialized
        

