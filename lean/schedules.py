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
        scales = self.params['scales']
        phases = jnp.linspace(0, 1, len(scales))
        linear = time
        time = time - phases
        time = scales * time ** 2
        time = jnp.exp(-time).sum()
        time = jnp.sin(time / (2 * math.pi)) * time
        time = time + linear
        return time
    
    @classmethod
    def init(cls, key:jax.random.PRNGKey, steps: int) -> 'Schedule': 
        scales = jax.random.normal(key, (steps,))
        return cls({'scales': scales})

        







