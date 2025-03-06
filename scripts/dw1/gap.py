import jax
from jax import numpy as jnp
from typing import NamedTuple
import optax

class Schedule(NamedTuple):
    coefficients: jnp.ndarray
    
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.polyval(self.coefficients, t).sum()

def run():
    schedule = Schedule(jnp.ones(10))
    optimizer = optax.adam(1e-3)
    optimizer_state = optimizer.init(schedule)
    
    def loss(schedule, key):
        t = jax.random.uniform(key, (100,))
        dydt = jax.grad(schedule)(t)
        gap = dydt.sum() * 0.01 + schedule(0.0) - schedule(1.0)
        return gap
    
    for idx in range(100000):
        _loss, grad = jax.value_and_grad(loss)(schedule, jax.random.PRNGKey(idx))
        updates, optimizer_state = optimizer.update(grad, optimizer_state)
        schedule = optax.apply_updates(schedule, updates)
        print(_loss)
        
    

    
if __name__ == "__main__":
    run()