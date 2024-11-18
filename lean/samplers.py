from math import gamma
import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple, Optional
from functools import partial
    
class OverdampedLangevinDynamics(NamedTuple):
    """Overdamped Langevin Dynamics.

    Parameters
    ----------
    potential : Callable
        Energy function.

    integrator : Callable
        Integrator function.

    steps: int
        Number of steps to ta
        
    """
        
    potential: Callable
    unbiasing_potential: Callable
    step_size: float
    time: float = 1.0
    
    def step(
            self, 
            position: jnp.ndarray, 
            A: float,
            B: float,
            key: jax.random.PRNGKey,
            epsilon: float = 1e-2,
            time: float = 0.0,
    ):
        """Run the Hamiltonian Monte Carlo algorithm.

        Parameters
        ----------
        position : jnp.ndarray
            Initial position.

        momentum : jnp.ndarray
            Initial momentum.
        """
                
        # compose potential energy
        # potential = lambda x: self.potential(x, time=time).sum()
        # biasing_energy = self.unbiasing_potential(position, time=time).sum()
        energy = jax.vmap(self.potential, in_axes=(0, None))(position, time)

        # compute force
        force = -jax.grad(self.potential)(position, time=time)
        unbiasing_force = jax.grad(self.unbiasing_potential)(position, time=time)
        
        # sample noise
        eta = jax.random.normal(key, shape=position.shape)
        
        # update position
        position = position \
            + epsilon * force * self.step_size \
            + unbiasing_force * self.step_size \
            + jnp.sqrt(2 * epsilon) * eta * self.step_size
                        
        # update B                    
        B = B \
            + (1 / epsilon) * (unbiasing_force ** 2).sum(-1).sum(-1) * self.step_size\
            + jnp.sqrt(2 / epsilon) * self.step_size * (unbiasing_force * eta).sum(-1).sum(-1)
        
        A = -energy - B
        return position, A, B
            
    def __call__(
            self,
            position: jnp.ndarray,
            key: jax.random.PRNGKey,
    ):
        """Run the Hamiltonian Monte Carlo algorithm.

        Parameters
        ----------
        position : jnp.ndarray
            Initial position.

        momentum : jnp.ndarray
            Initial momentum.
        """        
        # split keys
        steps = int(self.time / self.step_size)
        keys = jax.random.split(key, steps)
        times = jnp.linspace(0, 1, steps)
        
        # initialize state
        state = (position, jnp.zeros(len(position)), jnp.zeros(len(position)))
        
        def step_fn(state, idx):
            state = self.step(*state, time=times[idx], key=keys[idx])
            return state, state
        
        _, states = jax.lax.scan(
            step_fn,
            state,
            jnp.arange(steps),
        )

        # unpack
        position, A, _ = states
        position = position.swapaxes(0, 1)
        A = A.swapaxes(0, 1)
        return position, A
    