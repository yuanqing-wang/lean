import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple

class HamiltonianMonteCarlo(NamedTuple):
    """Hamiltonian Monte Carlo algorithm.

    Parameters
    ----------
    potential : Callable
        Energy function.

    kinetic : Callable
        Kinetic energy function.

    integrator : Callable
        Integrator function.

    steps: int
        Number of steps to take in the leapfrog integration.
    """
    potential: Callable
    kinetic: Callable
    step_size: float
    steps: int

    def step(self, position: jnp.ndarray, momentum: jnp.ndarray):
        """Run the Hamiltonian Monte Carlo algorithm.

        Parameters
        ----------
        position : jnp.ndarray
            Initial position.

        momentum : jnp.ndarray
            Initial momentum.
        """
        # compute force
        force = -jax.grad(self.potential)(position)

        # leapfrog integration
        # update momentum
        momentum = momentum - 0.5 * self.step_size * force

        # update position
        position = position + self.step_size * momentum

        # update force
        force = -jax.grad(self.potential)(position)

        # update momentum
        momentum = momentum - 0.5 * self.step_size * force

        return position, momentum
    
    def __call__(
            self,
            position: jnp.ndarray,
            momentum: jnp.ndarray,
    ):
        """Run the Hamiltonian Monte Carlo algorithm.

        Parameters
        ----------
        position : jnp.ndarray
            Initial position.

        momentum : jnp.ndarray
            Initial momentum.
        """
        # initialize state
        state = (position, momentum)

        # run leapfrog integration
        state = jax.lax.fori_loop(
            0, self.steps, lambda i, state: self.step(*state), state
        )

        # unpack
        position, momentum = state
        return position, momentum

        




