import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple, Optional
from functools import partial

class HamiltonianMonteCarlo(NamedTuple):
    """Hamiltonian Monte Carlo algorithm.

    Parameters
    ----------
    potential : Callable
        Energy function.

    integrator : Callable
        Integrator function.

    steps: int
        Number of steps to take in the leapfrog integration.
    """
    potential: Callable
    step_size: float
    steps: int

    def step(
            self, 
            position: jnp.ndarray, 
            momentum: jnp.ndarray,
            step: Optional[int] = None,
            reverse: bool = False
        ):
        """Run the Hamiltonian Monte Carlo algorithm.

        Parameters
        ----------
        position : jnp.ndarray
            Initial position.

        momentum : jnp.ndarray
            Initial momentum.
        """
        # compute time
        if step is not None:
            time = (step / self.steps)
        else:
            time = 0.0

        step_size = self.step_size * (-1 if reverse else 1)

        # compose potential energy
        potential = partial(self.potential, time=time)

        # compute force
        force = -jax.grad(potential)(position)

        # leapfrog integration
        # update momentum
        momentum = momentum + 0.5 * step_size * force

        # update position
        position = position + step_size * momentum

        # update force
        force = -jax.grad(potential)(position)

        # update momentum
        momentum = momentum + 0.5 * step_size * force

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
            0, self.steps, lambda i, state: self.step(*state, step=i), state
        )

        # unpack
        position, momentum = state
        return position, momentum
    
    def reverse(
            self,
            position: jnp.ndarray,
            momentum: jnp.ndarray,
    ):
        """Run the Hamiltonian Monte Carlo algorithm in reverse.

        Parameters
        ----------
        position : jnp.ndarray
            Initial position.

        momentum : jnp.ndarray
            Initial momentum.

        schedule : jnp.ndarray
            Schedule for the step size.
        """
        # initialize state
        state = (position, momentum)

        # run leapfrog integration
        state = jax.lax.fori_loop(
            0, self.steps,
            lambda i, state: self.step(*state, step=self.steps-i-1, reverse=True), 
            state,
        )

        # unpack
        position, momentum = state
        return position, momentum



