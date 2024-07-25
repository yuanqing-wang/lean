def test_hmc():
    import jax
    import jax.numpy as jnp
    position = jax.random.normal(jax.random.PRNGKey(0), (100,))
    momentum = jax.random.normal(jax.random.PRNGKey(0), (100,))
    potential = lambda x, time: 0.5 * jnp.sum(x ** 2)
    from lean.samplers import HamiltonianMonteCarlo
    sampler = HamiltonianMonteCarlo(potential, 0.1)
    position, momentum = sampler(position, momentum)

def test_constant_potential_reverse():
    import jax
    import jax.numpy as jnp
    position0 = jax.random.normal(jax.random.PRNGKey(0), (100,))
    momentum0 = jax.random.normal(jax.random.PRNGKey(0), (100,))
    potential = lambda x, time: 0.5 * jnp.sum(x ** 2)
    from lean.samplers import HamiltonianMonteCarlo
    sampler = HamiltonianMonteCarlo(potential, 0.1)
    position1, momentum1 = sampler(position0, momentum0)

    position2, momentum2 = sampler.reverse(position1, momentum1)
    assert jnp.allclose(position0, position2, atol=1e-1)
    assert jnp.allclose(momentum0, momentum2, atol=1e-1)

def test_time_dependent_potential_reverse():
    import jax
    import jax.numpy as jnp
    position0 = jax.random.normal(jax.random.PRNGKey(0), (100,))
    momentum0 = jax.random.normal(jax.random.PRNGKey(0), (100,))
    potential = lambda x, time: 0.5 * jnp.sum(time * x ** 2)
    from lean.samplers import HamiltonianMonteCarlo
    sampler = HamiltonianMonteCarlo(potential, 0.1)
    position1, momentum1 = sampler(position0, momentum0)

    position2, momentum2 = sampler.reverse(position1, momentum1)
    assert jnp.allclose(position0, position2)
    assert jnp.allclose(momentum0, momentum2)



