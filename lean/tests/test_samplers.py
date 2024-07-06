def test_hmc():
    import jax
    import jax.numpy as jnp
    position = jax.random.normal(jax.random.PRNGKey(0), (100,))
    momentum = jax.random.normal(jax.random.PRNGKey(0), (100,))
    potential = lambda x: 0.5 * jnp.sum(x ** 2)
    kinetic = lambda x: 0.5 * jnp.sum(x ** 2)
    from lean.samplers import HamiltonianMonteCarlo
    sampler = HamiltonianMonteCarlo(potential, kinetic, 0.1, 10)
    position, momentum = sampler(position, momentum)
    
