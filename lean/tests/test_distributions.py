import jax
import jax.numpy as jnp
def test_radial_gaussian():
    from lean.distributions import RadialLogNormal

    # test_sample
    radial = RadialLogNormal(1.0, 1.0)
    key = jax.random.PRNGKey(2666)
    x = radial.sample(key, shape=(),)
    log_prob = radial.log_prob(x)

def test_normalizing_constant():
    x_space = jnp.linspace(-10, 10, 10)
    y_space = jnp.linspace(-10, 10, 10)
    z_space = jnp.linspace(-10, 10, 10)

    from lean.distributions import RadialLogNormal
    total_prob = 0.0
    radial = RadialLogNormal(1.0, 1.0)
    for x in x_space:
        for y in y_space:
            for z in z_space:
                prob = jnp.exp(radial.log_prob(jnp.array([x, y, z])))
                total_prob += prob
    print(total_prob)
    