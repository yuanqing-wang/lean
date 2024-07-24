import jax
import jax.numpy as jnp

def test_velocity_condition():
    from lean.models import ConditionalVelocity
    model = ConditionalVelocity(hidden_features=6, depth=3)
    x = jax.random.normal(jax.random.PRNGKey(0), (10, 3))
    h = jax.random.normal(jax.random.PRNGKey(0), (10, 4))
    params = model.init(jax.random.PRNGKey(0), x, h, key=jax.random.PRNGKey(0))
    v = model.apply(params, x, h, key=jax.random.PRNGKey(0))