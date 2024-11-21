import jax
import jax.numpy as jnp

@jax.custom_vjp
def grad_reverse(x):
    return x

def grad_reverse_fwd(x):
    return x, None

def grad_reverse_bwd(res, g):
    return -g

grad_reverse.defvjp(grad_reverse_fwd, grad_reverse_bwd)

# Gumbel-Softmax sampling
def sample_gumbel(key, shape, dtype=jnp.float32, eps=1e-20):
    u = jax.random.uniform(key, shape=shape, minval=0.0, maxval=1.0, dtype=dtype)
    return -jnp.log(-jnp.log(u + eps) + eps)

def gumbel_softmax_sample(logits, temperature, key):
    y = logits + sample_gumbel(key, logits.shape)
    return jax.nn.softmax(y / temperature, axis=-1)

def gumbel_softmax(logits, temperature, key, hard=False):
    y = gumbel_softmax_sample(logits, temperature, key)
    if hard:
        y_hard = jax.nn.one_hot(jnp.argmax(y, axis=-1), logits.shape[-1])
        y = y_hard - jax.lax.stop_gradient(y) + y
    return y
