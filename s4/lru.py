import jax
from jax import numpy as jnp
import numpy as np

import flax
from flax import linen as nn

def uniform_spectral_init(r_min=0., r_max=1., max_phase=6.28):
    def init(key, shape, dtype=jnp.float_):
        key1, key2 = jax.random.split(key)
        u1 = jax.random.uniform(key1, shape, dtype)
        u2 = jax.random.uniform(key2, shape, dtype)

        nu_log = jnp.log(-0.5*jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        theta_log = jnp.log(max_phase * u2)

        diag_lambda = jnp.exp(-jnp.exp(nu_log) + 1j*jnp.exp(theta_log))
        gamma_log = jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda)**2))

        return {"nu_log": nu_log, "theta_log": theta_log, "gamma_log": gamma_log}

    return init

def binary_operator_diag(element_i, element_j):
    a_i, bu_i = element_i
    a_j, bu_j = element_j

    return a_j * a_i, a_j * bu_i + bu_j


class LRU(nn.Module):
    N: int
    H: int
    r_max: float = 1.
    r_min: float = 0.
    max_phase: float = 6.28
    l_max: int = 512
    decode: bool = False

    lr = {
        "nu_log": 0.5,
        "theta_log": 0.5,
        "gamma_log": 0.5,
        "B_re": 0.5,
        "B_im": 0.5,
    }

    @nn.compact
    def __call__(self, input_sequence):
        p = self.param("diagonalised_A", uniform_spectral_init(r_min=self.r_min, r_max=self.r_max, max_phase=self.max_phase), (self.N,))

        nu_log, theta_log, gamma_log = p["nu_log"], p["theta_log"], p["gamma_log"]

        B_re = self.param(
            "B_re",
            nn.initializers.normal(stddev=1/np.sqrt(2*self.H)),
            (self.N, self.H)
        )
        B_im = self.param(
            "B_im",
            nn.initializers.normal(stddev=1/np.sqrt(2*self.H)),
            (self.N, self.H)
        )
        C_re = self.param(
            "C_re",
            nn.initializers.normal(stddev=1/np.sqrt(self.N)),
            (self.H, self.N)
        )
        C_im = self.param(
            "C_im",
            nn.initializers.normal(stddev=1/np.sqrt(self.N)),
            (self.H, self.N)
        )
        D = self.param(
            "D",
            nn.initializers.normal(),
            (self.H,)
        )

        Lambda = jnp.exp(-jnp.exp(nu_log) + 1j*jnp.exp(theta_log))
        B_norm = (B_re + 1j*B_im) * jnp.expand_dims(jnp.exp(theta_log), axis=-1)
        C = C_re + 1j*C_im

        Lambda_elements = jnp.repeat(Lambda[None, ...], input_sequence.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(input_sequence)
        elements = (Lambda_elements, Bu_elements)

        _, inner_states = jax.lax.associative_scan(binary_operator_diag, elements)

        y = jax.vmap(lambda x, u: (C @ x).real + D * u)(inner_states, input_sequence)

        return y
