import flax.struct
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict
from typing import Sequence, Callable, Optional
import distrax
import optax
import flax
import numpy as np

class MultiHeadedMLP(nn.Module):
    n_heads: int
    output_dims: Sequence[int]
    hidden_sizes: Sequence[int]
    hidden_nonlinearity: Optional[Callable] = nn.relu
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for size in self.hidden_sizes:
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.Dense(size)(x)
            if self.hidden_nonlinearity:
                x = self.hidden_nonlinearity(x)
        outputs = []
        for out_dim in self.output_dims:
            outputs.append(nn.Dense(out_dim)(x))
        return outputs

class GaussianMLPTwoHeaded(nn.Module):
    action_dim: int
    hidden_sizes: Sequence[int]
    hidden_nonlinearity: Callable = nn.tanh
    learn_std: bool = True
    init_std: float = 1.0
    min_std: float = 1e-6
    max_std: Optional[float] = None
    std_parameterization: str = 'exp'
    layer_norm: bool = False

    def setup(self):
        self.net = MultiHeadedMLP(
            n_heads=2,
            output_dims=[self.action_dim, self.action_dim],
            hidden_sizes=self.hidden_sizes,
            hidden_nonlinearity=self.hidden_nonlinearity,
            layer_norm=self.layer_norm,
        )
        log_init = jnp.log(self.init_std)
        if self.learn_std:
            self.log_init_std = self.param('log_init_std', lambda rng: jnp.array(log_init))
        else:
            self.log_init_std = log_init

    @nn.compact
    def __call__(self, x):
        mean, log_std_uncentered = self.net(x)
        min_p = jnp.log(self.min_std) if self.min_std is not None else None
        max_p = jnp.log(self.max_std) if self.max_std is not None else None
        log_std = jnp.clip(log_std_uncentered, a_min=min_p, a_max=max_p)
        # log_std = log_std + self.log_init_std
        std = jnp.exp(log_std) if self.std_parameterization == 'exp' else jnp.log1p(jnp.exp(log_std))
        base_dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
        return base_dist
        # tanh_bij = distrax.Block(distrax.Tanh(), 1)
        # return distrax.Transformed(base_dist, tanh_bij)

from typing import Any, Mapping, Sequence, Union
from flax import struct

PyTree = Union[jax.typing.ArrayLike, Mapping[str, "PyTree"]]
Config = Union[Any, Mapping[str, "Config"]]
Params = Mapping[str, PyTree]
@struct.dataclass
class MetaworldModel:
    module: GaussianMLPTwoHeaded = struct.field(pytree_node=False)
    config: Config = struct.field(pytree_node=False)
    params: Params
    clip_epsilon: float = 1e-6

    @classmethod
    def from_config(cls,
                rng,
                config
            ):

        module = GaussianMLPTwoHeaded(
            **config['model_kwargs'],
        )

        dummy_obs = jnp.zeros((1, config['obs_dim']))
        variables = module.init(rng, dummy_obs)
        params = variables['params']

        return cls(module=module, config=config, params=params)
    
    @staticmethod
    def _clip_but_pass_gradient(x, lower: float = 0., upper: float = 1.) -> jnp.ndarray:
        clip_up = x > upper
        clip_low = x < lower
        clip = (upper - x) * clip_up + (lower - x) * clip_low
        return x + jax.lax.stop_gradient(clip)
    
    def per_sample_loss(self, dist, act_batch: jnp.ndarray) -> jnp.ndarray:
        pre_tanh_value = jnp.log(
            (1 + self.clip_epsilon + act_batch) / (1 + self.clip_epsilon - act_batch)) / 2
        norm_lp = dist.log_prob(pre_tanh_value)
        ret = (norm_lp - jnp.sum(
            jnp.log(self._clip_but_pass_gradient((1. - act_batch**2)) + self.clip_epsilon),
            axis=-1))
        loss = -ret

        return loss
    
    @jax.jit
    def get_fast_action(self, obs: jnp.ndarray):
        dist = self.module.apply({'params': self.params}, obs[None, :])
        
        # rng = jax.random.PRNGKey(0)
        # act = jnp.squeeze(dist.sample(seed=rng))
        act = jnp.squeeze(jnp.tanh(dist.mean()))
        act = jnp.tanh(act)

        return act

    def get_action(self, obs: jnp.ndarray) -> jnp.ndarray:
        
        act = self.get_fast_action(obs)
        
        # convert from jax to numpy
        act = jax.device_get(act)
        return np.asarray(act)

def make_mw_model(seed, obs_dim=39, action_dim=4):
    # following the conventions from metagradient code
    rng = jax.random.PRNGKey(seed)

    config = dict(
        # clip_epsilon=1e-6,
        obs_dim=obs_dim,
        model_kwargs=dict(
            action_dim=action_dim,
            hidden_sizes=[400, 400, 400],
            hidden_nonlinearity=nn.relu,
            learn_std=True,
            init_std=1.0,
            min_std=jnp.exp(-20.0),
            max_std=jnp.exp(2.0),
            std_parameterization='exp',
            layer_norm=False,
        )
    )
    return MetaworldModel.from_config(
        rng=rng,
        config=config,
    )
