from typing import Dict, Tuple

# import distrax
from einops import rearrange
import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

def per_sample_masked_mean(x, mask):
    mask = jnp.broadcast_to(mask, x.shape)

    sample_sum_x = jnp.sum(x * mask, axis=tuple(range(1, x.ndim)))
    sum_mask = jnp.sum(mask, axis=tuple(range(1, mask.ndim)))

    return sample_sum_x / jnp.clip(sum_mask, a_min=1e-5, a_max=None)

def chunk_actions(actions: ArrayLike, pred_horizon: int) -> Array:
    """Chunk actions for predicting actions `pred_horizon` steps into the future.

    The resulting actions have shape (batch, actions.shape[-2] - (pred_horizon - 1), pred_horizon, action_dim)

    For example: chunk_actions([a_1, a_2, a_3, a_4, a_5], 3) ->
        [
            [a_1, a_2, a_3],
            [a_2, a_3, a_4],
            [a_3, a_4, a_5],
        ]

    """
    assert (
        actions.ndim == 3
    ), f"Expected actions to have shape (batch, window_size, action_dim), but got shape {actions.shape}"
    window_size = actions.shape[1]
    assert (
        window_size >= pred_horizon
    ), f"pred_horizon {pred_horizon} too large for window size {window_size}"
    chunk_window_size = window_size - (pred_horizon - 1)

    curr_step = jnp.arange(chunk_window_size)
    action_offset = jnp.arange(pred_horizon)
    chunk_indices = curr_step[:, None] + action_offset[None, :]
    return actions[:, chunk_indices]


def _check_action_window_size(actions, window_size, pred_horizon):
    assert (
        actions.shape[1] >= window_size + pred_horizon - 1
    ), f"""
        To predict actions for window_size {window_size} and future prediction horizon {pred_horizon},
        the ground-truth actions must have at least {window_size + pred_horizon - 1} timesteps, but got shape {actions.shape}.

        Did you make sure to set "future_action_window_size" correctly in the data config?
    """

def per_sample_continuous_loss(
    pred_value: ArrayLike,
    ground_truth_value: ArrayLike,
    mask: ArrayLike,
    loss_type: str = "mse",
) -> Array:
    """
    Args:
        pred_value: shape (batch_dims...)
        ground_truth_value: continuous values w/ shape (batch_dims...)
        mask: broadcastable to ground_truth
    """
    if loss_type == "mse":
        loss = jnp.square(pred_value - ground_truth_value)
    elif loss_type == "l1":
        loss = jnp.abs(pred_value - ground_truth_value)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    loss = per_sample_masked_mean(loss, mask)

    mse = jnp.square(pred_value - ground_truth_value)
    mse = per_sample_masked_mean(mse, mask)
    return loss, {
        "loss": loss,
        "mse": mse,
    }

def diffusion_per_sample_loss(
    self,
    transformer_outputs,
    actions: ArrayLike,
    pad_mask: ArrayLike,
    train: bool = True,
) -> Tuple[Array, Dict[str, Array]]:
    """Computes the loss for the diffusion objective.

    Args:
        transformer_ouputs: must contain self.readout_key with shape (batch_size, window_size, num_tokens,
            embedding_size)
        actions: shape (batch_size, >= window_size + pred_horizon - 1, action_dim)
        pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep

    Returns:
        loss: float
        metrics: dict
    """
    batch_size, window_size = pad_mask.shape
    _check_action_window_size(actions, window_size, self.pred_horizon)
    actions_chunked = chunk_actions(actions, self.pred_horizon)
    actions_chunked = actions_chunked[:, :window_size]
    # fold action_dim and pred_horizon into one dimension
    actions_flat = rearrange(actions_chunked, "b w p a -> b w (p a)")
    actions_flat = jnp.clip(actions_flat, -self.max_action, self.max_action)

    # piggy-back on the dropout rng chain for diffusion rng
    rng = self.make_rng("dropout")

    time_key, noise_key = jax.random.split(rng)
    time = jax.random.randint(
        time_key, (batch_size, window_size, 1), 0, self.diffusion_steps
    )
    noise = jax.random.normal(noise_key, actions_flat.shape)

    alpha_hat = self.alpha_hats[time]
    alpha_1 = jnp.sqrt(alpha_hat)
    alpha_2 = jnp.sqrt(1 - alpha_hat)
    noisy_actions = alpha_1 * actions_flat + alpha_2 * noise

    pred_eps = self(
        transformer_outputs, train=train, time=time, noisy_actions=noisy_actions
    )

    loss, metrics = per_sample_continuous_loss(
        pred_eps, noise, pad_mask[:, :, None], loss_type=self.loss_type
    )
    # Sum over action dimension instead of averaging
    loss = loss * self.action_dim
    metrics["loss"] = metrics["loss"] * self.action_dim
    metrics["mse"] = metrics["mse"] * self.action_dim
    return loss, metrics