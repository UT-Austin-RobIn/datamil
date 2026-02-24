import os
import tensorflow as tf


def custom_dataset_transform(trajectory):
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.clip_by_value(trajectory["action"][:, -1:], 0, 1),
        ),
        axis=-1,
    )
    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :8]

    return trajectory

libero_dataset_kwargs = {
    "name": None,
    "data_dir": None,
    "image_obs_keys": {"primary": "image", "wrist": "wrist_image"},
    "state_obs_keys": ["state"],
    "language_key": "language_instruction",
    "action_proprio_normalization_type": "normal",
    "absolute_action_mask": [False, False, False, False, False, False, True],
    "action_normalization_mask": [True, True, True, True, True, True, False],
    "standardize_fn": custom_dataset_transform,
    "dataset_statistics": os.environ.get("DATASET_STATISTICS", None) # optional path to dataset statistics json file, which is used for standardization
}
