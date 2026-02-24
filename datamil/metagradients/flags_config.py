import os
from absl import flags
from ml_collections import config_flags

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")

config_flags.DEFINE_config_file(
    "config",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)