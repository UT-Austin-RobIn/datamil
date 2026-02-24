from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder
import copy
import os

import tensorflow as tf

def get_config(config_string=""):
    data_path = os.path.join(os.environ["DATA_DIR"], "mds")
    TRAIN_DATA_KWARGS = {
        "name": placeholder(str),
        "data_dir": data_path,
    }

    TARGET_DATA_KWARGS = copy.deepcopy(TRAIN_DATA_KWARGS)
    TARGET_DATA_KWARGS["name"] = placeholder(str)

    frozen_keys = None
    max_steps = FieldReference(1000)
    window_size = FieldReference(default=1)

    BATCH_SIZE = 512
    config = dict(
        use_proprio=True,
        action_chunks=8,
        loss_type="mse", #"l1",

        topk=0.1,

        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),

        ## added
        job_id=placeholder(int),
        folder_name=placeholder(str),
        num_workers=32, # fewer workers (half of # of cpus typically)

        ## added
        batch_size=BATCH_SIZE,
        mini_batch_size=BATCH_SIZE,
        val_batch_size=BATCH_SIZE,
        mini_val_batch_size=BATCH_SIZE,

        shuffle_buffer_size=10000,
        num_steps=max_steps,

        # added
        bob_steps=100,
        candidate_size=1.0,#0.2,

        log_interval=100,
        eval_interval=int(max_steps.get()//5),
        save_interval=int(max_steps.get()//5),
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="octo_finetune", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs=TRAIN_DATA_KWARGS,
        val_dataset_kwargs=TARGET_DATA_KWARGS,

        modality="language_conditioned",
        finetuning_mode="full",
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=3e-4,
                warmup_steps=int(max_steps.get()//25),
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,  # if you are using grad accumulation, you need to adjust max_steps accordingly
        ),
    )
    return ConfigDict(config)
