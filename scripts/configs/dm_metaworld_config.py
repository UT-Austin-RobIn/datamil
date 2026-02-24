from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder
import os

def get_config(config_string=""):

    data_path = os.path.join(os.environ['DATA_DIR'], "metaworld/mds_dataset")
    TRAIN_DATA_KWARGS = {
        "name": "prior",
        "data_dir": f"{data_path}",
    }

    TARGET_DATA_KWARGS = {
        "name": "pick-place-wall",
        "data_dir": f"{data_path}/seed",
    }

    bob_steps = 100
    max_steps = FieldReference(1000 + bob_steps)

    config = dict(
        loss_type="mse",

        ## added
        job_id=placeholder(int),
        folder_name=placeholder(str),
        num_workers=32, # fewer workers (half of # of cpus typically)

        ## added
        batch_size=16364,
        mini_batch_size=16364,
        val_batch_size=16364,
        mini_val_batch_size=16364,

        num_steps=max_steps,

        # added
        bob_steps=bob_steps,
        candidate_size=1.0,#0.2,

        log_interval=100,
        eval_interval=int(max_steps.get()//5),
        save_interval=int(max_steps.get()//5),
        save_dir=placeholder(str),
        seed=42,
        dataset_kwargs=TRAIN_DATA_KWARGS,
        val_dataset_kwargs=TARGET_DATA_KWARGS,

        optimizer=dict(
            learning_rate = dict(
                name="cosine",
                init_value=0.0001,
                peak_value=0.005,
                warmup_steps=int(max_steps.get()//25),
                decay_steps=max_steps.get(), # - int(max_steps.get()//25),
                end_value=1e-8,
            ),
            weight_decay=1e-5,
            clip_gradient=1.0,
            grad_accumulation_steps=None,  # if you are using grad accumulation, you need to adjust max_steps accordingly
        ),
    )

    return ConfigDict(config)
