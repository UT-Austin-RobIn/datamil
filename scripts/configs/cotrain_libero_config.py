from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder
import os
from datamil import libero_dataset_kwargs

def get_config(config_string=""):
    val_name, prior_name, idx_path = config_string.split(",")
    if idx_path == "none":
        idx_path = None

    DATA_DIR = f"{os.environ['DATA_DIR']}/tf"
    
    TARGET_KWARGS = libero_dataset_kwargs.copy()
    TARGET_KWARGS["name"] = val_name
    TARGET_KWARGS["data_dir"] = DATA_DIR
    TARGET_KWARGS["training_idxs_path"] = None  # use all target data

    PRIOR_KWARGS = libero_dataset_kwargs.copy()
    PRIOR_KWARGS["name"] = prior_name
    PRIOR_KWARGS["data_dir"] = DATA_DIR
    if idx_path is not None:
        PRIOR_KWARGS["training_idxs_path"] = idx_path
    
    mode, task = "full", "language_conditioned"
    frozen_keys = None
    max_steps = FieldReference(10_000)
    window_size = FieldReference(default=1)

    config = dict(
        use_proprio=True,
        action_chunks=8,
        
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        batch_size=128,
        shuffle_buffer_size=10000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=int(max_steps.get()//5),
        save_interval=int(max_steps.get()),
        save_ckpts=[10000],
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="octo_finetune", group=placeholder(str), entity=placeholder(str)
        ),

        dataset_kwargs_list=[TARGET_KWARGS, PRIOR_KWARGS],
        sample_weights=[1.0, 1.0], # ratio of target to prior data

        modality=task,
        finetuning_mode=mode,
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
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
        ),
        viz_kwargs=dict(
            eval_batch_size=128,
            trajs_for_metrics=100,
            trajs_for_viz=8,
            samples_per_state=8,
        ),
    )
    
    traj_transform_kwargs = dict(
        window_size=window_size,
        future_action_window_size=config['action_chunks']-1,
        goal_relabeling_strategy=None,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=0.0,
        ),
        # If the default data loading speed is too slow, try these:
        # num_parallel_calls=16,  # for less CPU-intensive ops
    )
    frame_transform_kwargs = dict(
        resize_size={
            "primary": (256, 256),  # workspace (3rd person) camera is at 256x256
            "wrist": (128, 128),  # wrist camera is at 128x128
        },
        image_augment_kwargs=[],
    )
    # If the default data loading speed is too slow, try these:
    config["frame_transform_threads"] = 16  # for the most CPU-intensive ops (decoding, resizing, augmenting)

    config["frame_transform_kwargs"] = frame_transform_kwargs
    config["traj_transform_kwargs"] = traj_transform_kwargs
    return ConfigDict(config)
