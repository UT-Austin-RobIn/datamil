#!/bin/bash

set -e

NUM_TRIALS=50

TASK=pick-place-wall
FOLDER=${TASK}_dm
DATASET=prior

# Use a for loop to iterate over the range
for ((job_id=0; job_id<NUM_TRIALS; job_id++)); do
    echo "Running job $job_id"

    python datamil/metagradients/do_mw_data_selection_parallel.py \
        --config scripts/configs/dm_metaworld_config.py \
        --config.job_id $job_id \
        --config.folder_name $FOLDER \
        --config.val_dataset_kwargs.name $TASK \
        --config.dataset_kwargs.name $DATASET \
        --config.save_dir ${EXP_DIR}/datamodels/metaworld \
        --config.candidate_size 0.5
done
