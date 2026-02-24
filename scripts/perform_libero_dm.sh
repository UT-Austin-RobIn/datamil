#!/bin/bash

set -e

NUM_TRIALS=30

FOLDER="book-caddy"
# FOLDER="bowl-cabinet"
# FOLDER="mug-mug"
# FOLDER="moka-moka"
# FOLDER="cream-butter"
# FOLDER="soup-sauce"
# FOLDER="stove-moka"
# FOLDER="mug-pudding"
# FOLDER="soup-cheese"
# FOLDER="mug-microwave"

DATASET="libero90"

TASK="study_scene1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
# TASK="kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
# TASK="living_room_scene5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
# TASK="kitchen_scene8_put_both_moka_pots_on_the_stove"
# TASK="living_room_scene2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"
# TASK="living_room_scene2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
# TASK="kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it"
# TASK="living_room_scene6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"
# TASK="living_room_scene1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
# TASK="kitchen_scene6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"


# Use a for loop to iterate over the range
for ((job_id=0; job_id<NUM_TRIALS; job_id++)); do
    echo "Running job $job_id"
    # Add your logic here for each job

    python datamil/metagradients/do_libero_data_selection_parallel.py \
        --config scripts/configs/dm_libero_config.py \
        --config.pretrained_path=hf://rail-berkeley/octo-small \
        --config.job_id $job_id \
        --config.folder_name $FOLDER \
        --config.val_dataset_kwargs.name $TASK \
        --config.dataset_kwargs.name $DATASET \
        --config.save_dir ${EXP_DIR}/datamodels

done

python datamil/utils/load_dm.py \
    --config scripts/configs/dm_libero_config.py \
    --config.pretrained_path=hf://rail-berkeley/octo-small \
    --config.folder_name $FOLDER \
    --config.val_dataset_kwargs.name $TASK \
    --config.dataset_kwargs.name $DATASET \
    --config.save_dir ${EXP_DIR}/datamodels \
    --config.topk 0.1