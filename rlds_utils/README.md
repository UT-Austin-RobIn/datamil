# Instructions for Downloading and Processing LIBERO Data
RLDS conversion code adapted from: https://github.com/kpertsch/rlds_dataset_builder

## Setting up LIBERO
1. Install LIBERO following [these](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/master?tab=readme-ov-file#installtion) instructions -- no need to install torch, torchvision and torchaudio so ignore that line. You should be able to do it on top of an existing octo environment (it worked for me).

2. Download the data as described here: https://github.com/Lifelong-Robot-Learning/LIBERO/tree/master?tab=readme-ov-file#datasets

## Converting LIBERO data into RLDS format
(Assuming ```export DATA_DIR=...``` is set to the directory where the original datasets are downloaded. The rlds data would be written to ```DATA_DIR/tf```).

1. The folders ```rlds_dataset_builders/libero90``` and ```rlds_dataset_builders/libero90_horizon``` correspond to trajectory and horizon level datasets respectively.

2. Run the conversion script,
    ```
    cd rlds_dataset_builders/libero90
    tfds build --overwrite --data_dir ${DATA_DIR}/tf
    ```

3. For individually converting all libero10 tasks, run ```./val_datagen_script.sh```

4. To generate normalization statistics for all datasets, 
```
python generate_dataset_statistics.py --prior_path ${DATA_DIR}/tf/libero90
```
This will save the statistics in the current directory called ```dataset_statistics_[hash].json```. To make sure all datasets use the same statistics, set ```export DATASET_STATISTICS=path/to/dataset/statistics```.