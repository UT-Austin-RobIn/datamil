# DataMIL: Selecting Data for Robot Imitation Learning with Datamodels
This repository is the official code for the paper [DataMIL: Selecting Data for Robot Imitation Learning with Datamodels](https://arxiv.org/abs/2505.09603). We provide detailed instructions for setting up the data, computing datamodels and training policies on selected data, on both the MetaWorld and LIBERO benchmarks.

## Installation
```
# create conda environment
conda create -n datamil python=3.10
conda activate datamil

# setup jax
python -m pip install tensorflow[and-cuda]==2.14.0
pip install numpy==1.24.3

conda install -c conda-forge cudnn=8.8 cuda-version=11.8
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# setup datamil repo
git clone git@github.com:ShivinDass/datamil.git
cd datamil
pip install -e .

# setup the base data and experiment paths
export DATA_DIR=/path/to/data_dir
export EXP_DIR=/path/to/experiments_dir
```


### Setup StreamingDataset
We use [streaming](https://github.com/mosaicml/streaming) for deterministic dataloading during metagradient estimation. Setup our provided fork.
```
git clone https://github.com/ShivinDass/streaming.git
cd streaming 
# make sure you're on 'datamil_v1' branch
pip install -e .
```

### (Optional) Install Octo (only for LIBERO experiments)
```
git clone https://github.com/octo-models/octo.git
cd octo
git checkout 653c54acde686fde619855f2eac0dd6edad7116b 
pip install -e . --no-deps
```

### (Optional) Install LIBERO (only for evaluating policies)
Follow instruction [here](https://github.com/Lifelong-Robot-Learning/LIBERO) to install the LIBERO Benchmark. (Preferably do this before installing other requirements to avoid dependency issues)

### (Optional) Install MetaWorld (only for evaluating policies)
Follow instructions [here](https://github.com/Farama-Foundation/Metaworld/tree/a98086ababc81560772e27e7f63fe5d120c4cc50) to install Metaworld. We use an older commit of metaworld.

Note: if cython compilation issues arise, then downgrade cython version to: `pip install cython==0.29.37`.

## Quick Start - MetaWorld

### 0. Setting Up Data
Download the metaworld dataset from [here](https://utexas.box.com/s/nwsri7jg3h85g686cdbd80h3viqple6d), unzip it and place it in your `DATA_DIR`.

### 1. Computing Datamodels
```
./scripts/perform_mw_dm.sh
```

If the training leads to file rate limit errors then set `ulimit -n 10000`.
The trained datamodels will be stored in `EXP_DIR/metaworld/<task>_dm` folder. 

### 2. Training policy on selected data
```
python scripts/selected_training_mw.py --data_path ${DATA_DIR}/metaworld/prior --dm_path ${EXP_DIR}/datamodels/metaworld/pick-place-wall_dm --topk 0.1
``` 
The script will train and evaluate 5 different seeds of a policy on the top 10% of the selected trajectories. The results would be printed on console. 

## Quick Start - LIBERO

### 0. Setting Up Data
Download the hdf5 dataset following the instructions on LIBERO's [websit](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/master?tab=readme-ov-file#datasets) and place it in your `DATA_DIR`.

**Converting data to [RLDS](https://github.com/google-research/rlds):** We use the rlds data format for training the octo policy. This would be useful for when we train the final policy on the selected data. Detailed instructions for converting LIBERO90 and LIBERO10 datasets are provided [here](rlds_utils/README.md).

**Converting data to [MDS](https://github.com/google-research/rlds):** We use [streaming](https://github.com/mosaicml/streaming) for deterministic data loading during data selection. Convert the rlds data to mds format using the provided script.
```
# convert libero90 rlds data to mds format
python scripts/libero_rlds_to_mds.py --data_path ${DATA_DIR}/tf/libero90 --out_path ${DATA_DIR}/mds

# convert book caddy task to mds format
python scripts/libero_rlds_to_mds.py --data_path ${DATA_DIR}/tf/study_scene1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy --out_path ${DATA_DIR}/mds
```

### 1. Computing Datamodels
```
export CUDA_VISIBLE_DEVICES=0,1,2,3 # adjust batch size in config accordingly
./scripts/perform_libero_dm.sh
```
The script will compute datamodels for the specified number of iterations and select the `topk` trajectories at the end, saving the results in `EXP_DIR/datamodels/libero90/<task>/`.

### 2. Training policy on selected data
```
export CUDA_VISIBLE_DEVICES=0,1,2,3 # adjust batch size in config accordingly
./scripts/selected_cotrain_libero.sh
```
The script will train and evaluate 5 different seeds of a policy on the selected indices from (1). The results would be saved in `EXP_DIR/eval_results/<task>`.

## Citation
```
@article{dass2025datamil,
  title={DataMIL: Selecting Data for Robot Imitation Learning with Datamodels},
  author={Dass, Shivin and Khaddaj, Alaa and Engstrom, Logan and Madry, Aleksander and Ilyas, Andrew and Martín-Martín, Roberto},
  journal={arXiv preprint arXiv:2505.09603},
  year={2025}
}
```
