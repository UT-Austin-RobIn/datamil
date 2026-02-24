import os
import glob
import numpy as np
from absl import flags, app
from ml_collections import config_flags

import datamil.metagradients.flags_config as flags_config
FLAGS = flags.FLAGS

def load_dm(path, n_clusters, iter_num=None):
    iter_num = len(glob.glob(os.path.join(path, 'iter_*'))) if iter_num is None else iter_num
    all_dms = []
    count_dms = 0

    for iter_id in range(iter_num+1):
        iter_path = os.path.join(path, f'iter_{iter_id}')
        # print(iter_path)
        try:
            filtered_dm = np.load(os.path.join(iter_path, 'datamodels.npy'))[1000000:1000000+n_clusters]
            all_dms.append(filtered_dm)
            count_dms += 1
        except:
            pass
    all_dms = np.array(all_dms)
    avg_dm = np.mean(all_dms, axis=0, where=all_dms!=0)
    avg_dm = np.nan_to_num(avg_dm)

    print(count_dms, np.sum(avg_dm!=0), n_clusters)
    
    return avg_dm

def main(_):
    mds_path = os.path.join(FLAGS.config.dataset_kwargs.data_dir, FLAGS.config.dataset_kwargs.name)
    n_clusters = len(glob.glob(os.path.join(mds_path, 'traj_*')))

    dm_path = os.path.join(FLAGS.config.save_dir, FLAGS.config.dataset_kwargs.name, FLAGS.config.folder_name)
    avg_dm = load_dm(dm_path, n_clusters)
    np.save(os.path.join(dm_path, 'avg_datamodel.npy'), avg_dm)

    selected_indices = np.argsort(avg_dm)[:int(len(avg_dm)*FLAGS.config.topk)]
    print(f"==> Selected {len(selected_indices)} clusters. Saving selected indices to {os.path.join(dm_path, f'selected_indices_topk{FLAGS.config.topk}.npy')}")
    np.save(os.path.join(dm_path, f'selected_indices_topk{FLAGS.config.topk}.npy'), selected_indices)

    # # plot
    # tasks = sorted(os.listdir(os.path.join(os.environ["DATA_DIR"], "libero_90")))
    # counts = np.zeros(len(tasks))
    # for idx in selected_indices:
    #     counts[idx//50] += 1
    # sorted_ = sorted(zip(tasks, counts), key=lambda x: x[1], reverse=True)
    # sorted_tasks, sorted_counts = zip(*sorted_)

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(30, 10))
    # plt.bar(sorted_tasks, sorted_counts)
    # plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    # plt.xlabel('Tasks')
    # plt.ylabel('Number of selected trajectories')
    # plt.savefig(os.path.join(dm_path, f'selected_tasks_topk{FLAGS.config.topk}.png'), bbox_inches='tight')

if __name__ == '__main__':
    app.run(main)