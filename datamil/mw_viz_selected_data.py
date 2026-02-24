import glob
import os  
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy

def load_dm_mask(path, topk, iter_num=None):
    iter_num = len(os.listdir(path)) if iter_num is None else iter_num
    all_dms = []
    count_dms = 0

    n_trajs = 8245
    for iter_id in range(iter_num):
        iter_path = os.path.join(path, f'iter_{iter_id}')
        
        try:
            filtered_dm = np.load(os.path.join(iter_path, 'datamodels.npy'))[1000000:1000000+n_trajs]
            all_dms.append(filtered_dm)
            count_dms += 1
        except:
            pass
    all_dms = np.array(all_dms)
    avg_dm = np.mean(all_dms, axis=0, where=all_dms!=0)
    avg_dm = np.nan_to_num(avg_dm)

    print(count_dms, np.sum(avg_dm!=0), n_trajs)
    # assert n_trajs == np.sum(avg_dm!=0)

    # Option 1
    threshold = np.sort(avg_dm, axis=0)[int(topk*n_trajs)]
    selected_dm = avg_dm <= threshold
    print(threshold, np.sum(selected_dm))

    return selected_dm, count_dms

def get_task_name_and_type_from_path(path):
    splits = path.split('/')
    task_name = splits[-1].split('_')[0]
    if 'v2' in task_name:
        task_name = task_name.replace('-v2', '')
    return (task_name, 'autonomous' in path)

def plot_percentages(filter_info, all_info, title, output_dir, task_name=None):

    filter_info['demos'] = {k: v/all_info['demos'][k] for k, v in filter_info['demos'].items()}
    filter_info['autonomous'] = {k: v/all_info['autonomous'][k] for k, v in filter_info['autonomous'].items()}

    items = [(k, filter_info['demos'][k], filter_info['autonomous'][k]) for k in filter_info['demos'].keys()]
    items.sort(key=lambda x: (x[1], x[2]), reverse=True)
    keys, _, _ = zip(*items)

    # plot task and autonomous bins side by side for each key
    fig, ax = plt.subplots(1, figsize=(15, 5))
    
    x = np.arange(len(keys))
    width=0.35
    colors = ['g']*len(keys)
    for k in keys:
        if task_name and k == task_name:
            colors[keys.index(k)] = 'r'
    values = [filter_info['demos'][k] for k in keys]
    ax.bar(x - width/2, values, width, label='Successful Traj', color=colors)
    values = [filter_info['autonomous'][k] for k in keys]
    ax.bar(x + width/2, values, width, label='Failed Traj', color='b')

    ax.set_title(title)
    ax.set_ylim([0, 1])
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_xlabel('Tasks')
    ax.set_ylabel('% trajectory selected')

    custom_legend_colors = [
        mpatches.Patch(color='g', label='Successful Traj'),
        mpatches.Patch(color='r', label='Successful Traj (Correct Task)'),
        mpatches.Patch(color='b', label='Failed Traj')
    ]
    ax.legend(handles=custom_legend_colors, loc='upper right')
    # plt.show()
    plt.savefig(f'{output_dir}/percentage_n-dm{title}.png', bbox_inches='tight')

def work(title, selected_mask, all_files):
    base_dict = {get_task_name_and_type_from_path(f)[0] : 0  for f in all_files}
    def init_info():
        return {
            'demos': base_dict.copy(),
            'autonomous': base_dict.copy()
        }
    filtered_info = init_info()
    all_info = init_info()
    for i, f in enumerate(all_files):
        name, autonomous = get_task_name_and_type_from_path(f)
        demo_type = 'autonomous' if autonomous else 'demos'
        all_info[demo_type][name] = all_info[demo_type].get(name, 0) + 1

        if selected_mask[i]:
            filtered_info[demo_type][name] = filtered_info[demo_type].get(name, 0) + 1

    plot_percentages(filtered_info, all_info, title, ".", task_name='pick-place-wall')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data/metaworld/prior/")
    parser.add_argument('--dm_path', type=str, default="results/metaworld/pick-place-wall_dm")
    parser.add_argument('--topk', type=float, default=0.1)
    args = parser.parse_args()
    
    all_files = sorted(glob.glob(os.path.join(args.data_path, "**", "*.h5"), recursive=True))

    selected_mask, count_dms = load_dm_mask(args.dm_path, topk=args.topk, iter_num=None)
    work(f"{count_dms}", selected_mask, all_files)

