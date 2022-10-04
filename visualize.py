# from https://github.com/aivclab/detr/blob/master/util/plot_utils.py

"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import functools
from pathlib import Path, PurePath


def plot_logs(log_folder, logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='logs.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]
    # dfs = [dfs[0][:]]
    # print(dfs)
    # exit()

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(
                    np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
                ).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'val_{field}'],
                    # y=[f'{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )

    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)
        ax.grid(True)

    plt.savefig('{}.png'.format(log_folder))

def plot_comparison(log_folder, logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='logs.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs_dict = {}
    for i in range(len(logs)):
        dfs_dict[i] = [pd.read_json(Path(logs[i]) / log_name, lines=True)]
        col = dfs_dict[i][0].columns
        col = ['{}_{}'.format(logs[i], col[j]) if col[j] != 'Epoch' else col[j] for j in range(len(col))]
        dfs_dict[i][0].columns = col
        # dfs = [dfs[0][:]]
        # print(dfs)
        # exit()

    # concat all dfs
    dfs = functools.reduce(lambda left,right: pd.merge(left,right,on='Epoch'), [dfs_dict[i][0] for i in range(len(logs))])

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    # print(['val_{}'.format(logs[i]) for i in range(len(logs))])
    for df, color in zip([dfs], sns.color_palette(n_colors=len(logs))):
        color = sns.color_palette(n_colors=len(logs))
        for j, field in enumerate(fields):
            df.plot(
                y=[f'train_loss', f'val_plcc', f'test_plcc'],
                # y=['val_{}'.format(field) for i in range(len(logs))],
                ax=axs[j],
                color=color,
                style=['-'] * 4
            )

    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)
        ax.grid(True)
        

    # plt.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2))
    # plt.tight_layout()
    plt.savefig('{}.png'.format(log_folder))


def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs


if __name__=='__main__':

    names = ['lr1e-4_plateau', 'triq_numclass1']
    # names = ['25pid_val143_batch256']

    # log_directory = []
    # for i in names:
    #     log_directory.append(Path('work_dirs/{}'.format(i)))
    # # print('log dir: ', log_directory)
    # fields_of_interest = (
    #     # 'loss',
    #     'plcc'
    #     )
    # plot_comparison('comp', log_directory,
    #         fields_of_interest)
    # exit()

    # for n in names:
    #     log_name = '25pid_val143_batch256'
    #     log_directory = [Path('work_dirs/{}'.format(log_name))]

    #     fields_of_interest = (
    #         # 'loss',
    #         'plcc'
    #         )
    #     plot_logs(log_name, log_directory,
    #             fields_of_interest)


    for name in names:
        print('work_dirs/{}/logs.txt'.format(name))
        log = pd.read_json('work_dirs/{}/logs.txt'.format(name), lines=True)
        train_y = log['train_plcc']
        val_y = log['val_plcc']
        test_y = log['test_plcc'][:100]

        print(len(train_y), len(val_y), len(test_y))

        # plt.plot(train_y, label='train_plcc')
        # plt.plot(val_y, label='val_plcc')
        plt.plot(test_y, label='{}_test_plcc'.format(name))
    plt.legend()
    plt.savefig('comp_tiqa_triq_256.png')