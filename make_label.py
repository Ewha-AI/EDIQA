import os
import os.path as osp
from glob import glob 
import pandas as pd
import numpy as np
import csv


def make_mri_label():
    save_dir = '/media/wonkyong/hdd2/data/D2IQA/temp_test.csv'

    csv_root = '/media/wonkyong/268c4ce6-d8a7-453b-9607-b1a121f48743/data/results/d2iqa_lbset_ABIDEII_GU_1124_6_40'
    csv_list = sorted(glob(osp.join(csv_root, '*.csv')))

    with open(save_dir, "w", newline='') as csvfile:
        wr = csv.writer(csvfile, dialect="excel")
        title = ['img', 'mean']
        wr.writerow(title)

    for csv_file in csv_list:
        df = pd.read_csv(csv_file)

        scores = df[:2]['bbox_mAP'].item()
        fid = 'ABIDEII-GU_1_{}_session_1_anat_1_1_{}'.format(csv_file.split('_')[-6], csv_file.split('_')[-1].split('.')[0])

        # print([fid] + scores)
        # exit()
        
        with open(save_dir, "a", newline='') as csvfile :
            wr = csv.writer(csvfile, dialect="excel")
            wr.writerow([fid] + [scores])

def make_phantom_label():
    save_dir = '/media/wonkyong/hdd2/data/D2IQA/phantom_test.csv'

    csv_root = '/media/wonkyong/268c4ce6-d8a7-453b-9607-b1a121f48743/data/results/phantom/' # '../../data/results/phantom_test/'
    csv_list = sorted(glob(osp.join(csv_root, '*.csv')))

    lv = ['level1_100', 'level2_050', 'level3_025', 'level4_010', 'level5_005']

    with open(save_dir, "w", newline='') as csvfile:
        wr = csv.writer(csvfile, dialect="excel")
        title = ['img', 'mean']
        wr.writerow(title)

    for csv_file in csv_list:
        df = pd.read_csv(csv_file)

        scores = df[:5]
        fid = 'ge_chest_' + csv_file.split('_')[-1].split('.')[0]
        scores = list(scores['bbox_mAP'])

        # print([fid] + scores)
        # exit()
        
        for i in range(len(lv)):
            with open(save_dir, "a", newline='') as csvfile :
                wr = csv.writer(csvfile, dialect="excel")
                wr.writerow([fid+'_{}'.format(lv[i])] + [scores[i]])

def make_phantom_df():
    save_dir = '/media/wonkyong/hdd2/data/D2IQA/phantom_df.csv'

    with open(save_dir, "w", newline='') as csvfile:
        wr = csv.writer(csvfile, dialect="excel")
        title = ['img', 'mean', 'std']
        wr.writerow(title)

    csv_root = '/media/wonkyong/268c4ce6-d8a7-453b-9607-b1a121f48743/data/results/phantom/'
    csv_list = sorted(glob(osp.join(csv_root, '*.csv')))

    with open(save_dir, "w", newline='') as csvfile:
        wr = csv.writer(csvfile, dialect="excel")
        title = ['img', '1', '2', '3', '4', '5']
        wr.writerow(title)

    for csv_file in csv_list:
        df = pd.read_csv(csv_file)

        scores = df[:5]
        fid = csv_file.split('_')[-1].split('.')[0] + '.tiff'
        scores = list(scores['bbox_mAP'])

        # print([fid] + scores)
        # exit()
        
        with open(save_dir, "a", newline='') as csvfile :
            wr = csv.writer(csvfile, dialect="excel")
            wr.writerow([fid] + scores)

def make_mayo_label():
    val_pid = ['L333']
    train_pid = ['L096', 'L291', 'L310', 'L109', 'L143', 'L192', 'L286']

    detect_path = '/media/wonkyong/268c4ce6-d8a7-453b-9607-b1a121f48743/data/results/mayo'
    total_pid = sorted(os.listdir(detect_path)) # path to D2IQA detection results folder


    # for mode in ['train', 'val']:
    for mode in ['total']:
        print('{} start..'.format(mode))
        save_dir = '/media/wonkyong/hdd2/data/D2IQA/mayo_total.csv' # path to save the resulting csv file '/home/wonkyong/data/mayo/mayo57np.csv'

        with open(save_dir, "w", newline='') as csvfile:
            wr = csv.writer(csvfile, dialect="excel")
            title = ['img', 'mean', 'std']
            wr.writerow(title)

        if mode == 'train':
            pid_list = train_pid
        elif mode == 'val':
            pid_list = val_pid
        else:
            pid_list = total_pid

        for pid in pid_list:
            csv_root = osp.join(detect_path, pid)

            csv_list = sorted(glob(osp.join(csv_root, '*.csv')))
            print('{} image count: '.format(pid), len(csv_list))

            lv = ['full', 'full_0.5', 'quarter', 'quarter_0.5', 'quarter_1.0', 'quarter_1.5', 'quarter_2.0']

            # with open(save_dir, "w", newline='') as csvfile:
            #     wr = csv.writer(csvfile, dialect="excel")
            #     title = ['img', 'mean', 'std']
            #     wr.writerow(title)

            for csv_file in csv_list:
                df = pd.read_csv(csv_file)
                
                # check if csv is in right format
                if len(df) != 21:
                    raise AssertionError('Csv file not in right fotmat. Check {}'.format(csv_file))

                scores = df[:7]
                p, fid = csv_file.split('_')[-3], csv_file.split('_')[-2]
                if fid[-2] != '.':
                    fid = '{:03d}'.format(int(fid))
                    fid = fid + '.0'
                
                for i, l in enumerate(lv):
                    score = scores.iloc[i].values
                    level, score = score[0], score[1:]
                    if level != l:
                        raise AssertionError("Level index does not match.")
                    mean = np.mean(score)
                    std = np.std(score)
                    imgname = '{}_{}_{}'.format(p, l, fid)
                    with open(save_dir, "a", newline='') as csvfile :
                        wr = csv.writer(csvfile, dialect="excel")
                        wr.writerow([imgname, mean, std])

            print('{} done..'.format(pid))


if __name__ == '__main__':
    # Make labels for each dataset
    make_mayo_label()
    # make_phantom_label()
    # make_mri_label()

    # Optional
    # make_phantom_df()