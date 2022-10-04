import os
import os.path as osp
from glob import glob 
import pandas as pd
import numpy as np
import csv


def make_mri_tiqa_label():
    save_dir = '../../data/results/csv/abide_d2iqa_label.csv'

    with open(save_dir, "w", newline='') as csvfile:
        wr = csv.writer(csvfile, dialect="excel")
        title = ['img', 'mean', 'std']
        wr.writerow(title)

    csv_root = '../../data/results/abide_low/'

    csv_list = sorted(glob(osp.join(csv_root, '*.csv')))

    lv = ['low']

    with open(save_dir, "w", newline='') as csvfile:
        wr = csv.writer(csvfile, dialect="excel")
        title = ['title', 'mean', 'std']
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

def make_phantom_tiqa_label():
    save_dir = '../../data/results/csv/phantom_test_label.csv'

    with open(save_dir, "w", newline='') as csvfile:
        wr = csv.writer(csvfile, dialect="excel")
        title = ['img', 'mean', 'std']
        wr.writerow(title)

    csv_root = '../../data/results/phantom_test/'

    csv_list = sorted(glob(osp.join(csv_root, '*.csv')))

    lv = ['level1_100', 'level2_050', 'level3_025', 'level4_010', 'level5_005']

    with open(save_dir, "w", newline='') as csvfile:
        wr = csv.writer(csvfile, dialect="excel")
        title = ['title', 'mean', 'std']
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




def make_phantom_label():
    save_dir = '../../data/results/csv/phantom.csv'

    with open(save_dir, "w", newline='') as csvfile:
        wr = csv.writer(csvfile, dialect="excel")
        title = ['img', 'mean', 'std']
        wr.writerow(title)

    csv_root = '../../data/results/phantom/'

    csv_list = sorted(glob(osp.join(csv_root, '*.csv')))

    lv = ['level1_100', 'level2_050', 'level3_025', 'level4_010', 'level5_005']

    with open(save_dir, "w", newline='') as csvfile:
        wr = csv.writer(csvfile, dialect="excel")
        title = ['title', '1', '2', '3', '4', '5']
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
    # train_pid = ['L096', 'L291', 'L310']
    val_pid = ['L333']
    train_pid = ['L096', 'L291', 'L310', 'L109', 'L143', 'L192', 'L286']
    total_pid = sorted(os.listdir('../../data/results/mayo'))
    # print(total_pid)
    # exit()

    # regression graph for mapping
    param = [-45.73929281275296, 63.34633213568207,-8.265482144511035, 19.104542650667085, 20.677533480046783]
    def objective(x, b1, b2, b3, b4, b5):
        logistic = (1/2) -(1/(1+np.exp((x-b3)*b2)))
        return b1 * logistic + b4 * x +b5


    # for mode in ['train', 'val']:
    for mode in ['total']:
        print('{} start..'.format(mode))
        save_dir = '/home/wonkyong/data/mayo/mayo57np.csv'#.format(mode)

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
            csv_root = '/media/wonkyong/268c4ce6-d8a7-453b-9607-b1a121f48743/data/results/mayo/{}/'.format(pid)

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
                # print(p, fid)
                if fid[-2] != '.':
                    fid = '{:03d}'.format(int(fid))
                    fid = fid + '.0'
                
                label = {}
                for i, l in enumerate(lv):
                    score = scores.iloc[i].values
                    level, score = score[0], score[1:]
                    # convert mAP to mos score by mapping to a regression graph
                    mscore = []
                    for m in range(len(score)):
                        # mscore.append(objective(score[m], param[0], param[1], param[2], param[3], param[4]))
                        mscore.append(score[m])
                    if level != l:
                        raise AssertionError("Level index does not match.")
                    mean = np.mean(mscore)
                    std = np.std(mscore)
                    imgname = '{}_{}_{}'.format(p, l, fid)
                    with open(save_dir, "a", newline='') as csvfile :
                        wr = csv.writer(csvfile, dialect="excel")
                        wr.writerow([imgname, mean, std])

            print('{} done..'.format(pid))


if __name__ == '__main__':
    # make_mayo_label()
    # make_phantom_label()
    # make_phantom_tiqa_label()
    make_mri_tiqa_label()