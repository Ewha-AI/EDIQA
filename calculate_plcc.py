import scipy
import scipy.stats
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='TIQA')
parser.add_argument('--work_dirs', type=str, required=True, help='work directory name to save model')
parser.add_argument('--data_type', type=str, default='mayo', help='data type (mayo|phantom|mri)')
args = parser.parse_args()

# calculate plcc and srocc for d2iqa
tiqa = pd.read_csv('results/{}_d2iqa_{}.csv'.format(args.work_dirs, args.data_type), header=None)
tiqa.index = tiqa[0]
if args.data_type == 'mayo':
    gt = pd.read_csv('../../data/nimg_3ch/mayo57np.csv')
    gt.index = gt['img']
    gt = gt[gt['img'].isin(tiqa[0])]
elif args.data_type == 'phantom':
    gt = pd.read_csv('../../data/csv/phantom_test_label.csv')
    gt.index = gt['title']
    gt = gt[gt['title'].isin(tiqa[0])]
else:
    gt = pd.read_csv('../../data/ABIDEII-GU_3ch/d2iqa_lbset_ABIDEII_GU_0928_lb_100.csv')
    gt.index = gt['title']
    gt = gt[gt['title'].isin(tiqa[0])]
total = tiqa.join(gt, how='outer')
if len(total[total.isnull()['mean']==True]) != 0:
    total = total.dropna()
    print('missing values exists. Drop missing {} rows'.format(len(total[total.isnull()['mean']==True])))

plcc = np.corrcoef(np.array(total['mean']), np.array(total[1]))[0,1]
srocc, _ = spearmanr(pd.DataFrame([np.array(total['mean']), np.array(total[1])]).T)

print('({})Correlation with D2IQA: PLCC={}, SROCC={}'.format(len(total), plcc, srocc))


# calculate plcc and srocc for rads
if args.data_type == 'mayo':
    tiqa = pd.read_csv('results/{}_rads_{}.csv'.format(args.work_dirs, args.data_type), header=None)
    tiqa.index = tiqa[0]
    gt = pd.read_csv('../../data/nimg-test-3channel/mayo_test.csv') # mayo_test.csv
    gt.index = gt['img']
    gt = gt[gt['img'].isin(tiqa[0])]
    total = tiqa.join(gt, how='outer')

    plcc = np.corrcoef(np.array(total['mean']), np.array(total[1]))[0,1]
    srocc, _ = spearmanr(pd.DataFrame([np.array(total['mean']), np.array(total[1])]).T)

    print('({})Correlation with Radiologists: PLCC={}, SROCC={}'.format(len(total), plcc, srocc))

elif args.data_type == 'mri':
    tiqa = pd.read_csv('results/{}_rads_{}.csv'.format(args.work_dirs, args.data_type), header=None)
    tiqa.index = tiqa[0]
    gt = pd.read_csv('../../data/nimg-test-3channel/mayo_test.csv') # mayo_test.csv
    gt.index = gt['img']
    gt = gt[gt['img'].isin(tiqa[0])]
    total = tiqa.join(gt, how='outer')

    plcc = np.corrcoef(np.array(total['mean']), np.array(total[1]))[0,1]
    srocc, _ = spearmanr(pd.DataFrame([np.array(total['mean']), np.array(total[1])]).T)

    print('({})Correlation with Radiologists: PLCC={}, SROCC={}'.format(len(total), plcc, srocc))