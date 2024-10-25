from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import argparse
import json


def calculate_metrics(pred_path, gt_path):
    corr = {}
    pred = pd.read_csv(pred_path, header=None)
    pred.index = pred[0]
    
    gt = pd.read_csv(gt_path)
    gt.index = gt['img']
    gt = gt[gt['img'].isin(pred[0])]

    total = pred.join(gt, how='outer')
    if len(total[total.isnull()['mean']==True]) != 0:
        total = total.dropna()
        print('missing values exists. Drop missing {} rows'.format(len(total[total.isnull()['mean']==True])))

    plcc = np.corrcoef(np.array(total['mean']), np.array(total[1]))[0,1]
    srocc, _ = spearmanr(pd.DataFrame([np.array(total['mean']), np.array(total[1])]).T)
    corr['plcc'], corr['srocc'] = plcc, srocc

    print(json.dumps(corr, indent=4))


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='TIQA')
    parser.add_argument('--pred_path', type=str, required=True, help='path to the prediction file')
    parser.add_argument('--gt_path', type=str, required=True, help='path to the ground truth file')
    args = parser.parse_args()

    calculate_metrics(args.pred_path, args.gt_path, args.data_type)