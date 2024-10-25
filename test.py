import torch
from torch.utils.data import DataLoader

from dataset import EDIQAData
from glob import glob
import pandas as pd
from tqdm import tqdm
from einops import rearrange
import argparse

from models.build import build_model


def load_model(model_type, work_dir, data_type):

    model = build_model(model_type, data_type)
    model = model.cuda()

    checkpoint = torch.load(work_dir)
    for key in list(checkpoint.keys()):
        checkpoint[key.replace('module.', '')] = checkpoint.pop(key)
    model.load_state_dict(checkpoint, strict=True)
    print('Succesfully loaded model.')

    return model

def test_model(model, test_loader, output_path):
    with open(output_path, "a") as log:
        model.eval()
        test_output, gt = [], []
        with torch.no_grad():
                for data, mean, imgname in tqdm(test_loader, desc='test'):
                    data = data.cuda()
                    mean = mean.cuda()
                    mean = rearrange(mean, 'b -> b 1')

                    output = model(data).cuda()
                    test_output.append(output)
                    gt.append(mean)

                    log.write('{},{}\n'.format(imgname[0], float(output[0])))


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='TIQA')

    parser.add_argument('--work_dir', type=str, required=True, help='checkpoint directory [Mayo|MRI|Phantom]')
    parser.add_argument('--output_path', type=str, required=True, default='./result/output.csv', help='directory to save the result file')
    parser.add_argument('--model_type', type=str, default='ediqa')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--data_type', type=str, default='mayo', help='data type [mayo|ncc|phantom|mri]')
    # data path
    parser.add_argument('--label_dir', type=str, default='/media/wonkyong/hdd2/data/D2IQA/mayo/mayo_test.csv')
    parser.add_argument('--img_path', type=str, default='/media/wonkyong/268c4ce6-d8a7-453b-9607-b1a121f48743/data/nimg-test-3channel/*/*/*.tiff')
    args = parser.parse_args()

    # Test settings
    batch_size = args.batch_size
    model = load_model(args.model_type, args.work_dir, args.data_type)

    # load test data
    label_dir = args.label_dir
    test_list = sorted(glob(args.img_path))
    if args.data_type == 'mayo':
        test_list = [t for t in test_list if '{}_{}_{}.0'.format(t.split('/')[-3], t.split('/')[-2], t.split('/')[-1].split('.')[0]) in pd.read_csv(args.label_dir)['img'].tolist()]
    elif args.data_type == 'phantom':
        test_list = [t for t in test_list if 'ge_chest_{}_{}'.format(t.split('/')[-1].split('_')[-1].split('.')[0], t.split('/')[-2]) in pd.read_csv(label_dir)['img'].tolist()]
    else: 
        test_list = [t for t in test_list if 'ABIDEII-GU_1_{}_session_1_anat_1_{}'.format(t.split('/')[-2], t.split('/')[-1].split('.')[0]) in pd.read_csv(args.label_dir)['img'].tolist()]
    
    test_data = EDIQAData(args.data_type, test_list, label_dir, mode='test')
    test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=False)

    test_model(model, test_loader, args.output_path)