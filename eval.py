import os
import argparse
from collections import Counter

import torch
from torchvision import transforms as vision_transforms
from tabulate import tabulate

from deepbrainprint.model import encoder
from deepbrainprint.dataset import ADNI2D
from deepbrainprint.augmentation import Gray2RGB
from deepbrainprint.sampler import single_scan_x_followup
from deepbrainprint.retrieval import images_to_embeddings, predicted_patients
from deepbrainprint import metrics
from deepbrainprint.utils import compute_statistics


def evaluate_model(model, test_set, device: str = 'cuda', metric: str = 'cosine'):
    """ Evaluate the model performances """
    embeddings = images_to_embeddings(model, test_set, device)         
    y_true, y_pred = predicted_patients(embeddings, test_set, k=5, metric=metric)
    return evaluate_predictions(y_true, y_pred)


def evaluate_predictions(y_true, y_pred):
    """ Compute mAP, recall, and accuracy a various k. """
    labels_count = Counter(y_true)
    mAP_3 = metrics.mean_average_precision(y_true, y_pred, 3, labels_count)
    R_3 = metrics.recall_at_k(y_true, y_pred, 3)
    return mAP_3, R_3
    
    
def beautify_metric(value):
    return f'{(value * 100):.2f} %'


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', help="Experiment output from train.py", type=str, required=True)
    parser.add_argument('--adni-dir', help="ADNI dataset dir", type=str, required=True)
    parser.add_argument('--synt-dir', help="SYNT-CONTR dataset dir", type=str, required=True)
    parser.add_argument('--cuda', help="run evaluation on GPU", type=bool, default=True)
    parser.add_argument('--best', help="evaluate the model with lowest validation loss", type=bool, default=True)
    args = parser.parse_args()

    print('EXP: ', args.exp_dir)
    
    assert os.path.exists(args.exp_dir), "Invalid experiment directory"
    assert os.path.exists(args.adni_dir), "Invalid ADNI dataset directory"
    assert os.path.exists(args.synt_dir), "Invalid SYNT-CONTR dataset directory"
    
    device = 'cuda' if args.cuda else 'cpu'
    model_filename = 'best_model.pth' if args.best else 'output_model.pth'
    state_dict_path = os.path.join(args.exp_dir, 'run-0', 'output_model.pth')
    model = encoder().to(device)
    model.load_state_dict(torch.load(state_dict_path))
    
    datasets = {
        'ADNI':         (args.adni_dir), 
        'SYNT-CONTR':   (args.synt_dir) 
    }

    tabulate_data = []

    for dataset_type, dataset_path in datasets.items():
        
        transforms = vision_transforms.Compose([
            vision_transforms.ToTensor(), 
            vision_transforms.Normalize(*compute_statistics(dataset_path)),
            Gray2RGB(),
        ])
        
        testset = ADNI2D(dataset_path, train=False, transform=transforms)
        testset = single_scan_x_followup(testset, followup_days=180)

        print('evaluating on', dataset_type)
        current_tabulate_data = []
        mAP_3, R_3 = evaluate_model(model, testset, device) 
        current_tabulate_data.append(dataset_type)                                     
        current_tabulate_data.append(beautify_metric(mAP_3))
        current_tabulate_data.append(beautify_metric(R_3))
        tabulate_data.append(current_tabulate_data)

    headers = ["dataset", "mAP@3", "R@3",]
    print('\n')
    print(tabulate(tabulate_data, headers=headers, tablefmt='grid'), "\n")