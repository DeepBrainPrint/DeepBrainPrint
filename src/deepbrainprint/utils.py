from copy import deepcopy
from typing import Optional

import torch
import numpy as np
from torch import nn
from torch.utils.data import Subset
from torchvision import transforms as vision_transforms 

from deepbrainprint.dataset import ADNI2D


def rec_search_orig_dataset(_dataset):
        if isinstance(_dataset, Subset):
            return rec_search_orig_dataset(_dataset.dataset)
        return _dataset


def group_mris_by_patient(dataset):
    """
    Fast grouping of the dataset MRIs by patient label.
    Works both on ADNI datasets and Subsets.
    """    
    ref_dataset = rec_search_orig_dataset(dataset)
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(dataset))
    
    patient_dict = {}
    for dataset_idx, real_idx in enumerate(indices):
        patient = ref_dataset.annotations.iloc[real_idx].label
        if patient not in patient_dict:
            patient_dict[patient] = {dataset_idx}
        else:
            patient_dict[patient].add(dataset_idx)    
    return patient_dict


def adni_train_val_split(dataset: ADNI2D, validation_perc: float):
    """ 
    Split the dataset into a training set and a validation set. 
    The validation set contains ~ `validation_perc`% of the dataset.
    The function ensures that if RMIs from a patient are contained 
    in one dataset, then all the RMIs of the patient are contained 
    in the dataset. 
    """
    assert validation_perc > 0 and validation_perc < 1, "Invalid validation percentage"
    patient_dict = group_mris_by_patient(dataset)
    validation_len = np.floor(len(dataset) * validation_perc)
    validation_indices = set()
    for indices in patient_dict.values():
        validation_indices = validation_indices.union(indices)
        if len(validation_indices) >= validation_len: break
    training_indices = [ i for i in range(len(dataset)) if i not in validation_indices ]
    return Subset(dataset, training_indices), Subset(dataset, list(validation_indices))


def compute_statistics(dataset_dir: str):
    """
    Computes dataset intensities mean and std
    """
    mean, std = 0, 0
    train_set = ADNI2D(dataset_dir, train=True, transform=vision_transforms.ToTensor())
    with torch.no_grad():
        for img, _ in train_set: # type: ignore
            mean += img.mean().item()
            std  += img.std().item()
    mean /= len(train_set)
    std  /= len(train_set)
    return mean, std


class AverageValueMeter:
    
    def __init__(self):
        self.reset()
        
    def reset(self) -> None:
        self.sum = 0
        self.num = 0
    
    def add(self, value: float) -> None:
        self.sum += value
        self.num += 1

    def value(self) -> float:
        if self.num == 0: return 0
        return self.sum / self.num


class EarlyStopping:
    
    """
    Stop training if the loss increases for the last
    `patience` epochs.
    """
    
    def __init__(self, model: nn.Module, patience: int = 5):
        self.patience = patience
        self.exp_model = model
        self.exp_model_device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
        self.backup_model = None
        self.loss_queue = []
        
    def stop_training(self, loss: float) -> bool:
        if len(self.loss_queue) < 1 or loss >= self.loss_queue[-1]:
            self.loss_queue.append(loss)
        else:
            self.loss_queue = []
            self.backup_model = deepcopy(self.exp_model).to('cpu')
            
        return len(self.loss_queue) >= self.patience
    

    def get_backup_model(self) -> nn.Module:
        assert self.backup_model is not None
        return self.backup_model.to(self.exp_model_device)
    
    
class SaveBestModel:
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.best_model = None
        self.best_loss = None
        self.model_device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
        
    def update(self, new_loss: Optional[float]) -> None: 
        if new_loss is None: return
        if self.best_loss is None or new_loss < self.best_loss:
            self.best_model = deepcopy(self.model).to('cpu')
            self.best_loss = new_loss
            
    def get(self) -> nn.Module:
        assert self.best_model is not None
        return self.best_model.to(self.model_device)