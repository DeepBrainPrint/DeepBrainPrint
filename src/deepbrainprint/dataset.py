import os
from typing import Optional, Callable

import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from sklearn.preprocessing import LabelEncoder
from skimage import io


class ADNI2D(Dataset):
    """
    DeepBrainPrint ADNI dataset has been preprocessed as described in the paper, 
    extracting 2D slices from 3D volumes. The csv called "annotations" contains
    the filename of each PNG, the patient ID and the scan date (for sampling). 
    """
    
    def __init__(self, 
                root_dir: str, 
                train: bool = True, 
                transform: Optional[Callable] = None
    ) -> None:
        self.root_dir = os.path.join(root_dir, 'train' if train else 'test')
        self.transform = transform
        self.train = train
        df = pd.read_csv(os.path.join(self.root_dir, 'annotations.csv'))
        self.annotations, self.label_encoder = self.patients_to_categorical(df)
        self.annotations['scan_date'] = pd.to_datetime(self.annotations['scan_date'], format='%Y%m%d')
        self.returns_date = False

    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, index):
        filename = self.annotations.iloc[index].top         # type: ignore 
        filepath = os.path.join(self.root_dir, filename)
        image = io.imread(filepath)
        label = self.annotations.iloc[index].label          # type: ignore
        sdate = self.annotations.iloc[index]['scan_date']   # type: ignore
        if self.transform: image = self.transform(image)
        return (image, label, sdate) if self.returns_date else (image, label) 
        
    def nlabels(self) -> int:
        return self.annotations.label.max() + 1

    def patient(self, index: int) -> int:
        return self.annotations.iloc[index]['label']        # type: ignore
    
    def patients_to_categorical(self, df: pd.DataFrame):
        le = LabelEncoder()
        le.fit(df.patient)
        df['label'] = le.transform(df.patient)
        df = df.astype({'label': int})
        return (df, le)
    
    
    
class AugmentedPairWrapper(Dataset):
    
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img1, label = self.dataset[index]
        img2, label = self.dataset[index]
        return img1, img2, label