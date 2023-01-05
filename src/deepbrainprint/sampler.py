from copy import deepcopy
from torch.utils.data import Dataset
from deepbrainprint.dataset import ADNI2D


class ScanInfo:    
    def __init__(self, idx, patient, date):
        self.idx = idx
        self.patient = patient
        self.date = date


class ScanInfoDatabase:
    
    def __init__(self):
        self.db = {}
        
    def add(self, si: ScanInfo):
        if si.patient not in self.db:
            self.db[si.patient] = []
        self.db[si.patient].append(si)
        
    def from_dataset(self, dataset: Dataset):
        # fast loading.        
        for index in range(len(dataset)):                               # type: ignore
            scan_date = dataset.annotations.iloc[index]['scan_date']    # type: ignore
            patient = dataset.annotations.iloc[index].label             # type: ignore
            si = ScanInfo(index, patient, scan_date)
            self.add(si)
            
    def get_scans(self, patient: int):
        assert patient in self.db, "patient doesn't exists"
        return self.db[patient]
    
    def get_patients_list(self):
        return list(self.db.keys())


def single_scan_x_followup(dataset: ADNI2D, 
                           turnoff_return_date: bool = True, 
                           followup_days: int = 180) -> ADNI2D:
    """
    Sample the dataset such that the resulting subset
    contains one scan for each followup period (180 days).
    """
    dataset.returns_date = True
    infodb = ScanInfoDatabase()
    infodb.from_dataset(dataset)
    subset_indices = []

    for patient in infodb.get_patients_list():
        patient_scans = infodb.get_scans(patient)
        scans_dates = [ s.date for s in patient_scans ]
        scans_dates.sort()

        first_scan_date = scans_dates[0]
        last_scan_date = scans_dates[-1]
        days_in_between = (last_scan_date - first_scan_date).days

        # If every scan from the patient hits the same
        # follow-up, then we will take only 1 scan. Since a 
        # keeping single scan is useless, we ignore the patient.
        if days_in_between < followup_days: continue

        selected_scans = []
        quantization_steps = list(range(0, days_in_between, followup_days)) + [days_in_between]

        # Split the scans timeline in 180-days bins
        # and take only 1 scan for each bin. 
        for i in range(1, len(quantization_steps)):
            curr_Q = quantization_steps[i]
            prev_Q = quantization_steps[i-1]
            for idx, si in enumerate(patient_scans):
                delta = (si.date - first_scan_date).days
                if delta >= prev_Q and delta <= curr_Q:
                    selected_scans.append(si)
                    break

        subset_indices += [ s.idx for s in selected_scans ]
    
    if turnoff_return_date: dataset.returns_date = False
    
    # We cannot use subset here because we sample the dataset 
    # before train/valid splitting. The latter process recursively
    # goes up to the first instance of the dataset that isn't a Subset.
    # If we set this as a subset, then the recursive process will go
    # up to the unsampled dataset. 
        
    dataset_copy = deepcopy(dataset)
    dataset_copy.annotations = dataset.annotations.iloc[subset_indices]
    return dataset_copy