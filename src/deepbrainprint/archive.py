import csv
import os

import torch


class ExperimentArchive:
    
    def __init__(self, expname: str):
        self.expname = self.generate_valid_expname(expname)
        self.currdir = '.'
        self.metadata = {}

    @property
    def experiment_dir(self) -> str:
        return os.path.join(self.currdir, self.expname)

    @property
    def metadata_path(self) -> str:
        return os.path.join(self.experiment_dir, 'metadata.csv')

    def setup(self) -> None:
        self.create_exp_directory()
        self.create_metadata_csv()
        
    def generate_valid_expname(self, expname: str) -> str:
        return expname.replace(' ', '-')

    def set_current_directory(self, currdir: str) -> None:
        assert os.path.exists(currdir), "path doesn't exists."
        self.currdir = currdir

    def set_metadata(self, metadata: dict) -> None:
        self.metadata = metadata

    def create_exp_directory(self) -> None:
        assert not os.path.exists(self.experiment_dir), "exp. directory exists."
        os.makedirs(self.experiment_dir)
        
    def create_metadata_csv(self) -> None:
        with open(self.metadata_path, 'w') as f:  
            w = csv.DictWriter(f, self.metadata.keys())
            w.writeheader()
            w.writerow(self.metadata)
            
    def create_run_directory(self, run_index: int) -> None:
        run_dir = self.get_run_directory_path(run_index)
        os.makedirs(run_dir)
        self.create_emergency_stop(run_index)
        
    def create_emergency_stop(self, run_index: int) -> None:
        run_dir = self.get_run_directory_path(run_index)
        emergency_stop_path = os.path.join(run_dir, '.emergency.stop')
        open(emergency_stop_path, 'a').close()
        
    def get_run_directory_path(self, run_index: int) -> str:
        return os.path.join(self.experiment_dir, f'run-{run_index}')
    
    def get_emergency_stop_path(self, run_index: int) -> str:
        run_path = self.get_run_directory_path(run_index)
        return os.path.join(run_path, '.emergency.stop')
    
    def emergency_stop_triggered(self, run_index: int) -> bool:
        return not os.path.exists(self.get_emergency_stop_path(run_index))
    
    def get_tensorboard_log_path(self, run_index: int) -> str:
        assert os.path.exists(self.get_run_directory_path(run_index)), "run doesn't exists."    
        return os.path.join(self.get_run_directory_path(run_index), 'tensorboard')            

    def save_model(self, run_index: int, model: torch.nn.Module, filename='output_model') -> None:
        assert os.path.exists(self.get_run_directory_path(run_index)), "run doesn't exists."
        model_path = os.path.join(self.get_run_directory_path(run_index), f'{filename}.pth')
        torch.save(model.state_dict(), model_path)
    
    def save_model_evaluation(self, run_index: int, evaluation_dict: dict, evalname: str = 'evaluation'):
        assert os.path.exists(self.get_run_directory_path(run_index)), "run doesn't exists."
        evaluation_csv_path = os.path.join(self.get_run_directory_path(run_index), f'{evalname}.csv')
        with open(evaluation_csv_path, 'w') as f:  
            w = csv.DictWriter(f, evaluation_dict.keys())
            w.writeheader()
            w.writerow(evaluation_dict)