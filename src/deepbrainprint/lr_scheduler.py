import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmup(_LRScheduler):
    """
    Cosine Annealing scheduler with initiali linear warmup.
    The step() function must be called after each step (batch processed). 
    """
    
    def __init__(self,
                 optimizer: Optimizer, 
                 total_epochs: int, 
                 batches_in_epoch: int,
                 batch_size: int, 
                 warmup_epochs: int = 10, 
                 batch_lr_scaling: bool = True):
        self.total_epochs = total_epochs
        self.batches_in_epoch = batches_in_epoch
        self.warmup_epochs = warmup_epochs
        self.max_steps = total_epochs * batches_in_epoch
        self.warmup_steps = 10 * batches_in_epoch
        self.steps_after_warmup = self.max_steps - self.warmup_steps
        self.origin_param_groups = [ group.copy() for group in optimizer.param_groups ]
        # learning rate values in the paper refer to a training made with batch 
        # size equal to 256, if we increase / decrease the batch size, we must
        # linearly increase or decrease the learning rate (see Krizhevsky, 2014)
        self.lr_linear_scaling = batch_size / 256 if batch_lr_scaling else 1.  
        super(CosineAnnealingWarmup, self).__init__(optimizer, last_epoch=-1, verbose=False) # type:ignore
        
        
    def get_lr(self):
        if self._step_count < self.warmup_steps: # type:ignore
            lr_factor =  (self._step_count / self.warmup_steps) * self.lr_linear_scaling # type:ignore
        else:
            ca_step = self._step_count - self.warmup_steps # type:ignore
            q = 0.5 * (1 + np.cos(np.pi * ca_step / self.steps_after_warmup))
            end_lr = self.lr_linear_scaling * 0.001
            lr_factor = self.lr_linear_scaling * q + end_lr * (1 - q)
        
        # we need to keep a copy of the original learning rates because this 
        # return statement will update the optimizer learning rates, and we need
        # the reference to the starting values. 
        return [ group['lr'] * lr_factor for group in self.origin_param_groups ]
        