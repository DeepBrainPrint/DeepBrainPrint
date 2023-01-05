import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms as vision_transforms
from pytorch_metric_learning.miners import BatchEasyHardMiner
from tqdm import tqdm

from deepbrainprint.model import Embedder
from deepbrainprint.loss import DeepBrainPrintLoss
from deepbrainprint.optim import LARS
from deepbrainprint.lr_scheduler import CosineAnnealingWarmup
from deepbrainprint.dataset import ADNI2D, AugmentedPairWrapper
from deepbrainprint.augmentation import Gray2RGB, domain_specific_augmentation
from deepbrainprint.sampler import single_scan_x_followup
from deepbrainprint.utils import adni_train_val_split, AverageValueMeter, SaveBestModel, EarlyStopping, compute_statistics
from deepbrainprint.archive import ExperimentArchive
from deepbrainprint.beta_scheduler import LinearBetaScheduler, IterBetaScheduler, StepBetaScheduler, ConstantBetaScheduler



def get_optimizer(model, weights_lr=0.2, bias_lr=0.0048, weight_decay=1e-6):
    param_biases = []
    param_weights = []
    for param in model.parameters():
        if param.ndim == 1: param_biases.append(param)
        else: param_weights.append(param)                
    
    parameters = [
        {'params': param_weights, 'lr': weights_lr }, 
        {'params': param_biases,  'lr': bias_lr }
    ]
        
    return LARS(parameters,
                lr=0.2, 
                weight_decay=weight_decay, 
                weight_decay_filter=True, 
                lars_adaptation_filter=True)              


def get_scheduler(optimizer, epochs, batch_size, batches_in_epochs) -> CosineAnnealingWarmup:
    return CosineAnnealingWarmup(optimizer,
                                 total_epochs=epochs, 
                                 batch_size=batch_size, 
                                 batches_in_epoch=batches_in_epochs, 
                                 warmup_epochs=10,
                                 batch_lr_scaling=True)


if __name__ == '__main__':
        
    EXP_NAME        = 'deepbrainprint-exp'
    BATCH_SIZE      = 64
    LOADER_WORKERS  = 4
    PIN_MEMORY      = True
    EPOCHS          = 180
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=EXP_NAME, help='output dir name')
    parser.add_argument('--dataset-dir', type=str, required=True, help='ADNI dataset dir')
    parser.add_argument('--output-dir', type=str, required=True, help='output dir')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--workers', type=int, default=LOADER_WORKERS, help='dataloaders workers')
    parser.add_argument('--pin-memory', type=bool, default=True, help='pin memory (GPU)')
    parser.add_argument('--cuda', type=bool, default=True, help='train on GPU')
    parser.add_argument('--sched', type=str, default='linear', help='weighting function (see paper appendix B.1)')
    parser.add_argument('--sched-hp', default=None, help='optional scheduler hyperparameter')
    args = parser.parse_args()
    
    assert args.sched in ['linear', 'step', 'iter', 'const'], "Invalid beta scheduler"
    
    print('\n🔌 starting deepbrainprint training with the following settings.\n')
    print(''.join(f'{k}={v}\n' for k, v in vars(args).items()))
    print('-' * 50)
    
    transforms = vision_transforms.Compose([
        vision_transforms.ToTensor(), 
        vision_transforms.Normalize(*compute_statistics(args.dataset_dir)),
        Gray2RGB(),
        domain_specific_augmentation()
    ])

    train_set = ADNI2D(args.dataset_dir, True, transforms)
    train_set = single_scan_x_followup(train_set, turnoff_return_date=True)
    train_set, valid_set = adni_train_val_split(train_set, .10)
    
    train_set = AugmentedPairWrapper(train_set)
    valid_set = AugmentedPairWrapper(valid_set)

    train_loader = DataLoader(train_set, 
                              batch_size=args.batch_size, 
                              shuffle=True,  
                              num_workers=args.workers,
                              pin_memory=args.pin_memory)
    
    valid_loader = DataLoader(valid_set, 
                              batch_size=args.batch_size, 
                              shuffle=False, 
                              num_workers=args.workers,
                              pin_memory=args.pin_memory)

    beta_scheduler = LinearBetaScheduler(args.epochs)   # default scheduler
    
    if args.sched == 'step':
        assert args.sched_hp is not None, "hyperparameter not set"
        beta_scheduler = StepBetaScheduler(args.sched_hp)
    
    elif args.sched == 'iter':
        assert args.sched_hp is not None, "hyperparameter not set"
        beta_scheduler = IterBetaScheduler(args.sched_hp)
    
    elif args.sched == 'const':
        beta_scheduler = ConstantBetaScheduler(0.5)

    device = 'cuda' if args.cuda else 'cpu'
    model = Embedder().to(device)
    criterion = DeepBrainPrintLoss(beta_scheduler, 2048)
    criterion.set_device(device)
    semihard_miner = BatchEasyHardMiner(neg_strategy=BatchEasyHardMiner.SEMIHARD)

    optim = get_optimizer(model)
    sched = get_scheduler(optim, args.epochs, args.batch_size, len(train_loader))
        
    archive = ExperimentArchive(args.exp_name)
    archive.set_metadata({
        'epochs': args.epochs, 
        'batch_size': args.batch_size,
    })
    archive.set_current_directory(args.output_dir)
    archive.setup()
    archive.create_run_directory(0)

    emerg_stop_path = archive.get_emergency_stop_path(0)
    train_loss_meter = AverageValueMeter()
    valid_loss_meter = AverageValueMeter()
    
    loaders     = { 'train': train_loader,      'valid': valid_loader }
    loss_meters = { 'train': train_loss_meter,  'valid': valid_loss_meter }
    
    best_model_saver = SaveBestModel(model)
    early_stopping = EarlyStopping(model, patience=7)
    writer = SummaryWriter(archive.get_tensorboard_log_path(0))
    
    
    for epoch in tqdm(range(args.epochs)):
        
        if archive.emergency_stop_triggered(run_index=0): 
            print('emergency stop triggered.')
            break
        
        train_loss_meter.reset()
        valid_loss_meter.reset()
    
        for mode in ['train', 'valid']:    
            loader = loaders[mode]
            loss_meter = loss_meters[mode]
    
            for batch_images_1, batch_images_2, batch_patients in loader:
                batch_images_1 = batch_images_1.to(device)
                batch_images_2 = batch_images_2.to(device)
                batch_patients = batch_patients.to(device)
                
                with torch.set_grad_enabled(mode == 'train'):
                    
                    embeddings_1 = model(batch_images_1)
                    embeddings_2 = model(batch_images_2)
                    indices_tuple = semihard_miner(embeddings_1, batch_patients)

                    loss = criterion(epoch=epoch, 
                                     embeddings_T1=embeddings_1, 
                                     embeddings_T2=embeddings_2, 
                                     labels=batch_patients, 
                                     indices_tuple=indices_tuple)
                    
                    loss_meter.add(loss.item())

                    if mode == 'train':
                        loss.backward()
                        optim.step()
                        optim.zero_grad()
                        sched.step()
                    
        avg_valid_loss = valid_loss_meter.value()
        best_model_saver.update(avg_valid_loss)
        
        writer.add_scalar('loss/train', train_loss_meter.value(), global_step=epoch)
        writer.add_scalar('loss/valid', valid_loss_meter.value(), global_step=epoch)
    
        if early_stopping.stop_training(avg_valid_loss):
            model = early_stopping.get_backup_model()
            break
    
    archive.save_model(0, model.get_encoder()) # type: ignore
    archive.save_model(0, best_model_saver.get().get_encoder(), 'best_model') # type: ignore