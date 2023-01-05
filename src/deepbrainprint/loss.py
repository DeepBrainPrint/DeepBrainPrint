import torch
from torch import nn
from pytorch_metric_learning.losses import NTXentLoss
from deepbrainprint.beta_scheduler import BetaScheduler



class BarlowTwinsLoss:
    """
    Loss depicted in https://github.com/facebookresearch/barlowtwins
    """
    
    def __init__(self, embedding_size=2048, lambd=0.0051):
        self.lambd = lambd
        self.bn = nn.BatchNorm1d(embedding_size, affine=False) # turn off the learnable parameters of BN.
        
    def __call__(self, emb1, emb2):
        if emb1.shape[0] == 0 or emb2.shape[0] == 0:
            return torch.tensor(0.)        
        CC = self.bn(emb1).T @ self.bn(emb2)
        CC.div_(len(emb1))
        diag_elements_loss = torch.diagonal(CC).add_(-1).pow(2).sum()
        offd_elements_loss = self.off_diagonal(CC).pow(2).sum()
        loss = diag_elements_loss + self.lambd * offd_elements_loss
        return loss        

    def off_diagonal(self, x):
        """ 
        return a flattened view of the off-diagonal elements of a square matrix
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    
    
class DeepBrainPrintLoss:
    
    def __init__(self, 
                 beta_scheduler: BetaScheduler, 
                 embedding_size: int, 
                 _temperature: float = 0.07,
                 _lambda: float = 0.0051):
        
        self.beta_scheduler = beta_scheduler
        self.self_superv_loss = BarlowTwinsLoss(embedding_size, _lambda)
        self.contrastive_loss = NTXentLoss(_temperature)
        
    def __call__(self, 
                 epoch, 
                 embeddings_T1, 
                 embeddings_T2, 
                 labels, 
                 indices_tuple):
        
        beta = self.beta_scheduler(epoch) # type:ignore
        l_nt = self.contrastive_loss(embeddings_T1, labels, indices_tuple=indices_tuple)
        l_bt = self.self_superv_loss(embeddings_T1, embeddings_T2)
        loss = beta * l_bt + (1 - beta) * (l_nt * 1e+4)
        return loss
        
    def set_device(self, device: str):
        self.self_superv_loss.bn = self.self_superv_loss.bn.to(device) 