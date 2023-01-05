from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from pytorch_metric_learning.utils.common_functions import Identity
from brainfetch.utils import common_functions as cf


def encoder(unfreeze_bn=True):
    """
    Encoder R for extracting image representations. 
    """
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

    cf.freeze_module(resnet.conv1)
    cf.freeze_module(resnet.layer1)
    if unfreeze_bn:
        cf.unfreeze_module(resnet.layer1[0].bn1)  # type: ignore
        cf.unfreeze_module(resnet.layer1[0].bn2)  # type: ignore

    cf.freeze_module(resnet.layer2)
    if unfreeze_bn:
        cf.unfreeze_module(resnet.layer2[0].bn1)  # type: ignore
        cf.unfreeze_module(resnet.layer2[0].bn2)  # type: ignore

    resnet.fc = Identity()  # type: ignore
    return resnet


class ProjectionHead(nn.Module):
    """
    Projection head P used during the training. 
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()        
        self.f = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False), 
            nn.BatchNorm1d(output_dim), 
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.f(x)


class Embedder(nn.Module):
    """
    Composition of the encoder R and the projection head P.
    """
    def __init__(self):
        super().__init__()
        self.encoder = encoder()
        self.projection_head = ProjectionHead(input_dim=512, output_dim=2048)

    def forward(self, x):
        return self.projection_head(self.encoder(x))
    
    def get_encoder(self):
        return self.encoder