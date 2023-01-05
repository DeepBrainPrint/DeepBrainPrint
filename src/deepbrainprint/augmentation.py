import random
from torchvision import transforms as vision_transforms
from monai import transforms as monai_transforms


class Gray2RGB(object):
    def __call__(self, img):
        return img.repeat(3, 1, 1)


def negative():
    """
    Transform an image using negative point-function
    """
    def _negative(tensor):        
        _min = tensor.min()
        _max = tensor.max()
        tensor_01 = (tensor - _min) / (_max - _min)
        tensor_01_neg = 1 - tensor_01
        tensor = (tensor_01_neg * (_max - _min)) + _min
        return tensor
    
    return vision_transforms.Lambda(_negative)


def random_negative(prob: float):
    """
    Apply `negative` transform randomly
    """
    return vision_transforms.RandomApply([ negative() ], p=prob)


def domain_specific_augmentation():
    """
    Augmentation (image distortion) proposed in our paper
    """
    return vision_transforms.Compose([
        random_negative(prob=0.4),
        vision_transforms.RandomRotation(degrees=3), 
        monai_transforms.RandShiftIntensity(offsets=0.25, prob=0.4),                            # type: ignore 
        monai_transforms.RandBiasField(prob=0.3),                                               # type: ignore
        monai_transforms.RandCoarseDropout(holes=3, spatial_size=(10,10), prob=.4),             # type: ignore
        monai_transforms.Rand2DElastic(spacing=(30, 30), magnitude_range=(1., 2.), prob=.3)     # type: ignore        
    ])