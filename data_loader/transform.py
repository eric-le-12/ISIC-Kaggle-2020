from torchvision import transforms
from torchtoolbox.transform.cutout import Cutout_PIL as Cutout
# define augmentation methods for training and validation/test set
from albumentations import *
from albumentations.pytorch import ToTensor

train_transform = Compose(
    [  Blur(p=0.5,blur_limit=3),
        VerticalFlip(p=.5),
       HorizontalFlip(p=.5),
       HueSaturationValue(),
    ]
)

val_transform = Compose(
    [transforms.ToTensor()]
)
