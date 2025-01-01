import numpy as np
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        # Convert PIL image to numpy array
        img = np.array(img)
        # Apply Albumentations transforms
        augmented = self.transform(image=img)
        return augmented['image']

def get_transforms(mean, std):
    train_transform = AlbumentationsTransform(
        A.Compose([
            A.Normalize(mean=mean, std=std),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, 
                           min_height=16, min_width=16, fill_value=mean, mask_fill_value=None),
            ToTensorV2(),
        ])
    )

    test_transform = AlbumentationsTransform(
        A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    )

    return train_transform, test_transform 