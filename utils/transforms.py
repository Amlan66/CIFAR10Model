import numpy as np
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

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
            A.ShiftScaleRotate(shift_limit=0.125, scale_limit=0.15, rotate_limit=15, p=0.5,border_mode=cv2.BORDER_REPLICATE),
            A.CoarseDropout(
                max_holes=1,
                max_height=8,
                max_width=8,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=mean,
                mask_fill_value=None,
                p=0.4
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
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