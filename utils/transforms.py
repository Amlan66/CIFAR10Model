import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(mean, std):
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.CoarseDropout(
            max_holes=1, max_height=16, max_width=16,
            min_holes=1, min_height=16, min_width=16,
            fill_value=mean, mask_fill_value=None,
            p=0.5
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    return train_transform, test_transform 