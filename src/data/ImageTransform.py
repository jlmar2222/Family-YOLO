import albumentations as A
from albumentations.pytorch import ToTensorV2



def get_train_transforms(input_size: int):

    return A.Compose([
        A.RandomResizedCrop(
            size=(input_size, input_size),
            scale=(0.8, 1.0),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.Affine(  # sustituye a Rotate suelto, más usado en detección
            translate_percent=0.1,
            scale=(0.9, 1.1),
            rotate=(-10, 10),
            p=0.5
        ),
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1,
            p=0.5
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))



def get_val_transforms(input_size: int):
    return A.Compose([
        A.Resize(height=input_size, width=input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['class_labels'],
        min_visibility=0.0  # en validación normalmente NO quieres filtrar cajas
    ))