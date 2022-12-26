import albumentations as A
import cv2

def transforms():
    return A.Compose([
        A.RandomBrightnessContrast(always_apply=True),
        A.RandomSunFlare(src_radius=120,p=60),
        A.RandomContrast(always_apply=True),
        A.Flip(),
        A.MotionBlur(always_apply=True),
        A.RandomRain(always_apply=True),
        A.OpticalDistortion(always_apply=True),
        A.SafeRotate(limit=3,always_apply=True,border_mode=cv2.BORDER_CONSTANT),
        A.ImageCompression(quality_lower=60, quality_upper=100, always_apply=True),
        A.ISONoise(always_apply=True),
        A.ElasticTransform(sigma=20, alpha_affine=30, border_mode=cv2.BORDER_CONSTANT, always_apply=True)
    ])

