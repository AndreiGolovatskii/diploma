import albumentations as A
import cv2

def transforms():
    return A.Compose([
        A.RandomBrightnessContrast(always_apply=True),
        A.RandomSunFlare(always_apply=True),
        A.RandomRotate90(p=0.5),
        A.MotionBlur(always_apply=True),
        A.RandomRain(always_apply=True),
        A.OpticalDistortion(always_apply=True),
        A.SafeRotate(limit=3,always_apply=True,border_mode=cv2.BORDER_CONSTANT)
    ])

