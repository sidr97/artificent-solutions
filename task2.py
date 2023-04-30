"""
Use the following augmentation methods on the sample image under data/sample.png
and save the result under this path: 'data/sample_augmented.png'

Note:
    - use torchvision.transforms
    - use the following augmentation methods with the same order as below:
        * affine: degrees: ±5, 
                  translation= 0.1 of width and height, 
                  scale: 0.9-1.1 of the original size
        * rotation ±5 degrees,
        * horizontal flip with a probablity of 0.5
        * center crop with height=320 and width=640
        * resize to height=160 and width=320
        * color jitter with:  brightness=0.5, 
                              contrast=0.5, 
                              saturation=0.4, 
                              hue=0.2
    - use default values for anything unspecified
"""

import torch
from torchvision import transforms as T
import numpy as np
import cv2
import random

torch.manual_seed(8)
np.random.seed(8)

img = cv2.imread('data/sample.png')

# write your code here ...
tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

given_transforms = torch.nn.Sequential(
    T.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.9, 1.1)),
    T.RandomRotation(degrees=(-5, 5)),
    T.RandomHorizontalFlip(p=0.5),
    T.CenterCrop(size=(320, 640)),
    T.Resize(size=(160, 320)),
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.2)
)

res = given_transforms(tensor)

cv2.imwrite('data/augmented_image.png', res[0].permute(1, 2, 0).numpy())
