from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import torchvision
import torchvision.transforms as T
import torch


__DATASET__ = {}


class CenterCropLongEdge(object):
    """Crops the given PIL Image on the long edge."""

    def __call__(self, img):
        return torchvision.transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int,
                   num_workers: int,
                   train: bool):
    dataloader = DataLoader(dataset,
                            batch_size,
                            shuffle=train,
                            num_workers=num_workers,
                            drop_last=train)
    return dataloader


class _RGBImageFolder(VisionDataset):
    """Folder of RGB images returning a single (optionally normalized) tensor.

    Used for the synthetic FFHQ / ImageNet validation sets.
    """
    _PATTERN = '/**/*.png'

    def __init__(self, root, transforms=None, image_size=256, normalize=True):
        super().__init__(root, transforms)
        self.fpaths = sorted(glob(root + self._PATTERN, recursive=True))
        assert len(self.fpaths) > 0, f"No images found under {root} ({self._PATTERN})."

        if transforms is not None:
            self.transforms = transforms
        else:
            norm_mean = [0.5, 0.5, 0.5]
            norm_std = [0.5, 0.5, 0.5]
            ops = [CenterCropLongEdge(),
                   torchvision.transforms.Resize(image_size, T.InterpolationMode.BICUBIC),
                   torchvision.transforms.ToTensor()]
            if normalize:
                ops.append(torchvision.transforms.Normalize(norm_mean, norm_std))
            self.transforms = torchvision.transforms.Compose(ops)

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index):
        img = Image.open(self.fpaths[index]).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


@register_dataset(name='ffhq')
class FFHQDataset(_RGBImageFolder):
    _PATTERN = '/**/*.png'


@register_dataset(name='imagenet')
class ImageNetDataset(_RGBImageFolder):
    _PATTERN = '/**/*.JPEG'


@register_dataset(name='lensless_ys')
class LenslessYSDataset(VisionDataset):
    """Real SNU/Yonsei Voronoi lensless data (MirFlickr captures).

    Returns (raw measurement tiff, ground-truth jpg). The raw measurement is
    kept unscaled; downstream `transform_padding_centercrop_real` reformats it
    onto the 512x512 capture grid.
    """

    def __init__(self, root: str, transforms: Optional[Callable] = None):
        super().__init__(root, transforms)
        self.lqpath = sorted(glob(root + '/raw/*.tiff', recursive=True))
        self.gtpath = sorted(glob(root + '/label/*.jpg', recursive=True))

    def __len__(self):
        return len(self.lqpath)

    def __getitem__(self, index: int):
        lq = cv2.imread(self.lqpath[index], -1).astype(np.float32)
        lq = cv2.cvtColor(lq, cv2.COLOR_BGR2RGB)
        gt = Image.open(self.gtpath[index]).convert('RGB')
        if self.transforms is not None:
            lq = self.transforms(lq)
            gt = self.transforms(gt)
        return lq, gt


@register_dataset(name='lensless_real_waller')
class LenslessWallerDataset(VisionDataset):
    """Real Waller-lab DiffuserCam data.

    Returns (measurement, ground-truth) pairs, each resized to 256x256 and
    normalized to [-1, 1].
    """

    def __init__(self, root: str, transforms: Optional[Callable] = None):
        super().__init__(root, transforms)
        self.lqpath = sorted(glob(root + '/LQ/*.png', recursive=True))
        self.gtpath = sorted(glob(root + '/GT/*.png', recursive=True))

        image_size = 256
        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size), T.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(norm_mean, norm_std),
        ])

    def __len__(self):
        return len(self.lqpath)

    def __getitem__(self, index: int):
        lq = Image.open(self.lqpath[index]).convert('RGB')
        gt = Image.open(self.gtpath[index]).convert('RGB')
        if self.transforms is not None:
            lq = self.transforms(lq)
            gt = self.transforms(gt)
        return lq, gt
