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
    """Crops the given PIL Image on the long edge.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
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

"""
@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img
"""


@register_dataset(name='ffhq')
class ImageDataset(VisionDataset):

    def __init__(self,
                 root,
                 transforms=None,
                 image_size=256,
                 normalize=True):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))

        if transforms is not None:
            self.transforms = transforms
        else:
            norm_mean = [0.5, 0.5, 0.5]
            norm_std = [0.5, 0.5, 0.5]
            if normalize:
                self.transforms = torchvision.transforms.Compose([
                    CenterCropLongEdge(),
                    torchvision.transforms.Resize(image_size,  T.InterpolationMode.BICUBIC),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(norm_mean, norm_std)
                ])
            else:
                self.transforms = torchvision.transforms.Compose([
                    CenterCropLongEdge(),
                    torchvision.transforms.Resize(image_size,T.InterpolationMode.BICUBIC),
                    torchvision.transforms.ToTensor()
                ])

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        #print ("before transform",img.shape)

        # transform
        if self.transforms is not None:
            img = self.transforms(img)
            #print ("after transform",img.shape)

        return img

    
@register_dataset(name='Flicker')
class ImageDataset(VisionDataset):

    def __init__(self,
                 root,
                 transforms=None,
                 image_size=256,
                 normalize=True):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))

        if transforms is not None:
            self.transforms = transforms
        else:
            norm_mean = [0.5, 0.5, 0.5]
            norm_std = [0.5, 0.5, 0.5]
            if normalize:
                self.transforms = torchvision.transforms.Compose([
                    CenterCropLongEdge(),
                    torchvision.transforms.Resize(image_size,  T.InterpolationMode.BICUBIC),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(norm_mean, norm_std)
                ])
            else:
                self.transforms = torchvision.transforms.Compose([
                    CenterCropLongEdge(),
                    torchvision.transforms.Resize(image_size,T.InterpolationMode.BICUBIC),
                    torchvision.transforms.ToTensor()
                ])

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        #print ("before transform",img.shape)

        # transform
        if self.transforms is not None:
            img = self.transforms(img)
            #print ("after transform",img.shape)

        return img

@register_dataset(name='imagenet')
class ImageDataset(VisionDataset):

    def __init__(self,
                 root,
                 transforms=None,
                 image_size=256,
                 normalize=True):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.JPEG', recursive=True))

        if transforms is not None:
            self.transforms = transforms
        else:
            norm_mean = [0.5, 0.5, 0.5]
            norm_std = [0.5, 0.5, 0.5]
            if normalize:
                self.transforms = torchvision.transforms.Compose([
                    CenterCropLongEdge(),
                    torchvision.transforms.Resize(image_size,  T.InterpolationMode.BICUBIC),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(norm_mean, norm_std)
                ])
            else:
                self.transforms = torchvision.transforms.Compose([
                    CenterCropLongEdge(),
                    torchvision.transforms.Resize(image_size,T.InterpolationMode.BICUBIC),
                    torchvision.transforms.ToTensor()
                ])

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        #print ("before transform",img.shape)

        # transform
        if self.transforms is not None:
            img = self.transforms(img)
            #print ("after transform",img.shape)

        return img

@register_dataset(name='lensless_real_voronoi')
class LLDataset_ys(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.lqpath = sorted(glob(root + '/measure/0/*.png', recursive=True))
        self.gtpath = sorted(glob(root + '/gt/0/*.png', recursive=True))


    def __len__(self):
        return len(self.lqpath)

    def __getitem__(self, index: int):
        lqpath = self.lqpath[index]
        gtpath = self.gtpath[index]

        breakpoint()
        lq = cv2.imread(lqpath, -1).astype(np.float32)
        #lq = Image.open(lqpath).convert('RGB')
        lq = cv2.cvtColor(lq, cv2.COLOR_BGR2RGB)
        #lq = lq.astype(np.uint8)

        lq[lq<60.0]= 0
        lq = lq / 1023.0

        print ('1.before scale:',lq.max())
        # lq[lq<60.0]= 0
        

        print ('2.after scale:',lq.max())

        gt = Image.open(gtpath).convert('RGB')
   
        if self.transforms is not None:
            lq = self.transforms(lq)
            gt = self.transforms(gt)

        lq = lq / 1024.0
        
       

        return lq ,gt

@register_dataset(name='lensless_real_turing')
class LLDataset_ys(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.lqpath = sorted(glob(root + '/measure/*.tiff', recursive=True))
        self.gtpath = sorted(glob(root + '/gt/*.png', recursive=True))


    def __len__(self):
        return len(self.lqpath)

    def __getitem__(self, index: int):
        lqpath = self.lqpath[index]
        gtpath = self.gtpath[index]

        # lq = cv2.imread(lqpath, -1).astype(np.float32)
        lq = Image.open(lqpath).convert('RGB')
        # lq = cv2.cvtColor(lq, cv2.COLOR_BGR2RGB)
        #lq = lq.astype(np.uint8)


        print ('before scale:',lq.max())
        # lq[lq<60.0]= 0


        print ('after scale:',lq.max())

        gt = Image.open(gtpath).convert('RGB')
   
        if self.transforms is not None:
            lq = self.transforms(lq)
            gt = self.transforms(gt)

        lq = lq / 255.0

        return lq ,gt


@register_dataset(name='lensless_ys')
class LLDataset_ys(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)
        self.lqpath = sorted(glob(root + '/raw/*.tiff', recursive=True))
        self.gtpath = sorted(glob(root + '/label/*.jpg', recursive=True))
    def __len__(self):
        return len(self.lqpath)
    def __getitem__(self, index: int):
        lqpath = self.lqpath[index]
        gtpath = self.gtpath[index]
        lq = cv2.imread(lqpath, -1).astype(np.float32)
        lq = cv2.cvtColor(lq, cv2.COLOR_BGR2RGB)

        print ('before scale:',lq.max())
        #lq[lq<60.0]= 0
        #lq = lq / 1023.0
        print ('after scale:',lq.max())
        gt = Image.open(gtpath).convert('RGB')
        if self.transforms is not None:
            lq = self.transforms(lq)
            gt = self.transforms(gt)
        return lq ,gt
    
    
@register_dataset(name='lensless_ys_origin')
class LLDataset_ys(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)
        self.lqpath = sorted(glob(root + '/LQ/*.tiff', recursive=True))
        self.gtpath = sorted(glob(root + '/GT/*.png', recursive=True))
    def __len__(self):
        return len(self.lqpath)
    def __getitem__(self, index: int):
        lqpath = self.lqpath[index]
        gtpath = self.gtpath[index]
    
        lq = cv2.imread(lqpath, -1).astype(np.float32)
        lq = cv2.cvtColor(lq, cv2.COLOR_BGR2RGB)
        #lq = lq.astype(np.uint8)
        
        #lq = lq /4.0
        lq[lq<64.0]= 0
        lq = lq / 1023.0
        print ('lq scale:',lq.max())
        gt = Image.open(gtpath).convert('RGB')
        if self.transforms is not None:
            lq = self.transforms(lq)
            gt = self.transforms(gt)
        return lq ,gt


class MinMaxNormalize:
    def __call__(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor


@register_dataset(name='lensless_real_waller')
class LLDataset_ys(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.lqpath = sorted(glob(root + '/LQ/*.png', recursive=True))
        self.gtpath = sorted(glob(root + '/GT/*.png', recursive=True))

        image_size = 256

        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size,image_size),  T.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(norm_mean, norm_std)
            #MinMaxNormalize()
        ])

    def __len__(self):
        return len(self.lqpath)

    def __getitem__(self, index: int):
        lqpath = self.lqpath[index]
        gtpath = self.gtpath[index]

        lq = Image.open(lqpath).convert('RGB')
        gt = Image.open(gtpath).convert('RGB')

        print (lq.size,gt.size)
   
        if self.transforms is not None:
            lq = self.transforms(lq)
            gt = self.transforms(gt)

        #print (lq.shape, gt.shape)

        return lq ,gt

