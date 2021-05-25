##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os, sys, torch
import os.path as osp
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from copy import deepcopy
from PIL import Image

from .DownsampledImageNet import ImageNet16
from .SearchDatasetWrap import SearchDataset
from config_utils import load_config
from torchvision.transforms import transforms
from PIL import ImageFilter, ImageOps
import random
import torchvision.datasets as datasets

Dataset2Class = {'cifar10' : 10,
                 'cifar100': 100,
                 'imagenet-1k-s':1000,
                 'imagenet-1k' : 1000,
                 'ImageNet16'  : 1000,
                 'ImageNet16-150': 150,
                 'ImageNet16-120': 120,
                 'ImageNet16-200': 200}


class CUTOUT(object):

  def __init__(self, length):
    self.length = length

  def __repr__(self):
    return ('{name}(length={length})'.format(name=self.__class__.__name__, **self.__dict__))

  def __call__(self, img):
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - self.length // 2, 0, h)
    y2 = np.clip(y + self.length // 2, 0, h)
    x1 = np.clip(x - self.length // 2, 0, w)
    x2 = np.clip(x + self.length // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img *= mask
    return img


imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class Lighting(object):
  def __init__(self, alphastd,
         eigval=imagenet_pca['eigval'],
         eigvec=imagenet_pca['eigvec']):
    self.alphastd = alphastd
    assert eigval.shape == (3,)
    assert eigvec.shape == (3, 3)
    self.eigval = eigval
    self.eigvec = eigvec

  def __call__(self, img):
    if self.alphastd == 0.:
      return img
    rnd = np.random.randn(3) * self.alphastd
    rnd = rnd.astype('float32')
    v = rnd
    old_dtype = np.asarray(img).dtype
    v = v * self.eigval
    v = v.reshape((3, 1))
    inc = np.dot(self.eigvec, v).reshape((3,))
    img = np.add(img, inc)
    if old_dtype == np.uint8:
      img = np.clip(img, 0, 255)
    img = Image.fromarray(img.astype(old_dtype), 'RGB')
    return img

  def __repr__(self):
    return self.__class__.__name__ + '()'


class Cifar10RandomLabels(datasets.CIFAR10):
  """CIFAR10 dataset, with support for randomly corrupt labels.
  Params
  ------
  rand_seed: int
    Default 0. numpy random seed.
  num_classes: int
    Default 10. The number of classes in the dataset.
  """
  def __init__(self, rand_seed=0, num_classes=10, **kwargs):
    super(Cifar10RandomLabels, self).__init__(**kwargs)
    self.n_classes = num_classes
    self.rand_seed = rand_seed
    self.random_labels()

  def random_labels(self):
    labels = np.array(self.targets)
    print('num_classes:{}, random labels num:{}, random seed:{}'.format(self.n_classes, len(labels), self.rand_seed))
    np.random.seed(self.rand_seed)
    rnd_labels = np.random.randint(0, self.n_classes, len(labels))
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in rnd_labels]

    self.targets = labels

class Cifar100RandomLabels(datasets.CIFAR100):
  """CIFAR10 dataset, with support for randomly corrupt labels.
  Params
  ------
  rand_seed: int
    Default 0. numpy random seed.
  num_classes: int
    Default 100. The number of classes in the dataset.
  """
  def __init__(self, rand_seed=0, num_classes=100, **kwargs):
    super(Cifar100RandomLabels, self).__init__(**kwargs)
    self.n_classes = num_classes
    self.rand_seed = rand_seed
    self.random_labels()

  def random_labels(self):
    labels = np.array(self.targets)
    print('num_classes:{}, random labels num:{}, random seed:{}'.format(self.n_classes, len(labels), self.rand_seed))
    np.random.seed(self.rand_seed)
    rnd_labels = np.random.randint(0, self.n_classes, len(labels))
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in rnd_labels]

    self.targets = labels

class ImageNet16RandomLabels(ImageNet16):
  """CIFAR10 dataset, with support for randomly corrupt labels.
  Params
  ------
  rand_seed: int
    Default 0. numpy random seed.
  num_classes: int
    Default 120. The number of classes in the dataset.
  """
  def __init__(self, rand_seed=0, num_classes=120, **kwargs):
    super(ImageNet16RandomLabels, self).__init__(**kwargs)
    self.n_classes = num_classes
    self.rand_seed = rand_seed
    self.random_labels()
    # print('min_label:{}, max_label:{}'.format(min(self.targets), max(self.targets)))

  def random_labels(self):
    labels = np.array(self.targets)
    print('num_classes:{}, random labels num:{}, random seed:{}'.format(self.n_classes, len(labels), self.rand_seed))
    np.random.seed(self.rand_seed)
    rnd_labels = np.random.randint(1, self.n_classes+1, len(labels))
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in rnd_labels]

    self.targets = labels

def get_datasets(name, root, cutout, rand_seed, byol_aug_type=None):

  if name == 'cifar10':
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std  = [x / 255 for x in [63.0, 62.1, 66.7]]
  elif name == 'cifar100':
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std  = [x / 255 for x in [68.2, 65.4, 70.4]]
  elif name.startswith('imagenet-1k'):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
  elif name.startswith('ImageNet16'):
    mean = [x / 255 for x in [122.68, 116.66, 104.01]]
    std  = [x / 255 for x in [63.22,  61.26 , 65.09]]
  else:
    raise TypeError("Unknow dataset : {:}".format(name))

  # Data Argumentation
  if name == 'cifar10' or name == 'cifar100':
    lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)]
    if cutout > 0 : lists += [CUTOUT(cutout)]
    if byol_aug_type is None:
      train_transform = transforms.Compose(lists)
    elif byol_aug_type=='byol':
      online_aug = get_train_transform('BYOL_Tau', 32, mean, std)
      target_aug = get_train_transform('BYOL_Tau_Hat', 32, mean, std)
      train_transform = TwoImageAugmentations(online_aug, target_aug)
    else:
      raise NotImplementedError
    test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    xshape = (1, 3, 32, 32)
  elif name.startswith('ImageNet16'):
    lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2), transforms.ToTensor(), transforms.Normalize(mean, std)]
    if cutout > 0 : lists += [CUTOUT(cutout)]
    if byol_aug_type is None:
      train_transform = transforms.Compose(lists)
    elif byol_aug_type=='byol':
      online_aug = get_train_transform('BYOL_Tau', 16, mean, std)
      target_aug = get_train_transform('BYOL_Tau_Hat', 16, mean, std)
      train_transform = TwoImageAugmentations(online_aug, target_aug)
    else:
      raise NotImplementedError
    test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    xshape = (1, 3, 16, 16)
  elif name == 'tiered':
    lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(80, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)]
    if cutout > 0 : lists += [CUTOUT(cutout)]
    train_transform = transforms.Compose(lists)
    test_transform  = transforms.Compose([transforms.CenterCrop(80), transforms.ToTensor(), transforms.Normalize(mean, std)])
    xshape = (1, 3, 32, 32)
  elif name.startswith('imagenet-1k'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if name == 'imagenet-1k':
      xlists    = [transforms.RandomResizedCrop(224)]
      xlists.append(
        transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2))
      xlists.append( Lighting(0.1))
    elif name == 'imagenet-1k-s':
      xlists    = [transforms.RandomResizedCrop(224, scale=(0.2, 1.0))]
    else: raise ValueError('invalid name : {:}'.format(name))
    xlists.append( transforms.RandomHorizontalFlip(p=0.5) )
    xlists.append( transforms.ToTensor() )
    xlists.append( normalize )
    train_transform = transforms.Compose(xlists)
    test_transform  = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    xshape = (1, 3, 224, 224)
  else:
    raise TypeError("Unknow dataset : {:}".format(name))

  if name == 'cifar10':
    train_data = Cifar10RandomLabels(root=root, train=True , transform=train_transform, download=True, rand_seed=rand_seed)
    test_data = datasets.CIFAR10(root=root, train=True , transform=test_transform, download=True)
    # test_data  = datasets.CIFAR10(root=root, train=False, transform=test_transform , download=True)
    assert len(train_data) == 50000 and len(test_data) == 50000
  elif name == 'cifar100':
    train_data = Cifar100RandomLabels(root=root, train=True , transform=train_transform, download=True, rand_seed=rand_seed)
    test_data  = datasets.CIFAR100(root=root, train=False, transform=test_transform , download=True)
    assert len(train_data) == 50000 and len(test_data) == 10000
  elif name.startswith('imagenet-1k'):
    train_data = dset.ImageFolder(osp.join(root, 'train'), train_transform)
    test_data  = dset.ImageFolder(osp.join(root, 'val'),   test_transform)
    assert len(train_data) == 1281167 and len(test_data) == 50000, 'invalid number of images : {:} & {:} vs {:} & {:}'.format(len(train_data), len(test_data), 1281167, 50000)
  elif name == 'ImageNet16':
    train_data = ImageNet16RandomLabels(root=root, train=True ,transform=train_transform, rand_seed=rand_seed)
    test_data  = ImageNet16(root=root, train=False, transform=test_transform)
    assert len(train_data) == 1281167 and len(test_data) == 50000
  elif name == 'ImageNet16-120':
    train_data = ImageNet16RandomLabels(root=root, train=True , transform=train_transform, num_classes=120, use_num_of_class_only=120, rand_seed=rand_seed)
    test_data  = ImageNet16(root=root, train=False, transform=test_transform , use_num_of_class_only=120)
    assert len(train_data) == 151700 and len(test_data) == 6000
  elif name == 'ImageNet16-150':
    train_data = ImageNet16RandomLabels(root=root, train=True , transform=train_transform, num_classes=150, use_num_of_class_only=150, rand_seed=rand_seed)
    test_data  = ImageNet16(root=root, train=False, transform=test_transform , use_num_of_class_only=150)
    assert len(train_data) == 190272 and len(test_data) == 7500
  elif name == 'ImageNet16-200':
    train_data = ImageNet16RandomLabels(root=root, train=True ,transform=train_transform, num_classes=200, use_num_of_class_only=200, rand_seed=rand_seed)
    test_data  = ImageNet16(root=root, train=False, transform=test_transform , use_num_of_class_only=200)
    assert len(train_data) == 254775 and len(test_data) == 10000
  else: raise TypeError("Unknow dataset : {:}".format(name))
  
  class_num = Dataset2Class[name]
  return train_data, test_data, xshape, class_num


def get_nas_search_loaders(train_data, valid_data, dataset, config_root, batch_size, workers):
  if isinstance(batch_size, (list,tuple)):
    batch, test_batch = batch_size
  else:
    batch, test_batch = batch_size, batch_size
  if dataset == 'cifar10':
    #split_Fpath = 'configs/nas-benchmark/cifar-split.txt'
    cifar_split = load_config('{:}/cifar-split.txt'.format(config_root), None, None)
    train_split, valid_split = cifar_split.train, cifar_split.valid # search over the proposed training and validation set
    #logger.log('Load split file from {:}'.format(split_Fpath))      # they are two disjoint groups in the original CIFAR-10 training set
    # To split data
    xvalid_data  = valid_data
    search_data   = SearchDataset(dataset, train_data, train_split, valid_split)
    # data loader
    search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch, shuffle=True , num_workers=workers, pin_memory=True)
    train_loader  = torch.utils.data.DataLoader(train_data , batch_size=batch, sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split), num_workers=workers, pin_memory=True)
    # valid_loader  = torch.utils.data.DataLoader(xvalid_data, batch_size=test_batch, sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split), num_workers=workers, pin_memory=True)
    valid_loader  = torch.utils.data.DataLoader(xvalid_data, batch_size=test_batch, sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split), num_workers=workers, pin_memory=True)

  elif dataset == 'cifar100':
    cifar100_test_split = load_config('{:}/cifar100-test-split.txt'.format(config_root), None, None)
    search_train_data = train_data
    search_valid_data = deepcopy(valid_data) ; search_valid_data.transform = train_data.transform
    search_data   = SearchDataset(dataset, [search_train_data,search_valid_data], list(range(len(search_train_data))), cifar100_test_split.xvalid)
    search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch, shuffle=True , num_workers=workers, pin_memory=True)
    train_loader  = torch.utils.data.DataLoader(train_data , batch_size=batch, shuffle=True , num_workers=workers, pin_memory=True)
    valid_loader  = torch.utils.data.DataLoader(valid_data , batch_size=test_batch, sampler=torch.utils.data.sampler.SubsetRandomSampler(cifar100_test_split.xvalid), num_workers=workers, pin_memory=True)
  elif dataset == 'ImageNet16-120':
    imagenet_test_split = load_config('{:}/imagenet-16-120-test-split.txt'.format(config_root), None, None)
    search_train_data = train_data
    search_valid_data = deepcopy(valid_data) ; search_valid_data.transform = train_data.transform
    search_data   = SearchDataset(dataset, [search_train_data,search_valid_data], list(range(len(search_train_data))), imagenet_test_split.xvalid)
    search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch, shuffle=True , num_workers=workers, pin_memory=True)
    train_loader  = torch.utils.data.DataLoader(train_data , batch_size=batch, shuffle=True , num_workers=workers, pin_memory=True)
    valid_loader  = torch.utils.data.DataLoader(valid_data , batch_size=test_batch, sampler=torch.utils.data.sampler.SubsetRandomSampler(imagenet_test_split.xvalid), num_workers=workers, pin_memory=True)
  else:
    raise ValueError('invalid dataset : {:}'.format(dataset))
  return search_loader, train_loader, valid_loader

def get_train_transform(aug, image_size, mean, std):

  if aug == 'BYOL_Tau':
    transform = transforms.Compose([
      transforms.RandomResizedCrop(image_size),
      transforms.RandomHorizontalFlip(),
      transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
      transforms.RandomGrayscale(p=0.2),
      transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=1.0),
      transforms.RandomApply([Solarization(128)], p=0.0),
      transforms.ToTensor(),
      transforms.Normalize(mean, std),

    ])
  elif aug == 'BYOL_Tau_Hat':
    transform = transforms.Compose([
      transforms.RandomResizedCrop(image_size),
      transforms.RandomHorizontalFlip(),
      transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
      transforms.RandomGrayscale(p=0.2),
      transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1),
      transforms.RandomApply([Solarization(128)], p=0.2),
      transforms.ToTensor(),
      transforms.Normalize(mean, std),
    ])
  else:
    raise NotImplementedError

  return transform


class TwoImageAugmentations:
  def __init__(self, online_aug, target_aug):
    self.online_aug = online_aug
    self.target_aug = target_aug

  def __call__(self, x):
    online_image = self.online_aug(x)
    target_image = self.target_aug(x)
    return [online_image, target_image]

class GaussianBlur(object):
  """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

  def __init__(self, sigma=[.1, 2.]):
    self.sigma = sigma

  def __call__(self, x):
    sigma = random.uniform(self.sigma[0], self.sigma[1])
    x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
    return x


class Solarization(object):
  def __init__(self, magnitude=128):
    self.magnitude = magnitude

  def __call__(self, x):
    x = ImageOps.solarize(x, self.magnitude)
    return x

if __name__ == '__main__':
  byol = True
  train_data, test_data, xshape, class_num = get_datasets('cifar10', '/home/zhangxuanyang/dataset/cifar.python/', -1, byol)
  search_loader, _, valid_loader = get_nas_search_loaders(train_data, test_data, 'cifar10', 'configs/nas-benchmark/', \
                                        (3, 3), 4)
  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(search_loader):
    print(base_inputs)
    break

#  import pdb; pdb.set_trace()
