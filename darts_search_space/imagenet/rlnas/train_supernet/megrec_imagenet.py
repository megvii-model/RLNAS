from megrec.torch_data import build_imagenet
from torchvision.transforms import transforms
import numpy as np
from megrec.torch_data.preprocess.loader import (  # pylint: disable=not-callable,import-error,no-name-in-module
    NoriLoader,
    pil_decoder,
)

class Preprocess(object):
    def __init__(self, transform):
        self.transform = transform
        self.loader = NoriLoader(unpickle=False)

    def __call__(self, sample):
        data = self.loader(sample['path'])
        img = self.transform(pil_decoder(data))
        # target = sample['label']
        target = np.random.randint(0,1000)
        return img, target

def get_train_transform(aug='NULL'):
    if aug == 'NULL':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    elif aug == 'Color':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transform

def get_eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_imagenet_dataset(data_dir, train, transform):
    preprocess = Preprocess(transform)
    dataset = build_imagenet(data_dir, train=train, preprocess=preprocess)
    return dataset