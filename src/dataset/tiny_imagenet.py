import os
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ])
}


def get_dataset(args):
    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data, 'tiny-imagenet-200', x), data_transforms[x]) 
                  for x in ['train', 'val','test']}
    dataloaders = {
        x: data.DataLoader(
        image_datasets[x], 
        batch_size=args.batch_size,
        shuffle=True if x =='train' else False,
        num_workers=args.num_workers
        ) for x in ['train', 'val']
        }
    
    return dataloaders