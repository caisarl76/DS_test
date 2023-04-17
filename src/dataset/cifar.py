import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

train_transform_224 = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

        
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

val_transform_224 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class CIFAR10_Subset(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, subset_ratio=1.0, rand_number=0, train=True, use_224=False,
                 transform=None, target_transform=None, download=True):
        super(CIFAR10_Subset, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num(self.cls_num, subset_ratio)
        self.gen_subset_data(img_num_list)
        
        self.transform = train_transform_224 if use_224 else train_transform
        if transform:
            self.transform=transform

    def get_img_num(self, cls_num, subset_ratio):
        img_num = len(self.data) / cls_num
        img_num = int(img_num*subset_ratio)
        img_num_per_cls = [img_num]*cls_num
        return img_num_per_cls
        

    def gen_subset_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            sample = self.transform(img)
            
        return sample, target

class CIFAR100_Subset(CIFAR10_Subset):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100

def get_dataset(args, transform=None, use224=False):
    is_100 = True if '100' in args.dataset else False
    if is_100:
        train_class = CIFAR100_Subset
        val_class = torchvision.datasets.CIFAR100
        args.num_classes = 100
    else:
        train_class = CIFAR10_Subset
        val_class = torchvision.datasets.CIFAR10
        args.num_classes = 10
    
    image_datasets = {
        'train': train_class(root=args.data_path, train=True, transform=transform, use_224=use224),
        'val':val_class(root=args.data_path, train=False, transform=val_transform_224 if use224 else val_transform)
    }
    dataloaders = {
        x: data.DataLoader(
        image_datasets[x], 
        batch_size=args.batch_size,
        shuffle=True if x =='train' else False,
        num_workers=2
        ) for x in ['train', 'val']
        }
    return dataloaders


if __name__ == '__main__':
    pass