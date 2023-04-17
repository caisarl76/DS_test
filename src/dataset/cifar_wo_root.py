from typing import Any, Callable, Optional, Tuple

from PIL import Image

from torch.utils.data import Dataset

class CIFAR10_wo_file(Dataset):
    def __init__(
        self,
        data_dict
    ) -> None:
        self.data: Any = []
        self.targets = []
        self.transform = None

        self.get_dataset(data_dict)
        
        
    def get_dataset(self, data_dict) -> None:
        self.data = data_dict['images']
        self.targets = data_dict['targets']
        self.transform = data_dict['transform']
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

