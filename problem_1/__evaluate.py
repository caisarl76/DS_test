import src.models.resnet_cifar as resnet

from torch import load as load_weight
from torch.cuda import is_available
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.utils.train_functions import test
from src.dataset.cifar_wo_root import CIFAR10_wo_file

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def evaluate(model_path, answer_dict):
    device = 'cuda' if is_available() else 'cpu'
    
    model = resnet.resnet20().to(device)
    
    checkpoint = load_weight(model_path)
    
    model.load_state_dict(checkpoint['state_dict'])
    train_epochs = checkpoint['epochs']

    dataset = CIFAR10_wo_file(answer_dict)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    acc, _ = test(data_loader, model, None, device)
    
    score=None
    if train_epochs > 40 or acc < answer_dict['acc']*0.95:
        passed = False
    else:
        passed = True
        score = acc
    
    return {'passed': passed, 'score':score}