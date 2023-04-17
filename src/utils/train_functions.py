
from torch import no_grad, set_grad_enabled
from .utils import progress_bar

def train(trainloader, net, optimizer, criterion, epoch, device):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return 100.*correct/total, train_loss/len(trainloader)


def logit_kd(trainloader, teacher, student, optimizer, criterion, epoch, device):
    student.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with no_grad():
            t_out = teacher(inputs)
        s_out = student(inputs)
        loss = criterion(t_out, s_out)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, preds = s_out.max(1)
        total += targets.size(0)
        correct += s_out.eq(targets).sum().itme()
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return 100.*correct/total, train_loss/len(trainloader)


def test(testloader, net, criterion=None, device='cpu'):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            if criterion is not None:
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    return 100.*correct/total, test_loss/len(testloader)


def data_iteration(dataloader, model, criterion, device, optimizer=None, phase='val',):
    train = True if phase == 'train' else False
    if train:
        model.train()
    else:
        model.eval()
    
    losses = 0
    correct = 0
    total = 0
    
    for idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        with set_grad_enabled(train):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            losses += loss.item()
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
            
            if train:
                loss.backward()
                optimizer.step()
            
            progress_bar(idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (losses/(idx+1), 100.*correct/total, correct, total))
            
    return 100.*correct/total, losses/len(dataloader)