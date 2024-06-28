import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from matplotlib import pyplot

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=28, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x,1)
        output = F.log_softmax(x,dim=1)
        return output

def train (model, device, train_loader, optmizer, epoch):
    model.train()
    running_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optmizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optmizer.step()
        if batch_idx % 1000 ==0:
            print('Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx *  len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        running_loss += loss.item()
    return running_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct =0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim = 1, keepdim =True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\n Test set: Average loss:{:.0f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)

if __name__ == '__main__':
    device = torch.device('cpu')

    transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    datasets1 = datasets.MNIST('./MNIST', train= True, download=True,transform=transforms)
    datasets2 = datasets.MNIST('./MNIST', train=False, transform=transforms)

    train_loader = torch.utils.data.DataLoader(datasets1, batch_size=64)
    test_loader = torch.utils.data.DataLoader(datasets2, batch_size=64)

    model = CNN().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    loss_list = list()
    accuracy_list = list()

    scheduler = StepLR(optimizer,step_size=1, gamma=0.7)
    for epoch in range (1, 15 + 1):
        loss = train(model, device, train_loader, optimizer, epoch)
        loss_list.append(loss)
        accuracy = test(model, device, test_loader)
        accuracy_list.append(accuracy)
        scheduler.step()
    state_dict = model.state_dict()


    pyplot.clf()
    pyplot.subplots(1,10)
    for i in range(10):
        pyplot.subplot(1, 10, i+1)
        pyplot.imshow(state_dict['conv1.weight'][i, 0].cpu().detach().numpy())
    pyplot.show()

    pyplot.clf()
    pyplot.plot(loss_list)
    pyplot.show()

    pyplot.clf()
    pyplot.plot(accuracy_list, label ='TEST')
    pyplot.legend()
    pyplot.show()