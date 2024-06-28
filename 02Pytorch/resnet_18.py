import torch.utils
import torch.utils.data
from torchvision import transforms, datasets
import torchvision
import torch
import time
from matplotlib import pyplot
import numpy

#Esto es un comentario

if __name__ =='__main__':
    transforms_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456],[0.229, 0.224, 0.255])
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.255])
    ])


    train_dir = ""
    test_dir = ""
    train_dataset = datasets.ImageFolder(train_dir, transforms_train)
    test_dataset = datasets.ImageFolder(test_dir, transforms_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=True, num_workers=0)

    model = torchvision.models.resnet18(pretrainded = True)
    num_features = model.fc.in_features
    print(num_features)

    model.fc = torch.nn.Linear(512,2)
    model = model.to('gpu')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.0001, momentum=0.9)

    train_loss = []
    train_acuracy = []
    test_loss = []
    test_acuracy = []

    num_epochs = 10
    start_time = time.time()

    for epoch in range (num_epochs):
        print("Epoch {} running".format(epoch))
        """ Training Phase """
        model.train()
        running_loss = 0.
        running_corrects = 0

        for i , (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to('gpu')
            labels = labels.to('gpu')

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_corrects = torch.sum(preds == labels.data).item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects / len(train_dataset) * 100

        train_loss.append(epoch_loss)
        train_acuracy.append(epoch_acc)

        print('[Train # {}] Loss: {_0.4f}% Time: {:4f}s'.format(epoch))

        model.eval()
        with torchvision.no_grad():
            running_loss = 0.
            running_corrects =0
            for inputs, labels in test_dataloader:
                inputs = inputs()
                labels = labels()
                outputs = model(inputs)
                _, preds = torch.max(outputs,1)
                loss = criterion(outputs, labels)
                running_loss = loss.item()
                running_corrects += torch.sum(preds== labels.data).item()
            epoch_loss = running_loss / len(test_dataset)
            epoch_acc = running_corrects / len(test_dataset) * 100.

            test_loss.append(epoch_loss)
            test_acuracy.append(epoch_acc)
        
    