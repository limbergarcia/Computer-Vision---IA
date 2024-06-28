import torch.utils
import torch.utils.data
from torchvision import transforms, datasets
import torchvision
import torch
import time
from matplotlib import pyplot
import numpy

import os

# Función para entrenar y evaluar un modelo
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    model = model.to('cuda')
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Fase de entrenamiento
        model.train()
        running_train_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects / len(train_loader.dataset) * 100.0
        train_loss.append(epoch_train_loss)
        train_accuracy.append(epoch_train_acc)

        print(f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_acc:.2f}%")

        # Fase de evaluación
        model.eval()
        running_test_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_test_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        epoch_test_acc = running_corrects / len(test_loader.dataset) * 100.0
        test_loss.append(epoch_test_loss)
        test_accuracy.append(epoch_test_acc)

        print(f"Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {epoch_test_acc:.2f}%")
        print()

    return train_loss, train_accuracy, test_loss, test_accuracy

if __name__ == '__main__':
    # Define transformaciones y carga de datos
    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])

    train_dataset = datasets.ImageFolder("skin_cancer_dataset/train", transforms_train)
    test_dataset = datasets.ImageFolder("skin_cancer_dataset/test", transforms_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=0)

    # Modelos a comparar
    models = {
        "ResNet-18": torchvision.models.resnet18(pretrained=True),
        "ResNet-34": torchvision.models.resnet34(pretrained=True)
    }

    # Parámetros comunes para el entrenamiento
    criterion = torch.nn.CrossEntropyLoss()

    # Entrenamiento y evaluación de cada modelo
    results = {}
    for model_name, model in models.items():
        # Mueve el modelo a GPU si está disponible
        model = model.to('cuda')

        # Optimizador específico para cada modelo
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

        print(f"Training {model_name}")
        results[model_name] = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=10)

    # Graficar resultados
    pyplot.figure(figsize=(10, 5))

    # Gráfico de pérdida
    pyplot.subplot(1, 2, 1)
    for model_name, (train_loss, _, test_loss, _) in results.items():
        pyplot.plot(range(10), train_loss, label=f"{model_name} Train")
        pyplot.plot(range(10), test_loss, label=f"{model_name} Test")
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Loss")
    pyplot.legend()

    # Gráfico de precisión
    pyplot.subplot(1, 2, 2)
    for model_name, (_, train_accuracy, _, test_accuracy) in results.items():
        pyplot.plot(range(10), train_accuracy, label=f"{model_name} Train")
        pyplot.plot(range(10), test_accuracy, label=f"{model_name} Test")
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Accuracy (%)")
    pyplot.legend()

    pyplot.tight_layout()
    pyplot.show()