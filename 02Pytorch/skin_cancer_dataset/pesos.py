import torch
import torchvision.models as models
from torchvision.models import ResNet34_Weights

model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
torch.save(model.state_dict(), 'resnet34-weights.pth')