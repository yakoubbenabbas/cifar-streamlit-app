import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights

# Clean memory before starting
gc.collect()

print("PyTorch loaded successfully!")
device = torch.device("cpu")

# Native 32x32 dimensions to minimize tensor allocation in RAM
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

print("Loading CIFAR-10 dataset...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)

print("Loading VGG16 backbone...")
model = vgg16(weights=VGG16_Weights.DEFAULT)

# Freeze convolutional feature extractor
for param in model.features.parameters():
    param.requires_grad = False

# Replace heavy 4096-node dense classifier with a lightweight head
model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(512 * 7 * 7, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)
)

model = model.to(device)
gc.collect()

print("VGG16 initialized cleanly!")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

print("Starting training pass...")
model.train()
for i, (inputs, labels) in enumerate(trainloader):
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if (i + 1) % 10 == 0:
        print(f"Batch {i + 1}/{len(trainloader)} - Loss: {loss.item():.4f}")
        break

print("Pipeline executed successfully without OOM!")
