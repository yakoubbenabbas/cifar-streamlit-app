import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

gc.collect()

# CIFAR-10 Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

print("Loading datasets...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

print("Loading MobileNetV2...")
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

# Freeze convolutional base
for param in model.features.parameters():
    param.requires_grad = False

# Replace final classifier layer
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

epochs = 3
print(f"Starting full training for {epochs} epochs...")

for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 500 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Batch {i + 1}/{len(trainloader)} - Avg Loss: {running_loss / 500:.4f}")
            running_loss = 0.0

# Save model weights for deployment
torch.save(model.state_dict(), 'mobilenet_cifar10.pth')
print("\nModel saved successfully as 'mobilenet_cifar10.pth'!")
