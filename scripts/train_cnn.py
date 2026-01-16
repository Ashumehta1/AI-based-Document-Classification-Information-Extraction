# import os
# import torch
# import torch.nn as nn
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# # ----------------------------
# # Config
# # ----------------------------
# DATASET_DIR = "dataset"
# MODEL_PATH = "models/vision/doc_classifier.pth"
# IMG_SIZE = 224
# BATCH_SIZE = 8
# EPOCHS = 15
# NUM_CLASSES = 4

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ----------------------------
# # Dataset
# # ----------------------------
# transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor()
# ])

# dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
# loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# # ----------------------------
# # CNN Model
# # ----------------------------
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )

#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 28 * 28, 256),
#             nn.ReLU(),
#             nn.Linear(256, NUM_CLASSES)
#         )

#     def forward(self, x):
#         return self.classifier(self.features(x))

# model = SimpleCNN().to(device)

# # ----------------------------
# # Training
# # ----------------------------
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# os.makedirs("models/vision", exist_ok=True)

# for epoch in range(EPOCHS):
#     total_loss = 0
#     for images, labels in loader:
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}")

# # ----------------------------
# # Save Model
# # ----------------------------
# torch.save(model.state_dict(), MODEL_PATH)
# print("Model saved to", MODEL_PATH)

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.vision.simple_cnn import SimpleCNN

# ----------------------------
# Config
# ----------------------------
DATASET_DIR = "dataset"
MODEL_PATH = "models/vision/doc_classifier.pth"
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 15
NUM_CLASSES = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    os.makedirs("models/vision", exist_ok=True)

    for epoch in range(EPOCHS):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved to", MODEL_PATH)


# THIS LINE IS THE KEY 
if __name__ == "__main__":
    train()
