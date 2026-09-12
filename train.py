import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V2_Weights

# =========================
# 1. SETTINGS
# =========================

DATASET_PATH = "dataset"
MODEL_PATH = "foodrescue_model.pth"

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# =========================
# 2. IMAGE TRANSFORMS
# =========================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# 3. LOAD DATASET
# =========================

dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=train_transform
)

print("Classes:")
print(dataset.classes)

print("Total images:", len(dataset))


# =========================
# 4. TRAIN / VALIDATION SPLIT
# =========================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# =========================
# 5. LOAD MOBILENETV2
# =========================

weights = MobileNet_V2_Weights.DEFAULT

model = models.mobilenet_v2(
    weights=weights
)

# Freeze feature extractor
for parameter in model.features.parameters():
    parameter.requires_grad = False


# Replace final classifier
number_of_classes = len(dataset.classes)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    number_of_classes
)

model = model.to(DEVICE)


# =========================
# 6. LOSS + OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier[1].parameters(),
    lr=LEARNING_RATE
)


# =========================
# 7. TRAINING
# =========================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_accuracy = 100 * correct / total


    # =========================
    # VALIDATION
    # =========================

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_accuracy = 100 * val_correct / val_total


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )


# =========================
# 8. SAVE MODEL
# =========================

torch.save(
    {
        "model_state": model.state_dict(),
        "classes": dataset.classes
    },
    MODEL_PATH
)

print()
print("===================================")
print("Training completed!")
print("Model saved as:", MODEL_PATH)
print("Classes:", dataset.classes)
print("===================================")
