import os
import argparse
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models


def age_to_group_index(age):
    if age <= 2: return 0
    if age <= 6: return 1
    if age <= 12: return 2
    if age <= 20: return 3
    if age <= 32: return 4
    if age <= 43: return 5
    if age <= 53: return 6
    return 7


AGE_GROUPS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60+"]


class UTKFaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        for filename in os.listdir(data_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            parts = filename.split("_")
            if len(parts) < 2:
                continue
            try:
                age = int(parts[0])
                gender = int(parts[1])  # UTKFace: 0 male, 1 female
                if gender not in [0, 1] or age < 0 or age > 120:
                    continue
                self.samples.append((filename, age, age_to_group_index(age), gender))
            except ValueError:
                continue

        if len(self.samples) == 0:
            raise RuntimeError("No valid UTKFace images found. Expected filenames like 25_0_1_xxx.jpg")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, age, age_group, gender = self.samples[idx]
        path = os.path.join(self.data_dir, filename)
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(age, dtype=torch.float32), torch.tensor(age_group), torch.tensor(gender)


class AgeGenderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.age_regressor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        self.age_group_classifier = nn.Linear(in_features, 8)
        self.gender_classifier = nn.Linear(in_features, 2)

    def forward(self, x):
        features = self.backbone(x)
        age = self.age_regressor(features).squeeze(1)
        age_group = self.age_group_classifier(features)
        gender = self.gender_classifier(features)
        return age, age_group, gender


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    full_dataset = UTKFaceDataset(args.data_dir, transform=train_transform)

    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # validation should not use augmentation
    val_set.dataset.transform = val_transform

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = AgeGenderNet().to(device)

    age_loss_fn = nn.L1Loss()
    group_loss_fn = nn.CrossEntropyLoss()
    gender_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_loss = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "best_utkface_model.pth")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        for images, ages, age_groups, genders in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images = images.to(device)
            ages = ages.to(device)
            age_groups = age_groups.to(device)
            genders = genders.to(device)

            pred_age, pred_group, pred_gender = model(images)

            loss_age = age_loss_fn(pred_age, ages)
            loss_group = group_loss_fn(pred_group, age_groups)
            loss_gender = gender_loss_fn(pred_gender, genders)

            loss = loss_age + loss_group + loss_gender

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss = 0
        gender_correct = 0
        group_correct = 0
        total = 0

        with torch.no_grad():
            for images, ages, age_groups, genders in val_loader:
                images = images.to(device)
                ages = ages.to(device)
                age_groups = age_groups.to(device)
                genders = genders.to(device)

                pred_age, pred_group, pred_gender = model(images)

                loss = (
                    age_loss_fn(pred_age, ages)
                    + group_loss_fn(pred_group, age_groups)
                    + gender_loss_fn(pred_gender, genders)
                )

                val_loss += loss.item()
                group_correct += (pred_group.argmax(1) == age_groups).sum().item()
                gender_correct += (pred_gender.argmax(1) == genders).sum().item()
                total += genders.size(0)

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        group_acc = group_correct / total * 100
        gender_acc = gender_correct / total * 100

        print(f"Epoch {epoch+1}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}, age_group_acc={group_acc:.2f}%, gender_acc={gender_acc:.2f}%")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "model_state": model.state_dict(),
                "age_groups": AGE_GROUPS
            }, best_path)
            print("Saved best model:", best_path)

    print("Training complete. Best model:", best_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset/UTKFace")
    parser.add_argument("--save_dir", default="models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
