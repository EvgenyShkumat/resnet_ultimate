import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==========================================
# КОНФИГУРАЦИЯ
# ==========================================
PARTICIPANT_SURNAME = "Ivanov"
TRAIN_DIR = 'train'  
TEST_DIR = 'test'

BATCH_SIZE = 64
IMG_SIZE = 224  # Обязательно 224 для ResNet
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

# ==========================================
# 1. Загрузка данных
# ==========================================
def get_dataloaders(train_dir, test_dir, augment=False):
    # Базовые (для валидации)
    base_transforms = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if augment:
        # Усиленная аугментация
        train_transforms = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Добавили работу с яркостью
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        train_transforms = base_transforms

    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} not found.")
        exit()

    full_dataset = datasets.ImageFolder(train_dir, transform=base_transforms)
    targets = full_dataset.targets 
    
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=0.2,
        shuffle=True,
        stratify=targets, 
        random_state=42
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(train_dir, transform=base_transforms)

    train_subset = Subset(train_dataset, train_idx)
    val_subset   = Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    if os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(test_dir, transform=base_transforms)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader, full_dataset.classes

# ==========================================
# 2. Модель: Pre-trained ResNet34 (Поглубже)
# ==========================================
class ResNetEmotion(nn.Module):
    def __init__(self, num_classes):
        super(ResNetEmotion, self).__init__()
        
        # Берем ResNet34 вместо 18 (он мощнее)
        self.net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # 1 канал
        original_weights = self.net.conv1.weight.data.mean(dim=1, keepdim=True)
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net.conv1.weight.data = original_weights 
        
        # Выход
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. Обучение
# ==========================================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / len(loader), correct / total

# ==========================================
# 4. Метрики + TTA (Test Time Augmentation)
# ==========================================
def evaluate_full_metrics(model, loader, classes, title_suffix="", use_tta=True):
    if loader is None: return 0.0
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    print(f"Evaluating {title_suffix} (TTA={'On' if use_tta else 'Off'})...")

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # 1. Прямой проход
            outputs = model(images)
            
            # 2. TTA: Зеркальный проход (если включено)
            if use_tta:
                # Переворачиваем картинки по горизонтали
                images_flipped = torch.flip(images, dims=[3]) 
                outputs_flipped = model(images_flipped)
                # Усредняем логиты (это повышает надежность)
                outputs = (outputs + outputs_flipped) / 2.0
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\n--- ОТЧЕТ: {title_suffix} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # ROC AUC
    try:
        n_classes = len(classes)
        y_test_bin = label_binarize(all_labels, classes=range(n_classes))
        y_score = np.array(all_probs)
        roc_auc_val = roc_auc_score(y_test_bin, y_score, multi_class='ovr', average='weighted')
        print(f"ROC AUC Score: {roc_auc_val:.4f}")
        
        plt.figure(figsize=(8, 6))
        fpr, tpr = dict(), dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            plt.plot(fpr[i], tpr[i], label=f'{classes[i]}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Curves - {title_suffix}')
        plt.legend()
        plt.savefig(f'roc_{title_suffix}.png')
        plt.close()
    except Exception as e:
        print(f"ROC Error: {e}")

    # Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title(f'Confusion Matrix - {title_suffix}')
    plt.savefig(f'cm_{title_suffix}.png')
    plt.close()

    return acc

# ==========================================
# 5. Запуск
# ==========================================
def run_experiment(exp_name, augment, epochs, load_weights=None):
    print(f"\n[{exp_name.upper()}] Start ResNet34 (224x224) + AdamW + Cosine...")
    
    train_loader, val_loader, test_loader, classes = get_dataloaders(TRAIN_DIR, TEST_DIR, augment=augment)

    # Меняем на ResNet34
    model = ResNetEmotion(len(classes)).to(DEVICE)
    
    if load_weights:
        print(f"Loading weights from {load_weights}...")
        model.load_state_dict(torch.load(load_weights))
        lr = 0.0001
    else:
        lr = 0.001
    
    # Label Smoothing помогает от переобучения
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # AdamW лучше работает с Transfer Learning
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    # CosineAnnealingLR - более современный шедулер, плавно снижает LR до 0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    temp_save_path = f"temp_{exp_name}.pth"

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        
        # Валидация (без TTA пока, для скорости)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                out = model(imgs)
                _, pred = torch.max(out, 1)
                val_correct += (pred == lbls).sum().item()
                val_total += lbls.size(0)
        val_acc = val_correct / val_total
        
        scheduler.step() # Для CosineAnnealing шаг делается каждую эпоху
        print(f"Ep {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val_Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), temp_save_path)

    print(f"\n>>> Results for {exp_name}...")
    model.load_state_dict(torch.load(temp_save_path))
    
    # Включаем TTA (Test Time Augmentation) для финальной оценки
    evaluate_full_metrics(model, val_loader, classes, title_suffix=f"{exp_name}_internal", use_tta=True)
    
    final_test_acc = 0.0
    if test_loader:
        print(">>> BLIND TEST (with TTA):")
        final_test_acc = evaluate_full_metrics(model, test_loader, classes, title_suffix=f"{exp_name}_BLIND", use_tta=True)
    
    final_name = f"{PARTICIPANT_SURNAME}_{exp_name}_resnet34.pth"
    if os.path.exists(final_name): os.remove(final_name)
    os.rename(temp_save_path, final_name)
    
    return final_name, final_test_acc

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # 1. Baseline ResNet34
    base_name, base_acc = run_experiment("resnet34_base", augment=False, epochs=12)

    # 2. Tuned ResNet34 (с TTA)
    # Чуть больше эпох для CosineAnnealing
    best_name, best_acc = run_experiment("resnet34_tuned", augment=True, epochs=18, load_weights=base_name)

    print("\n" + "="*30)
    print(f"Participant: {PARTICIPANT_SURNAME}")
    print(f"Baseline ResNet34 Acc: {base_acc:.4f}")
    print(f"Tuned ResNet34 Acc:    {best_acc:.4f}")
    print(f"Model File: {best_name}")