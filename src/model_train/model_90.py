import os
import numpy as np
import cv2
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms as T
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc, classification_report)
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ============================================================================
# 1. CONFIGURATION (90/5/5)
# ============================================================================
class Config:
    # --- PATH SETTINGS ---
    DATASET_PATH = 'E:/Dataset_2/' 
    
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(PROJECT_ROOT))
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'model')
    RESULTS_PATH = os.path.join(BASE_DIR, 'results')
    LOG_DIR = os.path.join(BASE_DIR, 'runs') 
    
    # --- TRAINING HYPERPARAMETERS ---
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    MAX_EPOCHS = 30          
    LEARNING_RATE = 0.0001
    
    # --- MODEL ARCHITECTURE ---
    FEATURE_DIMENSION = 1280  
    APPLY_HISTOGRAM_EQ = True
    
    # --- SVM SETTINGS ---
    SVM_KERNEL = 'rbf'
    SVM_C = 1.0
    
    # --- DATA SPLIT (90/5/5) ---
    TRAIN_RATIO = 0.90       # <--- 90% Training
    VAL_RATIO = 0.05         # <--- 5% Validation (Cukup untuk Early Stopping)
    TEST_RATIO = 0.05        # <--- 5% Testing (Untuk Final Evaluation)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

config = Config()

# --- DISPLAY INFO ---
print("="*50)
print(f" SYSTEM CONFIGURATION (LAZY LOADING)")
print("="*50)
print(f"Device:       {config.DEVICE}")
if torch.cuda.is_available():
    print(f"GPU Name:     {torch.cuda.get_device_name(0)}")
print(f"Split Ratio:  {int(config.TRAIN_RATIO*100)}/{int(config.VAL_RATIO*100)}/{int(config.TEST_RATIO*100)}")
print(f"Features Dim: {config.FEATURE_DIMENSION}")
print(f"Max Epochs:   {config.MAX_EPOCHS}")
print("="*50)

# ============================================================================
# 2. DATA LOADING (LAZY STRATEGY)
# ============================================================================
def load_and_split_data(dataset_path):
    
    filepaths = []
    labels = []
    class_names = sorted(os.listdir(dataset_path))
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    print("\nScanning file paths...")
    for cls_name, idx in class_to_idx.items():
        cls_path = os.path.join(dataset_path, cls_name)
        if os.path.isdir(cls_path):
            files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"  Found {len(files)} images in {cls_name}")
            for img_name in files:
                full_path = os.path.join(cls_path, img_name)
                filepaths.append(full_path)
                labels.append(idx)

    X = np.array(filepaths) 
    y = np.array(labels)
    
    # --- SPLIT LOGIC 90/5/5) ---
    # 1. Split Train (90%) vs Temp 10%)
    temp_size = config.VAL_RATIO + config.TEST_RATIO # 0.1 + 0.1 = 0.2
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=temp_size, 
        random_state=42, stratify=y
    )
    
    # 2. Split Temp (10%) -> Val (5%) & Test (5%)
    relative_test_size = config.TEST_RATIO / temp_size
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=relative_test_size, 
        random_state=42, stratify=y_temp
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# ============================================================================
# 3. PYTORCH DATASET (WITH ON-THE-FLY LOADING)
# ============================================================================
class MalariaDataset(Dataset):
    def __init__(self, filepaths, labels, img_size=(224, 224), apply_hist_eq=True, augment=False):
        self.filepaths = filepaths
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        self.img_size = img_size
        self.apply_hist_eq = apply_hist_eq
        self.augment = augment
        
        self.aug_transform = T.Compose([
            T.RandomAffine(degrees=20, translate=(0.2, 0.2), shear=0.2, scale=(0.8, 1.2)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # 1. Load Image Here (Lazy Loading)
        img_path = self.filepaths[idx]
        img = cv2.imread(img_path)
        
        if img is None:
            # Fallback for corrupt image (return black image)
            img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.float32)
        else:
            # 2. Preprocessing
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)

            if self.apply_hist_eq:
                img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
            
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1)) # HWC -> CHW

        img_tensor = torch.tensor(img, dtype=torch.float32)
        
        if self.augment:
            img_tensor = self.aug_transform(img_tensor)
            
        return img_tensor, self.labels[idx]

# ============================================================================
# 4. MODEL ARCHITECTURE
# ============================================================================
class HybridEfficientNet(nn.Module):
    def __init__(self, feature_dim=128): 
        super(HybridEfficientNet, self).__init__()
        weights = models.EfficientNet_B3_Weights.DEFAULT
        self.backbone = models.efficientnet_b3(weights=weights)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.backbone.classifier = nn.Identity()
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224) 
            out = self.backbone(dummy)
            in_features = out.shape[1]

        self.custom_head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.45),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, feature_dim), 
            nn.ReLU()
        )
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, x):
        features = self.custom_head(self.backbone(x))
        logits = self.classifier(features)
        return logits, features

# ============================================================================
# 5. SETUP
# ============================================================================

# Load Paths ONLY
X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data(config.DATASET_PATH)

total_samples = len(X_train) + len(X_val) + len(X_test)
print("\n" + "="*50)
print(f" DATASET SUMMARY (90/5/5)")
print("="*50)
print(f"Total Images:   {total_samples}")
print("-" * 30)
print(f"Training Set:   {len(X_train)} samples ({len(X_train)/total_samples*100:.1f}%)")
print(f"Validation Set: {len(X_val)} samples ({len(X_val)/total_samples*100:.1f}%)")
print(f"Test Set:       {len(X_test)} samples ({len(X_test)/total_samples*100:.1f}%)")
print("="*50 + "\n")

# Initialize DataLoaders
train_dataset = MalariaDataset(X_train, y_train, img_size=config.IMG_SIZE, apply_hist_eq=config.APPLY_HISTOGRAM_EQ, augment=True)
val_dataset = MalariaDataset(X_val, y_val, img_size=config.IMG_SIZE, apply_hist_eq=config.APPLY_HISTOGRAM_EQ, augment=False)
test_dataset = MalariaDataset(X_test, y_test, img_size=config.IMG_SIZE, apply_hist_eq=config.APPLY_HISTOGRAM_EQ, augment=False)
# For SVM feature extraction (Train data without augmentation)
train_feat_dataset = MalariaDataset(X_train, y_train, img_size=config.IMG_SIZE, apply_hist_eq=config.APPLY_HISTOGRAM_EQ, augment=False)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
train_feat_loader = DataLoader(train_feat_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

# Initialize Model with Updated Feature Dimension
model = HybridEfficientNet(feature_dim=config.FEATURE_DIMENSION).to(config.DEVICE)
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

class EarlyStoppingHandler:
    def __init__(self, patience=15):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopper = EarlyStoppingHandler(patience=15)

# ============================================================================
# 6. TRAINING LOOP
# ============================================================================
print("="*50)
print(" PHASE 1: TRAINING CNN")
print("="*50)

best_epoch = 0
history_log = [] 

for epoch in range(config.MAX_EPOCHS):
    model.train()
    train_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.MAX_EPOCHS}", leave=False)
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
        
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    all_preds_val, all_targets_val = [], []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            logits, _ = model(inputs)
            val_loss += criterion(logits, labels).item()
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_preds_val.extend(preds.cpu().numpy())
            all_targets_val.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    
    val_acc = accuracy_score(all_targets_val, all_preds_val)
    val_prec = precision_score(all_targets_val, all_preds_val, zero_division=0)
    val_rec = recall_score(all_targets_val, all_preds_val, zero_division=0)
    
    print(f"Epoch {epoch+1:02d}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
          f"Acc: {val_acc*100:.2f}% | Prec: {val_prec*100:.2f}% | Recall: {val_rec*100:.2f}%")
    
    history_log.append({
        'Epoch': epoch + 1,
        'Train_Loss': avg_train_loss,
        'Val_Loss': avg_val_loss,
        'Val_Accuracy': val_acc,
        'Val_Precision': val_prec,
        'Val_Recall': val_rec
    })

    early_stopper(avg_val_loss, model)
    if early_stopper.early_stop:
        print(f"Early stopping triggered at Epoch {epoch+1}")
        best_epoch = epoch + 1 - early_stopper.patience
        break
    
    if avg_val_loss == early_stopper.best_loss:
        best_epoch = epoch + 1

if early_stopper.best_model_state:
    model.load_state_dict(early_stopper.best_model_state)
    print(f"Restored best weights from epoch {best_epoch}")

# ============================================================================
# 7. SVM TRAINING & EVALUATION
# ============================================================================
print("\n" + "="*50)
print(f" PHASE 2: SVM TRAINING & EVALUATION (Features: {config.FEATURE_DIMENSION}D)")
print("="*50)

def extract_features(loader, model):
    model.eval()
    feats, labels = [], []
    print("Extracting features...") 
    with torch.no_grad():
        for inputs, targets in tqdm(loader, leave=False):
            inputs = inputs.to(config.DEVICE)
            _, features = model(inputs)
            feats.append(features.cpu().numpy())
            labels.append(targets.numpy())
    return np.vstack(feats), np.vstack(labels).ravel()

X_train_feats, y_train_labels = extract_features(train_feat_loader, model)
X_test_feats, y_test_labels = extract_features(test_loader, model)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feats)
X_test_scaled = scaler.transform(X_test_feats)

print("Fitting SVM Model...")
svm_model = SVC(kernel=config.SVM_KERNEL, C=config.SVM_C, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train_labels)

# Evaluation
y_pred = svm_model.predict(X_test_scaled)
y_proba = svm_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test_labels, y_pred)
precision = precision_score(y_test_labels, y_pred)
recall = recall_score(y_test_labels, y_pred)
f1 = f1_score(y_test_labels, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test_labels, y_pred).ravel()
roc_auc = auc(*roc_curve(y_test_labels, y_proba)[:2])

print(f"Test Accuracy : {accuracy*100:.4f}%")
print(f"Test Recall   : {recall*100:.4f}%")
print(f"Test AUC      : {roc_auc:.4f}")
print("-" * 30)
print(classification_report(y_test_labels, y_pred, target_names=['Uninfected', 'Parasitized'], digits=4))

# ============================================================================
# 8. SAVING RESULTS
# ============================================================================
# Autonaming based on split ratio (model80_...)
name_tag = f"model{int(config.TRAIN_RATIO*100)}_{config.FEATURE_DIMENSION}D_{config.MAX_EPOCHS}"

# 1. Generate & Save Training Graph
epochs = [h['Epoch'] for h in history_log]
t_loss = [h['Train_Loss'] for h in history_log]
v_loss = [h['Val_Loss'] for h in history_log]
v_acc = [h['Val_Accuracy'] for h in history_log]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, t_loss, label='Train Loss', marker='.')
plt.plot(epochs, v_loss, label='Val Loss', marker='.')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, v_acc, label='Val Accuracy', color='green', marker='.')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

graph_path = os.path.join(config.RESULTS_PATH, f'Training_Graph_{name_tag}.png')
plt.tight_layout()
plt.savefig(graph_path)
plt.close()
print(f"Saved Training Graph: {graph_path}")

# 2. Save Confusion Matrix Plot
plt.figure(figsize=(6, 5))
sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Uninfected', 'Parasitized'], yticklabels=['Uninfected', 'Parasitized'])
plt.title(f'Confusion Matrix ({name_tag})')
cm_path = os.path.join(config.RESULTS_PATH, f'CM_{name_tag}.png')
plt.savefig(cm_path)
plt.close()
print(f"Saved Confusion Matrix: {cm_path}")

# 3. Save ROC Curve
plt.figure(figsize=(8, 8))
plt.plot(*roc_curve(y_test_labels, y_proba)[:2], color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title(f'ROC Curve ({name_tag})')
plt.legend(loc="lower right")
roc_path = os.path.join(config.RESULTS_PATH, f'ROC_{name_tag}.png')
plt.savefig(roc_path)
plt.close()
print(f"Saved ROC Curve: {roc_path}")

# 4. Save Text Report
report_txt = f"""
# HYBRID CNN-SVM REPORT
Tag: {name_tag}
GPU Used: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}
Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}
Config: {config.FEATURE_DIMENSION} Dim, 224x224 Img, {int(config.TRAIN_RATIO*100)}/{int(config.VAL_RATIO*100)}/{int(config.TEST_RATIO*100)} Split
Epochs: {config.MAX_EPOCHS}
"""
txt_path = os.path.join(config.RESULTS_PATH, f'Result_{name_tag}.txt')
with open(txt_path, 'w') as f:
    f.write(report_txt)
print(f"Saved Text Report: {txt_path}")

# 5. Save Model
pkl_path = os.path.join(config.MODEL_SAVE_PATH, f'{name_tag}.pkl')
joblib.dump({
    'svm_model': svm_model,
    'scaler': scaler,
    'cnn_state_dict': model.state_dict(), 
    'metrics': {'accuracy': accuracy, 'auc': roc_auc, 'precision': precision,'recall': recall, 'f1': f1 },
    'config': {'feature_dim': config.FEATURE_DIMENSION, 'img_size': config.IMG_SIZE}
}, pkl_path)
print(f"Saved Model Payload: {pkl_path}")

print("\nAll processes completed successfully!")