#!/usr/bin/env python3
"""
PGD adversarial training script for both spectrogram images and raw waveform.
Saves per-epsilon CSV logs, best model weights and plots. Uses tqdm progress bars.
"""

import os
import csv
import time
from collections import defaultdict, Counter
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------- Configuration ----------------
WAV_DATASET_PATH = "../speech_commands_numbers_resampled"   # change if needed
SPEC_DATASET_PATH = "training/spectrograms"                # change if needed

BATCH_SIZE = 64
EPOCHS = 3
LR = 1e-3
PGD_EPSILONS = [0]         # example epsilons; set as desired
PGD_ITERS = 20
PGD_ALPHA = None                    # None -> adaptive α inside pgd_attack
PGD_RANDOM_START = True

# ---------------- PGD Attack (waveform version) ----------------
def pgd_attack(model, X, y, epsilon=0.05, alpha=None, iters=20, random_start=True):
    device = next(model.parameters()).device
    X = X.to(device)
    y = y.to(device)

    if alpha is None:
        alpha = epsilon / max(iters/2, 1)

    if random_start:
        X_adv = X.clone().detach() + torch.empty_like(X).uniform_(-epsilon, epsilon)
        X_adv = torch.clamp(X_adv, 0.0, 1.0)
    else:
        X_adv = X.clone().detach()

    X_adv.requires_grad = True

    for _ in range(iters):
        output = model(X_adv)
        loss = F.cross_entropy(output, y)
        model.zero_grad()
        loss.backward()
        grad = X_adv.grad.data
        X_adv.data = X_adv.data + alpha * torch.sign(grad)
        eta = torch.clamp(X_adv.data - X.data, min=-epsilon, max=epsilon)
        X_adv.data = torch.clamp(X.data + eta, 0.0, 1.0)
        if X_adv.grad is not None:
            X_adv.grad.data.zero_()

    return X_adv.detach()

# ---------------- Datasets ----------------
class RawAudioDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        self.classes = []
        self.class_counts = defaultdict(int)

        for entry in sorted(os.listdir(root_dir)):
            full_path = os.path.join(root_dir, entry)
            if os.path.isdir(full_path) and not entry.startswith('.') and 'checkpoints' not in entry:
                self.classes.append(entry)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            for file in sorted(os.listdir(cls_folder)):
                if file.endswith('.wav'):
                    path = os.path.join(cls_folder, file)
                    self.samples.append((path, self.class_to_idx[cls]))
                    self.class_counts[cls] += 1

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform, sr = torchaudio.load(path)  # (1, 8000)
        waveform = (waveform + 1.0) / 2.0     # Normalize to [0,1]
        return waveform, label

    def summary(self):
        print(f"Found {len(self.classes)} classes:")
        for cls in self.classes:
            print(f"  - Class '{cls}': {self.class_counts[cls]} files")
        print(f"Total samples: {len(self.samples)}")
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------- Models ----------------
class NN_spectrogram(nn.Module):
    def __init__(self, in_channels=1, out_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 6 * 6, 100)
        self.fc2 = nn.Linear(100, out_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class NN_waveform(nn.Module):
    def __init__(self, n_input=1, n_output=10, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv2d(n_input, n_channel, kernel_size=(1, 80), stride=(1, stride))
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.pool1 = nn.MaxPool2d((1, 4), ceil_mode=True)

        self.conv2 = nn.Conv2d(n_channel, n_channel, kernel_size=(1, 5))
        self.bn2 = nn.BatchNorm2d(n_channel)
        self.pool2 = nn.MaxPool2d((1, 4))

        self.conv3 = nn.Conv2d(n_channel, 2 * n_channel, kernel_size=(1, 3))
        self.bn3 = nn.BatchNorm2d(2 * n_channel)
        self.pool3 = nn.MaxPool2d((1, 4))

        self.conv4 = nn.Conv2d(2 * n_channel, 2 * n_channel, kernel_size=(1, 4))
        self.bn4 = nn.BatchNorm2d(2 * n_channel)
        self.pool4 = nn.MaxPool2d((1, 4), stride=(1, 4))

        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x); x = F.relu(self.bn1(x)); x = self.pool1(x)
        x = self.conv2(x); x = F.relu(self.bn2(x)); x = self.pool2(x)
        x = self.conv3(x); x = F.relu(self.bn3(x)); x = self.pool3(x)
        x = self.conv4(x); x = F.relu(self.bn4(x)); x = self.pool4(x)
        x = x.squeeze(2).squeeze(-1)
        return self.fc1(x)

# ---------------- Utils (CSV/plot) ----------------
def save_metrics_to_csv(csv_path, metrics, header=None):
    # Check if file exists and is empty
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    # print(write_header)
    # print(header != None) 
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header and header is not None:
            # print("Writing Header row")
            writer.writerow(header)
        else:
            # print("Writing Metric Row")
            writer.writerow(metrics)

def plot_and_save(folder, eps, train_loss_clean, train_loss_adv, train_loss_total,
                  val_loss_clean, val_loss_adv, val_loss_total,
                  train_acc_clean, train_acc_adv, train_acc_total,
                  val_acc_clean, val_acc_adv, val_acc_total,
                  best_row):
    # best_row: [train_loss_clean, train_loss_adv, train_loss_total, train_acc_clean, train_acc_adv, train_acc_total,
    #            val_loss_clean, val_loss_adv, val_loss_total, val_acc_clean, val_acc_adv, val_acc_total]
    best_idx = int(np.argmin(train_loss_total)) if len(train_loss_total)>0 else 0

    # Accuracy plot
    plt.figure(figsize=(10,5))
    plt.plot(train_acc_clean, label="Train Acc Clean")
    plt.plot(train_acc_adv, label="Train Acc Adv")
    plt.plot(train_acc_total, label="Train Acc Total")
    plt.plot(val_acc_clean, label="Val Acc Clean")
    plt.plot(val_acc_adv, label="Val Acc Adv")
    plt.plot(val_acc_total, label="Val Acc Total")
    if best_row is not None:
        plt.scatter(best_idx, best_row[3], color="red", s=60, label="Best Epoch")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title(f"Accuracy (eps={eps})")
    epochs2= len(train_acc_clean)
    plt.xticks(range(epochs2), range(1, epochs2+1))
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(folder, f"accuracy_eps_{eps}.png"))
    plt.close()

    # Loss plot
    plt.figure(figsize=(10,5))
    plt.plot(train_loss_clean, label="Train Loss Clean")
    plt.plot(train_loss_adv, label="Train Loss Adv")
    plt.plot(train_loss_total, label="Train Loss Total")
    plt.plot(val_loss_clean, label="Val Loss Clean")
    plt.plot(val_loss_adv, label="Val Loss Adv")
    plt.plot(val_loss_total, label="Val Loss Total")
    if best_row is not None:
        plt.scatter(best_idx, best_row[0], color="red", s=60, label="Best Epoch")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"Loss (eps={eps})")
    epochs2= len(train_loss_clean)
    plt.xticks(range(epochs2), range(1, epochs2+1))
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(folder, f"loss_eps_{eps}.png"))
    plt.close()

# numpy needed for best idx
import numpy as np

# ---------------- Training loop for spectrograms ----------------
def train_spectrogram(model, train_loader, val_loader, device, epsilons, main_folder,
                      epochs=EPOCHS, lr=LR, pgd_iters=PGD_ITERS):

    header = [
        "epoch",
        "train_loss_clean","train_loss_adv","train_loss_total",
        "train_acc_clean","train_acc_adv","train_acc_total",
        "val_loss_clean","val_loss_adv","val_loss_total",
        "val_acc_clean","val_acc_adv","val_acc_total"
    ]

    for eps in epsilons:
        folder = os.path.join(main_folder, "spectrograms", f"eps_{eps}")
        os.makedirs(folder, exist_ok=True)
        csv_file = os.path.join(folder, f"log_spectrogram_eps_{eps}.csv")

        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, verbose=True)
        criterion = nn.CrossEntropyLoss()

        # metrics lists
        train_loss_clean, train_loss_adv, train_loss_total = [], [], []
        train_acc_clean, train_acc_adv, train_acc_total = [], [], []
        val_loss_clean, val_loss_adv, val_loss_total = [], [], []
        val_acc_clean, val_acc_adv, val_acc_total = [], [], []

        best_val_loss = float("inf")
        best_row = None
        epochs_no_improve = 0
        patience = 1e9

        # write header
        write_header = not os.path.exists(csv_file)
        if write_header:
            save_metrics_to_csv(csv_file, header, header=header)

        for epoch in range(1, epochs+1):
            model.train()
            total_loss_clean = total_loss_adv = total_loss_total = 0.0
            correct_clean = correct_adv = correct_total = 0
            total_samples = 0

            for x, y in tqdm(train_loader, desc=f"[Spec Train eps={eps}] Epoch {epoch}/{epochs}", leave=True):
                x, y = x.to(device), y.to(device)   # x: (B,1,32,32)

                optimizer.zero_grad()
                # clean forward
                out_clean = model(x)
                loss_clean = criterion(out_clean, y)

                # adversarial forward
                if eps > 0:
                    x_adv = pgd_attack(model, x, y, epsilon=eps, alpha=PGD_ALPHA, iters=pgd_iters, random_start=PGD_RANDOM_START)
                    out_adv = model(x_adv)
                    loss_adv = criterion(out_adv, y)
                    out_total = torch.cat([out_clean, out_adv], dim=0)
                    y_total = torch.cat([y, y], dim=0)
                    loss_total = criterion(out_total, y_total)
                else:
                    out_adv = None
                    loss_adv = torch.tensor(0.0)
                    out_total, y_total = out_clean, y
                    loss_total = loss_clean

                loss_total.backward()
                optimizer.step()

                batch_n = y.size(0)
                total_samples += batch_n
                total_loss_clean += loss_clean.item()
                total_loss_adv += loss_adv.item()
                total_loss_total += loss_total.item()
                correct_clean += (out_clean.argmax(1) == y).sum().item()
                if out_adv is not None:
                    correct_adv += (out_adv.argmax(1) == y).sum().item()
                correct_total += (out_total.argmax(1) == y_total).sum().item()

            # compute train epoch averages
            train_loss_clean.append(total_loss_clean / len(train_loader))
            train_loss_adv.append(total_loss_adv / len(train_loader))
            train_loss_total.append(total_loss_total / len(train_loader))
            train_acc_clean.append(100.0 * correct_clean / total_samples)
            train_acc_adv.append(100.0 * correct_adv / total_samples)
            train_acc_total.append(100.0 * correct_total / (total_samples * 2 if eps>0 else total_samples))

            # validation
            model.eval()
            v_clean_loss = v_adv_loss = v_comb_loss = 0.0
            v_clean_correct = v_adv_correct = v_comb_correct = 0
            val_total = 0

            # compute clean stats
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc=f"[Spec Val clean eps={eps}] Epoch {epoch}/{epochs}", leave=True):
                    x, y = x.to(device), y.to(device)
                    out_clean = model(x)
                    loss_clean = criterion(out_clean, y)
                    v_clean_loss += loss_clean.item()
                    v_clean_correct += (out_clean.argmax(1) == y).sum().item()
                    val_total += y.size(0)

            # compute adv stats if eps>0 (need grads for PGD generation)
            if eps > 0:
                for x, y in tqdm(val_loader, desc=f"[Spec Val adv eps={eps}] Epoch {epoch}/{epochs}", leave=True):
                    x, y = x.to(device), y.to(device)
                    x_adv = pgd_attack(model, x, y, epsilon=eps, alpha=PGD_ALPHA, iters=pgd_iters, random_start=PGD_RANDOM_START)
                    with torch.no_grad():
                        out_adv = model(x_adv)
                        loss_adv = criterion(out_adv, y)
                        v_adv_loss += loss_adv.item()
                        v_adv_correct += (out_adv.argmax(1) == y).sum().item()
                        # combined counted as avg of clean+adv
                        v_comb_loss += 0.5 * (criterion(model(x), y).item() + loss_adv.item())
                        v_comb_correct += ((0.5*(model(x) + out_adv)).argmax(1) == y).sum().item()
            else:
                v_adv_loss = 0.0
                v_adv_correct = 0
                v_comb_loss = v_clean_loss
                v_comb_correct = v_clean_correct

            # append validation metrics
            val_loss_clean.append(v_clean_loss / len(val_loader))
            val_loss_adv.append(v_adv_loss / len(val_loader))
            val_loss_total.append(v_comb_loss / len(val_loader))
            val_acc_clean.append(100.0 * v_clean_correct / (val_total if val_total>0 else 1))
            val_acc_adv.append(100.0 * v_adv_correct / (val_total if val_total>0 else 1))
            val_acc_total.append(100.0 * v_comb_correct / (val_total if val_total>0 else 1))

            # log row
            metrics_row = [
                epoch,
                train_loss_clean[-1], train_loss_adv[-1], train_loss_total[-1],
                train_acc_clean[-1], train_acc_adv[-1], train_acc_total[-1],
                val_loss_clean[-1], val_loss_adv[-1], val_loss_total[-1],
                val_acc_clean[-1], val_acc_adv[-1], val_acc_total[-1]
            ]
            save_metrics_to_csv(csv_file, metrics_row)

            # save best by val combined loss
            if val_loss_total[-1] < best_val_loss:
                best_val_loss = val_loss_total[-1]
                epochs_no_improve = 0
                best_row = metrics_row[1:]  # exclude epoch
                torch.save(model.state_dict(), os.path.join(folder, f"best_model_eps{eps}.pth"))
            else:
                epochs_no_improve += 1

            scheduler.step(val_loss_total[-1])
            if epochs_no_improve >= 1e9:
                break

        # save summary
        summary_csv = os.path.join(main_folder, "best_summary_spectrogram.csv")
        header_summary = ["epsilon"] + header[1:]
        write_header = not os.path.exists(summary_csv)
        with open(summary_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header_summary)
            writer.writerow([eps] + best_row)

        # create plots (best_row used to highlight)
        plot_and_save(folder, eps,
                      train_loss_clean, train_loss_adv, train_loss_total,
                      val_loss_clean, val_loss_adv, val_loss_total,
                      train_acc_clean, train_acc_adv, train_acc_total,
                      val_acc_clean, val_acc_adv, val_acc_total,
                      best_row)

# ---------------- Training loop for waveforms ----------------
def train_waveform(model, train_loader, val_loader, device, epsilons, main_folder,
                   epochs=EPOCHS, lr=LR, pgd_iters=PGD_ITERS):

    header = [
        "epoch",
        "train_loss_clean","train_loss_adv","train_loss_total",
        "train_acc_clean","train_acc_adv","train_acc_total",
        "val_loss_clean","val_loss_adv","val_loss_total",
        "val_acc_clean","val_acc_adv","val_acc_total"
    ]

    for eps in epsilons:
        folder = os.path.join(main_folder, "waveforms", f"eps_{eps}")
        os.makedirs(folder, exist_ok=True)
        csv_file = os.path.join(folder, f"log_waveform_eps_{eps}.csv")

        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, verbose=True)
        criterion = nn.CrossEntropyLoss()

        # metrics lists
        train_loss_clean, train_loss_adv, train_loss_total = [], [], []
        train_acc_clean, train_acc_adv, train_acc_total = [], [], []
        val_loss_clean, val_loss_adv, val_loss_total = [], [], []
        val_acc_clean, val_acc_adv, val_acc_total = [], [], []

        best_val_loss = float("inf")
        best_row = None
        epochs_no_improve = 0
        patience = 1e9

        # write header
        write_header = not os.path.exists(csv_file)
        if write_header:
            save_metrics_to_csv(csv_file, header, header=header)

        for epoch in range(1, epochs+1):
            model.train()
            total_loss_clean = total_loss_adv = total_loss_total = 0.0
            correct_clean = correct_adv = correct_total = 0
            total_samples = 0

            for X, y in tqdm(train_loader, desc=f"[Wave Train eps={eps}] Epoch {epoch}/{epochs}", leave=True):
                X, y = X.to(device), y.to(device)
                X = X.unsqueeze(2)   # (B,1,1,8000)

                optimizer.zero_grad()

                out_clean = model(X)
                loss_clean = criterion(out_clean, y)

                if eps > 0:
                    X_adv = pgd_attack(model, X, y, epsilon=eps, alpha=PGD_ALPHA, iters=pgd_iters, random_start=PGD_RANDOM_START)
                    out_adv = model(X_adv)
                    loss_adv = criterion(out_adv, y)
                    out_total = torch.cat([out_clean, out_adv], dim=0)
                    y_total = torch.cat([y, y], dim=0)
                    loss_total = criterion(out_total, y_total)
                else:
                    out_adv = None
                    loss_adv = torch.tensor(0.0)
                    out_total, y_total = out_clean, y
                    loss_total = loss_clean

                loss_total.backward()
                optimizer.step()

                batch_n = y.size(0)
                total_samples += batch_n
                total_loss_clean += loss_clean.item()
                total_loss_adv += loss_adv.item()
                total_loss_total += loss_total.item()
                correct_clean += (out_clean.argmax(1) == y).sum().item()
                if out_adv is not None:
                    correct_adv += (out_adv.argmax(1) == y).sum().item()
                correct_total += (out_total.argmax(1) == y_total).sum().item()

            # compute train epoch averages
            train_loss_clean.append(total_loss_clean / len(train_loader))
            train_loss_adv.append(total_loss_adv / len(train_loader))
            train_loss_total.append(total_loss_total / len(train_loader))
            train_acc_clean.append(100.0 * correct_clean / total_samples)
            train_acc_adv.append(100.0 * correct_adv / total_samples)
            train_acc_total.append(100.0 * correct_total / (total_samples * 2 if eps>0 else total_samples))

            # validation
            model.eval()
            v_clean_loss = v_adv_loss = v_comb_loss = 0.0
            v_clean_correct = v_adv_correct = v_comb_correct = 0
            val_total = 0

            # clean val (no grad)
            with torch.no_grad():
                for X, y in tqdm(val_loader, desc=f"[Wave Val clean eps={eps}] Epoch {epoch}/{epochs}", leave=True):
                    X, y = X.to(device), y.to(device)
                    X = X.unsqueeze(2)
                    out_clean = model(X)
                    loss_clean = criterion(out_clean, y)
                    v_clean_loss += loss_clean.item()
                    v_clean_correct += (out_clean.argmax(1) == y).sum().item()
                    val_total += y.size(0)

            # adv val (generate with grads)
            if eps > 0:
                for X, y in tqdm(val_loader, desc=f"[Wave Val adv eps={eps}] Epoch {epoch}/{epochs}", leave=True):
                    X, y = X.to(device), y.to(device)
                    X = X.unsqueeze(2)
                    X_adv = pgd_attack(model, X, y, epsilon=eps, alpha=PGD_ALPHA, iters=pgd_iters, random_start=PGD_RANDOM_START)
                    with torch.no_grad():
                        out_adv = model(X_adv)
                        loss_adv = criterion(out_adv, y)
                        v_adv_loss += loss_adv.item()
                        v_adv_correct += (out_adv.argmax(1) == y).sum().item()
                        v_comb_loss += 0.5 * (criterion(model(X), y).item() + loss_adv.item())
                        v_comb_correct += ((0.5*(model(X) + out_adv)).argmax(1) == y).sum().item()
            else:
                v_adv_loss = 0.0
                v_adv_correct = 0
                v_comb_loss = v_clean_loss
                v_comb_correct = v_clean_correct

            val_loss_clean.append(v_clean_loss / len(val_loader))
            val_loss_adv.append(v_adv_loss / len(val_loader))
            val_loss_total.append(v_comb_loss / len(val_loader))
            val_acc_clean.append(100.0 * v_clean_correct / (val_total if val_total>0 else 1))
            val_acc_adv.append(100.0 * v_adv_correct / (val_total if val_total>0 else 1))
            val_acc_total.append(100.0 * v_comb_correct / (val_total if val_total>0 else 1))

            # log row
            metrics_row = [
                epoch,
                train_loss_clean[-1], train_loss_adv[-1], train_loss_total[-1],
                train_acc_clean[-1], train_acc_adv[-1], train_acc_total[-1],
                val_loss_clean[-1], val_loss_adv[-1], val_loss_total[-1],
                val_acc_clean[-1], val_acc_adv[-1], val_acc_total[-1]
            ]
            save_metrics_to_csv(csv_file, metrics_row)

            # save best by val combined loss
            if val_loss_total[-1] < best_val_loss:
                best_val_loss = val_loss_total[-1]
                epochs_no_improve = 0
                best_row = metrics_row[1:]
                torch.save(model.state_dict(), os.path.join(folder, f"best_model_eps{eps}.pth"))
            else:
                epochs_no_improve += 1

            scheduler.step(val_loss_total[-1])
            if epochs_no_improve >= patience:
                break

        # save summary
        summary_csv = os.path.join(main_folder, "best_summary_waveform.csv")
        header_summary = ["epsilon"] + header[1:]
        write_header = not os.path.exists(summary_csv)
        with open(summary_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header_summary)
            writer.writerow([eps] + best_row)

        # create plots
        plot_and_save(folder, eps,
                      train_loss_clean, train_loss_adv, train_loss_total,
                      val_loss_clean, val_loss_adv, val_loss_total,
                      train_acc_clean, train_acc_adv, train_acc_total,
                      val_acc_clean, val_acc_adv, val_acc_total,
                      best_row)


# ---------------- Main ----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_folder = f"runs_{timestamp}"
    os.makedirs(main_folder, exist_ok=True)

    # --- Spectrogram dataset ---
    spec_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    spec_dataset = datasets.ImageFolder(root=SPEC_DATASET_PATH, transform=spec_transform)
    spec_train_size = int(0.8 * len(spec_dataset))
    spec_val_size = len(spec_dataset) - spec_train_size
    spec_train, spec_val = random_split(spec_dataset, [spec_train_size, spec_val_size])
    spec_train_loader = DataLoader(spec_train, batch_size=BATCH_SIZE, shuffle=True)
    spec_val_loader = DataLoader(spec_val, batch_size=BATCH_SIZE, shuffle=False)

    # Get the number of classes
    num_classes = len(spec_dataset.classes)
    print(f"Found {num_classes} classes:")

    # Count samples per class
    class_counts = Counter(spec_dataset.targets)

    # Print number of samples for each class
    for idx, class_name in enumerate(spec_dataset.classes):
        print(f"  - Class '{class_name}': {class_counts[idx]} files")
    
    # instantiate model with correct out_dim
    spec_out_dim = len(spec_dataset.classes)
    spec_model = NN_spectrogram(in_channels=1, out_dim=spec_out_dim)
    print(f"Number of model's parameters: {count_parameters(spec_model)}")

    # run spectrogram PGD training
    train_spectrogram(spec_model, spec_train_loader, spec_val_loader, device, PGD_EPSILONS, main_folder, epochs=EPOCHS, lr=LR, pgd_iters=PGD_ITERS)

    # --- Waveform dataset ---
    wav_dataset = RawAudioDataset(WAV_DATASET_PATH)
    wav_dataset.summary()
    wav_train_size = int(0.8 * len(wav_dataset))
    wav_val_size = len(wav_dataset) - wav_train_size
    wav_train, wav_val = random_split(wav_dataset, [wav_train_size, wav_val_size])
    wave_train_loader = DataLoader(wav_train, batch_size=BATCH_SIZE, shuffle=True)
    wave_val_loader = DataLoader(wav_val, batch_size=BATCH_SIZE, shuffle=False)

    wave_out_dim = len(wav_dataset.classes)
    wave_model = NN_waveform(n_input=1, n_output=wave_out_dim)
    print(f"Number of model's parameters: {count_parameters(wave_model)}")

    # run waveform PGD training
    train_waveform(wave_model, wave_train_loader, wave_val_loader, device, PGD_EPSILONS, main_folder, epochs=EPOCHS, lr=LR, pgd_iters=PGD_ITERS)

    print("All runs finished. Outputs saved in:", main_folder)

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_results(RESULTS_DIR, summary_file, title_prefix=""):
    SUMMARY_CSV = os.path.join(RESULTS_DIR, summary_file)

    if not os.path.exists(SUMMARY_CSV):
        print(f"No summary file found at {SUMMARY_CSV}")
        return

    df = pd.read_csv(SUMMARY_CSV)
    df = df.sort_values(by="epsilon").reset_index(drop=True)

    # ----------------------
    # Accuracy Plots (separate)
    # ----------------------
    plt.figure(figsize=(8,5))
    plt.plot(df["epsilon"], df["train_acc_clean"], marker="o", label="Train Acc Clean")
    plt.plot(df["epsilon"], df["train_acc_adv"], marker="o", label="Train Acc Adv")
    plt.plot(df["epsilon"], df["train_acc_total"], marker="o", label="Train Acc Total")
    plt.xlabel("Epsilon"); plt.ylabel("Accuracy (%)")
    plt.title(f"{title_prefix} Training Accuracy vs Epsilon")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"train_accuracy_vs_epsilon_{title_prefix}.png"))
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(df["epsilon"], df["val_acc_clean"], marker="o", label="Val Acc Clean")
    plt.plot(df["epsilon"], df["val_acc_adv"], marker="o", label="Val Acc Adv")
    plt.plot(df["epsilon"], df["val_acc_total"], marker="o", label="Val Acc Total")
    plt.xlabel("Epsilon"); plt.ylabel("Accuracy (%)")
    plt.title(f"{title_prefix} Validation Accuracy vs Epsilon")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"val_accuracy_vs_epsilon_{title_prefix}.png"))
    plt.close()

    # ----------------------
    # Loss Plots (separate)
    # ----------------------
    plt.figure(figsize=(8,5))
    plt.plot(df["epsilon"], df["train_loss_clean"], marker="o", label="Train Loss Clean")
    plt.plot(df["epsilon"], df["train_loss_adv"], marker="o", label="Train Loss Adv")
    plt.plot(df["epsilon"], df["train_loss_total"], marker="o", label="Train Loss Total")
    plt.xlabel("Epsilon"); plt.ylabel("Loss")
    plt.title(f"{title_prefix} Training Loss vs Epsilon")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"train_loss_vs_epsilon_{title_prefix}.png"))
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(df["epsilon"], df["val_loss_clean"], marker="o", label="Val Loss Clean")
    plt.plot(df["epsilon"], df["val_loss_adv"], marker="o", label="Val Loss Adv")
    plt.plot(df["epsilon"], df["val_loss_total"], marker="o", label="Val Loss Total")
    plt.xlabel("Epsilon"); plt.ylabel("Loss")
    plt.title(f"{title_prefix} Validation Loss vs Epsilon")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"val_loss_vs_epsilon_{title_prefix}.png"))
    plt.close()

    # ----------------------
    # Combined Accuracy Plot
    # ----------------------
    plt.figure(figsize=(10,6))
    # Training
    plt.plot(df["epsilon"], df["train_acc_clean"], marker="o", linestyle="--", label="Train Acc Clean")
    plt.plot(df["epsilon"], df["train_acc_adv"], marker="o", linestyle="--", label="Train Acc Adv")
    plt.plot(df["epsilon"], df["train_acc_total"], marker="o", linestyle="--", label="Train Acc Total")
    # Validation
    plt.plot(df["epsilon"], df["val_acc_clean"], marker="s", label="Val Acc Clean")
    plt.plot(df["epsilon"], df["val_acc_adv"], marker="s", label="Val Acc Adv")
    plt.plot(df["epsilon"], df["val_acc_total"], marker="s", label="Val Acc Total")

    plt.xlabel("Epsilon"); plt.ylabel("Accuracy (%)")
    plt.title(f"{title_prefix} Training & Validation Accuracy vs Epsilon")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"combined_accuracy_vs_epsilon_{title_prefix}.png"))
    plt.close()

    # ----------------------
    # Combined Loss Plot
    # ----------------------
    plt.figure(figsize=(10,6))
    # Training
    plt.plot(df["epsilon"], df["train_loss_clean"], marker="o", linestyle="--", label="Train Loss Clean")
    plt.plot(df["epsilon"], df["train_loss_adv"], marker="o", linestyle="--", label="Train Loss Adv")
    plt.plot(df["epsilon"], df["train_loss_total"], marker="o", linestyle="--", label="Train Loss Total")
    # Validation
    plt.plot(df["epsilon"], df["val_loss_clean"], marker="s", label="Val Loss Clean")
    plt.plot(df["epsilon"], df["val_loss_adv"], marker="s", label="Val Loss Adv")
    plt.plot(df["epsilon"], df["val_loss_total"], marker="s", label="Val Loss Total")

    plt.xlabel("Epsilon"); plt.ylabel("Loss")
    plt.title(f"{title_prefix} Training & Validation Loss vs Epsilon")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"combined_loss_vs_epsilon_{title_prefix}.png"))
    plt.close()

    print(f"✅ Plots saved in {RESULTS_DIR} for {title_prefix}")

# After waveform PGD training
plot_training_results(main_folder, summary_file="best_summary_waveform.csv", title_prefix="Waveform")

# After spectrogram PGD training
plot_training_results(main_folder, summary_file="best_summary_spectrogram.csv", title_prefix="Spectrogram")
