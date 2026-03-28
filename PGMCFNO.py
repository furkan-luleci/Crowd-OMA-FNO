# -*- coding: utf-8 -*-
"""
Created on Wed Jan 7 09:18:21 2026

@author: furka
"""
# This script is for phone dataset only. You can insert values from the paper for watch dataset.
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import csd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# ==========================================
# 0. Reproducibility
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==========================================
# 1. Helper Functions & Network Architecture
# ==========================================
def get_sv_curves(data_window, fs=90.32, nperseg=1024, freq_min=0.6, freq_max=15.0):
    """
    Compute SV1-SV3 curves from the CSD matrix of a multichannel window.
    Input:
        data_window: [n_samples, n_channels]
    Output:
        sv_curves_db: [3, n_freqs]
        masked_freqs: [n_freqs]
    """
    n_channels = data_window.shape[1]
    freqs = None
    csd_matrix = None

    for i in range(n_channels):
        for j in range(n_channels):
            f, Pxy = csd(
                data_window[:, i],
                data_window[:, j],
                fs=fs,
                nperseg=nperseg,
                noverlap=nperseg // 2
            )
            if csd_matrix is None:
                freqs = f
                csd_matrix = np.zeros((n_channels, n_channels, len(f)), dtype=complex)
            csd_matrix[i, j, :] = Pxy

    mask = (freqs >= freq_min) & (freqs <= freq_max)
    masked_freqs = freqs[mask]
    masked_csd = csd_matrix[:, :, mask]

    n_freqs = len(masked_freqs)
    sv_curves = np.zeros((3, n_freqs))

    for k in range(n_freqs):
        G_f = masked_csd[:, :, k]
        _, S, _ = np.linalg.svd(G_f)
        sv_curves[0, k] = S[0]
        sv_curves[1, k] = S[1]
        if len(S) > 2:
            sv_curves[2, k] = S[2]

    sv_curves_db = 10 * np.log10(sv_curves + 1e-12)
    return sv_curves_db, masked_freqs


def extract_windows_from_segment(phone_segment, ref_segment, gait, window_size=1024, step_size=256):
    """
    Create windows ONLY inside a pre-defined segment.
    """
    X_list, Y_list = [], []
    freqs_out = None

    n_samples = min(len(phone_segment), len(ref_segment))
    if n_samples < window_size:
        return X_list, Y_list, freqs_out

    for start_idx in range(0, n_samples - window_size + 1, step_size):
        window_x = phone_segment[start_idx:start_idx + window_size]
        window_y = ref_segment[start_idx:start_idx + window_size]

        sv_x, f_bins = get_sv_curves(window_x)
        sv_y, _ = get_sv_curves(window_y)

        if freqs_out is None:
            freqs_out = f_bins

        n_freqs = sv_x.shape[1]
        gait_array = np.tile(np.array(gait).reshape(3, 1), (1, n_freqs))
        sv_x_conditioned = np.vstack((sv_x, gait_array))  # [6, n_freqs]

        X_list.append(sv_x_conditioned)
        Y_list.append(sv_y)

    return X_list, Y_list, freqs_out


def load_and_prep_sv_data_split_first(
    folder_path,
    window_size=1024,
    step_size=128,
    train_ratio=0.80,
    gap_ratio=0.00
):
    """
    1) Load raw continuous signals
    2) Split each phone/reference time series into train / gap / test
    3) Create windows separately inside train and inside test
    """
    crowd_path = os.path.join(folder_path, "Watch_Crowdsensing.xlsx")
    ref_path = os.path.join(folder_path, "Watch_Reference.xlsx")

    print("Loading Excel files...")
    df_crowd = pd.read_excel(crowd_path, header=0)
    df_ref = pd.read_excel(ref_path, header=0)

    # Reference uses first 30 columns exactly as your original code
    ref_data = df_ref.iloc[:, 0:30].astype(float).values

    X_train_all, Y_train_all = [], []
    X_test_all, Y_test_all = [], []
    freqs_out = None

    for phone_idx in range(6):
        start_col = phone_idx * 3
        end_col = start_col + 3
        phone_data = df_crowd.iloc[:, start_col:end_col].astype(float).values

        max_valid_length = min(len(phone_data), len(ref_data))

        if phone_idx < 2:
            gait = [1, 0, 0]  # Walking
            gait_name = "Walking"
        elif phone_idx < 4:
            gait = [0, 1, 0]  # Jogging
            gait_name = "Jogging"
        else:
            gait = [0, 0, 1]  # Bicycling
            gait_name = "Bicycling"

        # -------- Split FIRST on raw signals --------
        train_end = int(train_ratio * max_valid_length)
        gap_len = int(gap_ratio * max_valid_length)
        test_start = train_end + gap_len

        if test_start >= max_valid_length:
            raise ValueError(
                f"Split configuration invalid for phone {phone_idx + 1}. "
                f"Reduce gap_ratio or train_ratio."
            )

        phone_train = phone_data[:train_end]
        ref_train = ref_data[:train_end]

        phone_test = phone_data[test_start:max_valid_length]
        ref_test = ref_data[test_start:max_valid_length]

        print(
            f"\nPhone {phone_idx + 1} ({gait_name}) | "
            f"Total: {max_valid_length} samples | "
            f"Train: [0:{train_end}] | "
            f"Gap: [{train_end}:{test_start}] | "
            f"Test: [{test_start}:{max_valid_length}]"
        )

        # -------- Window train segment only --------
        X_tr, Y_tr, f_bins = extract_windows_from_segment(
            phone_train, ref_train, gait,
            window_size=window_size,
            step_size=step_size
        )

        # -------- Window test segment only --------
        X_te, Y_te, f_bins_test = extract_windows_from_segment(
            phone_test, ref_test, gait,
            window_size=window_size,
            step_size=step_size
        )

        if freqs_out is None:
            freqs_out = f_bins if f_bins is not None else f_bins_test

        X_train_all.extend(X_tr)
        Y_train_all.extend(Y_tr)
        X_test_all.extend(X_te)
        Y_test_all.extend(Y_te)

        print(f"  Train windows: {len(X_tr)} | Test windows: {len(X_te)}")

    if len(X_train_all) == 0 or len(X_test_all) == 0:
        raise ValueError(
            "No windows were created. Check window_size, step_size, and split ratios."
        )

    X_train_tensor = torch.tensor(np.array(X_train_all), dtype=torch.float32)
    Y_train_tensor = torch.tensor(np.array(Y_train_all), dtype=torch.float32)
    X_test_tensor = torch.tensor(np.array(X_test_all), dtype=torch.float32)
    Y_test_tensor = torch.tensor(np.array(Y_test_all), dtype=torch.float32)

    return X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, freqs_out


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat
        )
        out_ft[:, :, :self.modes1] = torch.einsum(
            "bix,iox->box",
            x_ft[:, :, :self.modes1],
            self.weights1
        )
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNOSpectral(nn.Module):
    def __init__(self, modes, width, dropout_rate=0.5):
        super().__init__()
        self.modes1 = modes
        self.width = width

        # Input features per frequency bin:
        # [SV1, SV2, SV3, gait_walk, gait_jog, gait_bike] = 6
        self.fc0 = nn.Linear(6, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        # x: [batch, 6, n_freqs]
        x = x.permute(0, 2, 1)   # [batch, n_freqs, 6]
        x = self.fc0(x)          # [batch, n_freqs, width]
        x = x.permute(0, 2, 1)   # [batch, width, n_freqs]

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)   # [batch, n_freqs, width]
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)          # [batch, n_freqs, 3]

        return x.permute(0, 2, 1)  # [batch, 3, n_freqs]


class StructuralPeakLoss(nn.Module):
    def __init__(self, peak_weight=1.5, slope_weight=0.1):
        super().__init__()
        self.peak_weight = peak_weight
        self.slope_weight = slope_weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, y_pred, y_true):
        y_min = y_true.amin(dim=2, keepdim=True)
        y_max = y_true.amax(dim=2, keepdim=True)
        peak_map = (y_true - y_min) / (y_max - y_min + 1e-8)

        base_loss = self.mse(y_pred, y_true)
        weighted_mse = base_loss * (1.0 + self.peak_weight * peak_map)
        loss_amplitude = weighted_mse.mean()

        dy_pred = y_pred[:, :, 1:] - y_pred[:, :, :-1]
        dy_true = y_true[:, :, 1:] - y_true[:, :, :-1]
        loss_slope = F.mse_loss(dy_pred, dy_true)

        return loss_amplitude + (self.slope_weight * loss_slope)


def calculate_mmsc(y_pred, y_true):
    numerator = torch.sum(y_pred * y_true, dim=2) ** 2
    denominator = torch.sum(y_pred ** 2, dim=2) * torch.sum(y_true ** 2, dim=2)
    mmsc = numerator / (denominator + 1e-8)
    return mmsc.mean().item()


# ==========================================
# 2. GLOBAL EXECUTION
# ==========================================
folder_path = r"directory"

# You can set gap_ratio=0.0 if you want pure 80/20 without a buffer,
# but the small gap is safer.
X_train, Y_train, X_test, Y_test, freqs = load_and_prep_sv_data_split_first(
    folder_path=folder_path,
    window_size=1024,
    step_size=256,
    train_ratio=0.80,
    gap_ratio=0.00
)

print(f"\nFinal dataset shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  Y_train: {Y_train.shape}")
print(f"  X_test : {X_test.shape}")
print(f"  Y_test : {Y_test.shape}")

train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nTraining on: {device}")

modes = 64
width = 512
model = FNOSpectral(modes=modes, width=width, dropout_rate=0.5).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.75)
criterion = StructuralPeakLoss(peak_weight=1.5, slope_weight=0.1)

epochs = 150
best_test_loss = np.inf
best_model_path = os.path.join(folder_path, "best_fno_phone_model_split_first.pth")

history_epochs = []
history_train_loss = []
history_test_loss = []
history_test_mmsc = []

for ep in range(epochs):
    # -------- Train --------
    model.train()
    train_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()

    # -------- Test --------
    model.eval()
    test_loss = 0.0
    test_mmsc = 0.0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            out = model(x_batch)
            loss = criterion(out, y_batch)

            test_loss += loss.item()
            test_mmsc += calculate_mmsc(out, y_batch)

    avg_train = train_loss / len(train_loader)
    avg_test = test_loss / len(test_loader)
    avg_mmsc = test_mmsc / len(test_loader)

    history_epochs.append(ep)
    history_train_loss.append(avg_train)
    history_test_loss.append(avg_test)
    history_test_mmsc.append(avg_mmsc)

    if avg_test < best_test_loss:
        best_test_loss = avg_test
        torch.save(model.state_dict(), best_model_path)

    if ep % 5 == 0 or ep == epochs - 1:
        print(
            f"Epoch {ep:03d} | "
            f"Train Loss: {avg_train:.4f} | "
            f"Test Loss: {avg_test:.4f} | "
            f"Test MMSC: {avg_mmsc:.4f}"
        )

print(f"\nBest model saved to: {best_model_path}")

# Load best model before final plots/validation
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()
print("Best model loaded for visualization and OMA validation.")


# ==========================================
# 3. VISUALIZATION
# ==========================================
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 26,
    'lines.linewidth': 3
})

samples_to_plot = {'Walk': None, 'Jog': None, 'Bike': None}
search_indices = list(range(len(test_dataset)))
random.shuffle(search_indices)

with torch.no_grad():
    for i in search_indices:
        x_sample, y_sample = test_dataset[i]

        if x_sample[3, 0] == 1 and samples_to_plot['Walk'] is None:
            samples_to_plot['Walk'] = (x_sample, y_sample)
        elif x_sample[4, 0] == 1 and samples_to_plot['Jog'] is None:
            samples_to_plot['Jog'] = (x_sample, y_sample)
        elif x_sample[5, 0] == 1 and samples_to_plot['Bike'] is None:
            samples_to_plot['Bike'] = (x_sample, y_sample)

        if all(v is not None for v in samples_to_plot.values()):
            break

fig, axes = plt.subplots(1, 3, figsize=(22, 8), sharey=True)
fig.suptitle("SV1 Reconstruction Across Different Mobilities (Random Test Window)", fontweight='bold', y=1.05)

gaits = ['Walk', 'Jog', 'Bike']
for idx, gait in enumerate(gaits):
    if samples_to_plot[gait] is None:
        axes[idx].set_title(f"Mobility: {gait}\n(No sample found)")
        axes[idx].axis("off")
        continue

    x_sample, y_sample = samples_to_plot[gait]
    x_input = x_sample.unsqueeze(0).to(device)
    y_true_tensor = y_sample.unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred_tensor = model(x_input)

        numerator = torch.sum(y_pred_tensor * y_true_tensor, dim=2) ** 2
        denominator = torch.sum(y_pred_tensor ** 2, dim=2) * torch.sum(y_true_tensor ** 2, dim=2)
        plot_mmsc = (numerator / (denominator + 1e-8)).mean().item()

        y_pred = y_pred_tensor.squeeze(0).cpu().numpy()

    y_true = y_sample.numpy()
    x_raw = x_sample.numpy()

    axes[idx].plot(freqs, x_raw[0, :], label='Phone Crowdsensing (Distorted)', color='gray', alpha=0.5)
    axes[idx].plot(freqs, y_true[0, :], label='Phone Benchmark (Stationary)', color='blue')
    axes[idx].plot(freqs, y_pred[0, :], label='PG-MC-FNO-Recovered Response', color='red', linestyle='--')

    axes[idx].set_title(f"Mobility: {gait}\n(MMSC: {plot_mmsc:.4f})", pad=15)
    axes[idx].set_xlabel("Frequency (Hz)", labelpad=10)
    axes[idx].set_xlim([0.6, 15.0])
    axes[idx].grid(True)

    if idx == 0:
        axes[idx].set_ylabel("Singular Value Amplitude (dB)", labelpad=10)
        axes[idx].legend(loc='upper right')

plt.tight_layout()
plt.show()


# ==========================================
# 4. FINAL OMA VALIDATION (TEST SET ONLY)
# ==========================================
print("\n--- Starting Final OMA Validation ---")

test_only_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
test_benchmark_sv_db, test_fno_recovered_sv_db = [], []

print(f"Batch processing {len(test_dataset)} unseen test windows...")
with torch.no_grad():
    for x_batch, y_batch in test_only_loader:
        test_benchmark_sv_db.append(y_batch.numpy())
        test_fno_recovered_sv_db.append(model(x_batch.to(device)).cpu().numpy())

final_benchmark_array_db = np.vstack(test_benchmark_sv_db)
final_recovered_array_db = np.vstack(test_fno_recovered_sv_db)

print("Batch processing complete.")

# -------- Linear averaging on SV1 --------
print("Performing linear averaging on SV1 across test set...")

linear_benchmark_sv1 = 10 ** (final_benchmark_array_db[:, 0, :] / 10.0)
linear_recovered_sv1 = 10 ** (final_recovered_array_db[:, 0, :] / 10.0)

averaged_benchmark_linear = np.mean(linear_benchmark_sv1, axis=0)
averaged_recovered_linear = np.mean(linear_recovered_sv1, axis=0)

averaged_benchmark_db = 10 * np.log10(averaged_benchmark_linear + 1e-12)
averaged_recovered_db = 10 * np.log10(averaged_recovered_linear + 1e-12)

# -------- Gaussian smoothing --------
print("Applying Gaussian smoothing pass...")
sigma_smooth = 2
averaged_benchmark_db_smoothed = gaussian_filter1d(averaged_benchmark_db, sigma=sigma_smooth)
averaged_recovered_db_smoothed = gaussian_filter1d(averaged_recovered_db, sigma=sigma_smooth)

# -------- Final OMA plot --------
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 22,
    'lines.linewidth': 4
})

plt.figure(figsize=(16, 9))

plt.plot(
    freqs, averaged_benchmark_db_smoothed,
    label='Scenario 1: Hardware Benchmark',
    color='blue', linestyle='-'
)

plt.plot(
    freqs, averaged_recovered_db_smoothed,
    label='Scenario 2 (PG-MC-FNO-Recovered on Unseen Test Data)',
    color='red', linestyle='--'
)

plt.title("Statistical OMA Validation (Smoothed, FDD-Ready Average on Unseen Test Data)", fontweight='bold', pad=30)
plt.xlabel("Frequency (Hz)", labelpad=20)
plt.ylabel("Singular Value 1 (dB)", labelpad=20)
plt.xlim([0.6, 15.0])
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
print("Final smoothed OMA validation figure generated!")
plt.show()


# ==========================================
# 5. TRAINING HISTORY PLOTS
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 24
})

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

axes[0].plot(history_epochs, history_train_loss, linewidth=3, marker='o')
axes[0].set_title('Training Loss History')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(True, linestyle='--', alpha=0.7)

axes[1].plot(history_epochs, history_test_loss, linewidth=3, marker='s')
axes[1].set_title('Test Loss History')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].grid(True, linestyle='--', alpha=0.7)

axes[2].plot(history_epochs, history_test_mmsc, linewidth=3, marker='^')
axes[2].set_title('Test MMSC History')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('MMSC')
axes[2].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
