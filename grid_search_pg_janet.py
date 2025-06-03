import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time

from pg_janet import PGJanetRNN

# Dataset definition (with efficient conversion)
class PGJanetSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, mat_path='for_DPD.mat', seq_len=10):
        mat = loadmat(mat_path, squeeze_me=True)
        X = mat['TX1_BB']
        Y = mat['TX1_SISO']
        amplitudes = np.abs(X).astype(np.float32)
        phases = np.angle(X).astype(np.float32)
        targets = np.stack([Y.real, Y.imag], axis=-1).astype(np.float32)
        N = len(amplitudes)
        self.x_abs_seq = []
        self.theta_seq = []
        self.target_seq = []
        for i in range(N - seq_len):
            self.x_abs_seq.append(amplitudes[i:i+seq_len])
            self.theta_seq.append(phases[i:i+seq_len])
            self.target_seq.append(targets[i:i+seq_len])
        self.x_abs_seq = torch.from_numpy(np.array(self.x_abs_seq))
        self.theta_seq = torch.from_numpy(np.array(self.theta_seq))
        self.target_seq = torch.from_numpy(np.array(self.target_seq))
    def __len__(self):
        return self.x_abs_seq.shape[0]
    def __getitem__(self, idx):
        return self.x_abs_seq[idx].unsqueeze(-1), self.theta_seq[idx].unsqueeze(-1), self.target_seq[idx]

def nMSE_loss(y_pred, y_true):
    mse = torch.sum((y_pred - y_true) ** 2)
    denom = torch.sum(y_true ** 2) + 1e-8
    return mse / denom

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for x_abs, theta, targets in val_loader:
            x_abs, theta, targets = x_abs.to(device), theta.to(device), targets.to(device)
            outputs = model(x_abs, theta)
            loss = nMSE_loss(outputs, targets)
            total_loss += loss.item() * x_abs.size(0)
            total_samples += x_abs.size(0)
    model.train()
    return total_loss / total_samples

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameter grid (make these as large as you like)
    seq_lens = [8, 12, 16]
    hidden_sizes = [8, 16, 32]

    mat_path = 'for_DPD.mat'

    results = []  # To keep loss, time, etc.

    best_val_loss = float('inf')
    best_params = {}
    best_train_curve = None  # For plotting

    for seq_len in seq_lens:
        dataset = PGJanetSequenceDataset(mat_path, seq_len=seq_len)
        N = len(dataset)
        split = int(N * 0.8)
        indices = list(range(N))
        train_indices, val_indices = indices[:split], indices[split:]
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

        for hidden_size in hidden_sizes:
            print(f"Training seq_len={seq_len}, hidden_size={hidden_size}")
            model = PGJanetRNN(hidden_size=hidden_size).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            n_epochs = 5  # You can raise this for final training

            train_curve = []
            t0 = time.time()
            skip = False
            for epoch in range(n_epochs):
                epoch_loss = 0
                count = 0
                for batch_x_abs, batch_theta, batch_targets in train_loader:
                    batch_x_abs, batch_theta, batch_targets = (
                        batch_x_abs.to(device), batch_theta.to(device), batch_targets.to(device)
                    )
                    outputs = model(batch_x_abs, batch_theta)
                    loss = nMSE_loss(outputs, batch_targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    count += 1
                train_curve.append(epoch_loss / count)
                # Early skip if too slow
                if time.time() - t0 > 60:  # 60 seconds per combo max, change if you want
                    print(f"Combo seq_len={seq_len}, hidden_size={hidden_size} is too slow, skipping.")
                    skip = True
                    break

            if skip:
                continue

            val_loss = evaluate(model, val_loader, device)
            print(f"seq_len={seq_len}, hidden_size={hidden_size}, val_loss={val_loss:.6f}")

            results.append({
                'seq_len': seq_len,
                'hidden_size': hidden_size,
                'val_loss': val_loss,
                'val_loss_db': 10 * np.log10(val_loss),
                'train_curve': train_curve,
            })
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = {'seq_len': seq_len, 'hidden_size': hidden_size}
                best_train_curve = train_curve.copy()

    print("Best hyperparameters found:")
    print(best_params)
    print(f"Best validation nMSE: {best_val_loss:.6f}")

    # ---- Plotting Section ----
    # 1. Validation loss (dB) heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Make grid for heatmap
    grid = np.full((len(seq_lens), len(hidden_sizes)), np.nan)
    for r, seq_len in enumerate(seq_lens):
        for c, hidden_size in enumerate(hidden_sizes):
            for res in results:
                if res['seq_len'] == seq_len and res['hidden_size'] == hidden_size:
                    grid[r, c] = res['val_loss_db']

    plt.figure(figsize=(8,5))
    sns.heatmap(grid, annot=True, fmt=".1f", cmap="viridis",
                xticklabels=hidden_sizes, yticklabels=seq_lens)
    plt.xlabel("Hidden size")
    plt.ylabel("Sequence length")
    plt.title("Validation nMSE Loss (dB)\nLower is better")
    plt.tight_layout()
    plt.show()

    # 2. Training curve for best model
    if best_train_curve is not None:
        plt.figure()
        plt.plot(best_train_curve, marker='o')
        plt.title(f"Best Model Training nMSE vs Epochs\nseq_len={best_params['seq_len']} hidden={best_params['hidden_size']}")
        plt.xlabel("Epoch")
        plt.ylabel("Training nMSE")
        plt.yscale('log')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 3. Scatter plot (optional, for a quick overview)
    plt.figure()
    x = [f"{res['seq_len']}-{res['hidden_size']}" for res in results]
    y = [res['val_loss_db'] for res in results]
    plt.scatter(x, y, c='red')
    plt.title("All tried hyperparameters (nMSE in dB)")
    plt.ylabel("Validation nMSE (dB)")
    plt.xlabel("seq_len - hidden_size")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
