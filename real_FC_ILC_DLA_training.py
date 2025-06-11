import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from scipy.io import loadmat
from fc_model_class import FC_Model, Signal_Dataset, FC_Trainer

"""
This script implements the ILC + FC model + DLA fine-tuning workflow:
1. ILC: Iteratively update the input to the PA using the FC model and measured PA output.
2. Train/validation split and training of FC model on ILC-refined data.
3. DLA fine-tuning: Further train the FC model using the ILC-refined data.
"""

# -----------------------------
# Hyperparameters
# -----------------------------
M = 3  # Number of taps for FC input
hidden_dim = 40
n_epochs = 30
batch_size = 64
val_ratio = 0.2
ilc_iterations = 5  # Number of ILC iterations
fine_tune_epochs = 10  # DLA fine-tuning epochs
mat_path = 'for_DPD.mat'

# -----------------------------
# Load full input/output from .mat
# -----------------------------
mat = loadmat(mat_path, squeeze_me=True)
X_full = mat['TX1_BB']  # Complex baseband input
Y_full = mat['TX1_SISO']  # Distorted output

# -----------------------------
# 1. ILC Integration
# -----------------------------
print('Starting ILC integration with FC model...')
# Initial predistorted input is the original input
X_ilc = X_full.copy()

# Instantiate FC model and load initial weights (random or pre-trained)
fc_model = FC_Model(input_dim=2*M, hidden_dim=hidden_dim, output_dim=2)
fc_model.eval()

for ilc_iter in range(ilc_iterations):
    print(f'ILC Iteration {ilc_iter+1}/{ilc_iterations}')
    # 1. Pass current input through PA (simulate by using Y_full as measured output)
    Y_measured = Y_full  # [N]
    # 2. Calculate error (desired - measured)
    error = X_full - Y_measured  # [N], complex
    # 3. Update predistorted input using FC model
    # Prepare input features for FC model (sliding window)
    inputs = []
    for n in range(M-1, len(X_ilc)):
        x_taps = X_ilc[n-M+1:n+1]
        input_vec = np.empty(2*M, dtype=np.float32)
        input_vec[0::2] = x_taps.real
        input_vec[1::2] = x_taps.imag
        inputs.append(input_vec)
    inputs_tensor = torch.from_numpy(np.stack(inputs)).float()  # [N-M+1, 2M]
    with torch.no_grad():
        model_output = fc_model(inputs_tensor).cpu().numpy()  # [N-M+1, 2]
    model_output_complex = model_output[:, 0] + 1j * model_output[:, 1]
    # 4. Update input for next iteration (ILC update rule)
    learning_rate = 0.7
    # Only update the valid region (M-1:)
    X_ilc[M-1:] = X_ilc[M-1:] + learning_rate * error[M-1:]

# After ILC, X_ilc is the refined predistorted input

# -----------------------------
# 2. Train/Validation Split and Training on ILC-refined data
# -----------------------------
print('Preparing dataset and training FC model on ILC-refined data...')
class ILCRefinedDataset(torch.utils.data.Dataset):
    def __init__(self, X_ilc, Y_target, M=3):
        self.M = M
        inputs, targets = [], []
        for n in range(M-1, len(X_ilc)):
            x_taps = X_ilc[n-M+1:n+1]
            input_vec = np.empty(2*M, dtype=np.float32)
            input_vec[0::2] = x_taps.real
            input_vec[1::2] = x_taps.imag
            inputs.append(input_vec)
            targets.append([Y_target[n].real, Y_target[n].imag])
        self.x_data = torch.from_numpy(np.stack(inputs)).float()
        self.y_data = torch.from_numpy(np.stack(targets)).float()
        self.n_samples = self.x_data.shape[0]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.n_samples

ilc_dataset = ILCRefinedDataset(X_ilc, X_full, M=M)
n_val = int(len(ilc_dataset) * val_ratio)
n_train = len(ilc_dataset) - n_val
train_set, val_set = random_split(ilc_dataset, [n_train, n_val])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

fc_model = FC_Model(input_dim=2*M, hidden_dim=hidden_dim, output_dim=2)
loss_function = nn.MSELoss()
optimizer = optim.Adam(fc_model.parameters(), lr=1e-3)

print('Training FC model on ILC-refined data...')
for epoch in range(n_epochs):
    fc_model.train()
    epoch_loss = 0.0
    for batch_in, batch_tar in train_loader:
        outputs = fc_model(batch_in)
        loss = loss_function(outputs, batch_tar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}")

# -----------------------------
# 3. DLA Fine-tuning (using ILC-refined data)
# -----------------------------
print('Starting DLA fine-tuning...')
fine_tune_loader = DataLoader(ilc_dataset, batch_size=batch_size, shuffle=True)
optimizer = optim.Adam(fc_model.parameters(), lr=1e-4)
for epoch in range(fine_tune_epochs):
    fc_model.train()
    epoch_loss = 0.0
    for batch_in, batch_tar in fine_tune_loader:
        outputs = fc_model(batch_in)
        loss = loss_function(outputs, batch_tar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(fine_tune_loader)
    print(f"DLA Fine-tune Epoch {epoch+1}/{fine_tune_epochs}, Loss: {epoch_loss:.6f}")
print('DLA fine-tuning complete.')

# Save the final model
torch.save(fc_model.state_dict(), 'fc_model_real_ilc_dla.pth')
print('Final FC model saved as fc_model_real_ilc_dla.pth')
