
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import scipy.io # Library to load the mat file

# Fully connected model

class FC_Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        # Hidden layer, 6->40. input of 3 timesteps, real and imagenary
        self.hidden = nn.Linear(6, 40)
        ## Output of one timestep, real and imagenary
        self.output = nn.Linear(40, 2)
        self.relu = nn.ReLU()             
    def forward(self, x):
        x = self.relu(self.hidden(x)) #Activate function just on the hidden layer    
        x = self.output(x)                
        return x


class Signal_Dataset(Dataset):

    def __init__(self):
        # Load and initialize the data
        file_path = r'C:\Users\User\Desktop\for_DPD.mat' # Change it to your path
        mat = scipy.io.loadmat(file_path)
        TX1_BB = mat['TX1_BB'] # I left names from the file because i dont know what is the original signal
        TX1_SISO = mat['TX1_SISO']
        # Decomposition into real and imaginary parts
        self.TX1_BB_real = torch.from_numpy(TX1_BB.real)
        self.TX1_SISO_real = torch.from_numpy(TX1_SISO.real)
        self.TX1_BB_imag = torch.from_numpy(TX1_BB.imag)
        self.TX1_SISO_imag = torch.from_numpy(TX1_SISO.imag)
        self.n_samples = TX1_BB.shape[0]

    #  From here it is just copy paste from the gitHub of the course
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# create dataset and model
dataset = Signal_Dataset()
model = FC_Model()
