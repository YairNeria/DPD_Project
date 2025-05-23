
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from scipy.io import loadmat # for mat files

# Fully connected model
#First FC is the linear to 40 neurons 
#Second FC is the 

class FC_Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 40) #input layer : 6 inputs -> 40 neurons
        self.relu = nn.ReLU() #ReLU activation  
        self.fc2 = nn.Linear(40, 2)  #Hidden layer 40 -> 2 outputs           
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)                
        return x


class Signal_Dataset(Dataset):

    def __init__(self,M=3):
        # Load and initialize the data
        mat = loadmat('for_DPD.mat')
        #X_n is the baseband signal TX1_BB
        # Y_n is the recorded distorted signal TX1_SISO 
        X = mat['TX1_BB'].squeeze()
        Y = mat['TX1_SISO'].squueze()
        self.M= M

        inputs=[]
        targets=[]

        #prepare the data to the FC_Model
        for n in range(M-1,len(X)):
            x_taps = X[n-M:-1]
            if len(x_taps)<M:
                continue
            input_vec=np.empty(2*M,dtype=np.float32)
            input_vec[0::2]=np.real(x_taps)
            input_vec[1::2]=np.imag(x_taps)
            inputs.append(input_vec)
            targets.append([np.real(Y[n],np.imag(Y[n]))])


        # Decomposition in tensors
        self.x_data = torch.tensor(np.stack(inputs), dtype=torch.float32)
        self.y_data = torch.tensor(np.stack(targets), dtype=torch.float32)
        self.n_samples = self.x_data.shape[0]

    #  From here it is just copy paste from the gitHub of the course
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# create dataset and model
dataset = Signal_Dataset(M=3)
loader= DataLoader(dataset,batch_size=64,shuffle=True)
model = FC_Model()

#set up training tools / ADAM = adaptive Moment estimation
loss_function=nn.MSELoss()
optimizer=optim.Adam(model.parameters(), lr=1e-3)

#training loop template 
n_epochs=20

for epoch in range(n_epochs):
    #Forward pass
    for batch_in, batch_tar in loader:
        outputs=model(batch_in)
        loss=loss_function(outputs,batch_tar)
    #Backward pass
    optimizer.zero_grad() #important!
    loss.backward()
    optimizer.step()

print(f"Epoch {epoch+1},Loss: {loss.item():.6f}")