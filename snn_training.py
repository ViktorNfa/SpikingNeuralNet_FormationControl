import pickle 
import norse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split


### ------------------------------------------------------------------------------------------------------------ ###
## Load dataset

with open('dataset/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

features = torch.stack([torch.from_numpy(a["feature"]).float() for a in dataset.values()], dim=0)
labels = torch.stack([torch.from_numpy(a["label"]).float() for a in dataset.values()], dim=0)

print(features.shape)
print(labels.shape)
n_time_points = features[0].shape[1]


### ------------------------------------------------------------------------------------------------------------ ###
## Create the network and dataset

# Define the Network class
class Network(torch.nn.Module):
    def __init__(self, train_mode: bool):
        super(Network, self).__init__()
        
        
        time_constant1 = torch.nn.Parameter(torch.tensor([200.]))
        time_constant2 = torch.nn.Parameter(torch.tensor([300.]))
        time_constant3 = torch.nn.Parameter(torch.tensor([600.]))
        
        voltage1 = torch.nn.Parameter(torch.tensor([0.006]))
        voltage2 = torch.nn.Parameter(torch.tensor([0.008]))
        voltage3 = torch.nn.Parameter(torch.tensor([0.013]))


        # Define three different neuron layers with varying temporal dynamics
        lif_params_1 = norse.torch.LIFBoxParameters(tau_mem_inv= time_constant1 ,v_th = voltage1 )
        lif_params_2 = norse.torch.LIFBoxParameters(tau_mem_inv= time_constant2 ,v_th = voltage2 )
        lif_params_3 = norse.torch.LIFBoxParameters(tau_mem_inv= time_constant3 ,v_th = voltage3 )
        
        self.temporal_layer_1 = norse.torch.LIFBoxCell(p=lif_params_1)
        self.temporal_layer_2 = norse.torch.LIFBoxCell(p=lif_params_2)
        self.temporal_layer_3 = norse.torch.LIFBoxCell(p=lif_params_3)
        
        # lifting
        self.temporal_layer_1_lifted = norse.torch.Lift(self.temporal_layer_1)
        self.temporal_layer_2_lifted = norse.torch.Lift(self.temporal_layer_2)
        self.temporal_layer_3_lifted = norse.torch.Lift(self.temporal_layer_3)
            
        
        self.temporal_layer_1.register_parameter("time_constant",time_constant1)
        self.temporal_layer_1.register_parameter("voltage",voltage1)
        
        self.temporal_layer_2.register_parameter("time_constant",time_constant2)
        self.temporal_layer_2.register_parameter("voltage",voltage2)
        
        self.temporal_layer_3.register_parameter("time_constant",time_constant3)
        self.temporal_layer_3.register_parameter("voltage",voltage3)
    
        
        
        # First convolutional layer
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        
        # Third convolutional layer
        self.linear = torch.nn.Linear(in_features=10,out_features=2)
        
        self.train_mode = train_mode
        self.state_1 = None
        self.state_2 = None
        self.state_3 = None
        
    def forward(self, inputs:torch.Tensor):
        
        
        outputs = []
        if inputs.ndim == 2: # to deal with a batch
            inputs = inputs.unsqueeze(0)
        if inputs.ndim == 1: 
            inputs = inputs.unsqueeze(0)
            inputs = inputs.unsqueeze(2)
        
        for input in inputs:
            input = torch.transpose(input, 0, 1) #[time,state]
            
            if self.train_mode:
                response_1,_ = self.temporal_layer_1_lifted(input) 
                response_2,_ = self.temporal_layer_2_lifted(input)
                response_3,_ = self.temporal_layer_3_lifted(input)
            
            else : # update current state
                
                if self.state_1 == None:
                    response_1,self.state_1 = self.temporal_layer_1_lifted(input)
                    response_2,self.state_2 = self.temporal_layer_2_lifted(input)
                    response_3,self.state_3 = self.temporal_layer_3_lifted(input)
                else :
                    response_1,self.state_1 = self.temporal_layer_1(input,self.state_1)
                    response_2,self.state_2 = self.temporal_layer_2(input,self.state_2)
                    response_3,self.state_3 = self.temporal_layer_3(input,self.state_3)
                
            
            response_1 = torch.transpose(response_1,0,1)
            response_2 = torch.transpose(response_2,0,1)
            response_3 = torch.transpose(response_3,0,1)
            
            output = torch.stack([response_1, response_2, response_3], dim=0)
            output = self.conv1(output)
            output = torch.transpose(output, 1, 2)
            output = self.linear(output)
            output = torch.transpose(output, 1, 2)
            outputs += [output.squeeze(0)]
        
        if inputs.shape[0] == 1:
            return outputs[0]
        else:
            return torch.stack(outputs, dim=0) # return the batch

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

# Create the dataset
dataset = CustomDataset(features, labels)

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=15, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

def loss_fn(predicted_optimal_inputs, computed_optimal_inputs):
    # expects a batch
    cost = 0
    batch_len = predicted_optimal_inputs.shape[0]
    for jj in  range(batch_len):
        cost += torch.sum((predicted_optimal_inputs[jj] -  computed_optimal_inputs[jj])**2)
    return cost / batch_len / n_time_points # control input error at each time instant


### ------------------------------------------------------------------------------------------------------------ ###
## Train and test network

network = Network(train_mode=True)
criterion = loss_fn
optimizer = torch.optim.Adam(network.parameters(), lr=0.002)

# Training loop with early stopping
num_epochs = 1500
patience = 15
best_loss = float('inf')
epochs_no_improve = 0

for epoch in range(num_epochs):
    network.train()
    train_loss = 0
    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = network(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss}')
    
    # Early stopping
    if train_loss < best_loss:
        best_loss = train_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

# Evaluation on the test set
network.eval()
with torch.no_grad():
    test_loss = 0
    for batch_features, batch_labels in test_loader:
        outputs = network(batch_features)
        # loss = criterion(outputs, batch_labels)
        loss = criterion(outputs, batch_labels.reshape(batch_labels.shape[1], batch_labels.shape[2]))
        test_loss += loss.item()
    
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss}')

torch.save(network.state_dict(), 'model.pth')