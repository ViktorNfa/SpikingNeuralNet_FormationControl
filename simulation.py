import ros2_ws.src.formation_control as fc
import numpy as np
import pickle as pkl
import os
import norse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split


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
            # output = torch.transpose(output, 1, 2)
            output = self.linear(output)
            output = torch.transpose(output, 1, 2)
            outputs += [output.squeeze(0)]
        
        if inputs.shape[0] == 1:
            return outputs[0]
        else:
            return torch.stack(outputs, dim=0) # return the batch


### ------------------------------------------------------------------------------------------------------------ ###
## Testing scenario

# Create an instance of the Network class
network = Network(train_mode=False)
# load the model from file
network.load_state_dict(torch.load('model.pth'))
# Select snn agent
snn_agent = 0


# Dimensionality of the problem
dim = 2

# Window size
winx = 20
winy = 20

# Arena size
x_max = winx
y_max = winy

# Robot size/radius (modelled as a circle with a directional arrow)
r_robot = 0.5

# Frequency of update of the simulation (in Hz)
freq = 50

# Maximum time of the simulation (in seconds)
max_T = 30

# Ideal formation positions
formation_positions1 = [[0, 3], [0, 0], [0, -3], [3, 3], [3, -3]]
formation_positions2 = [[10, 0], [-6.5, -10], [6.5, -10], [-10, 0], [0, 10]]

# Get the number of robots
number_robots = len(formation_positions1)

# List of neighbours for each robot
neighbours = [[2], [1, 3, 4, 5], [2], [2], [2]]
neighbours2 = [[3, 5], [3, 4], [1, 2], [2, 5], [1, 4]]
# neighbours = [[2, 4], [1, 3, 4, 5], [2, 5], [1, 2], [2, 3]]
neighbours_fc = [[i+1 for i in range(number_robots) if i != j] for j in range(number_robots)]

# CBF obstacle avoidance activation 
# (1 is activated/0 is deactivated)
oa = 1

# Safe distance for obstacle avoidance
d_oa = 1.5

# Linear alpha function with parameter
alpha = 1

# Time size
max_time_size = max_T*freq

## For neighbor list 1
# Get the number of neighbours for each robot
number_neighbours = []
# Create edge list
edges = []
# Create Laplacian matrix for the graph
L_G = np.zeros((number_robots,number_robots))
for i in range(number_robots):
    number_neighbours.append(len(neighbours[i]))
    L_G[i, i] = number_neighbours[i]
    for j in neighbours[i]:
        L_G[i, j-1] = -1
        if (i+1,j) not in edges and (j,i+1) not in edges:
            edges.append((i+1,j))

## For neighbor list 2
# Get the number of neighbours for each robot
number_neighbours2 = []
# Create edge list
edges2 = []
# Create Laplacian matrix for the graph
L_G2 = np.zeros((number_robots,number_robots))
for i in range(number_robots):
    number_neighbours2.append(len(neighbours2[i]))
    L_G2[i, i] = number_neighbours2[i]
    for j in neighbours2[i]:
        L_G2[i, j-1] = -1
        if (i+1,j) not in edges2 and (j,i+1) not in edges2:
            edges2.append((i+1,j))

# Edges for a fully connected network
edges_fc = []
for i in range(number_robots):
    for j in neighbours_fc[i]:
        if (i+1,j) not in edges_fc and (j,i+1) not in edges_fc:
            edges_fc.append((i+1,j))

# Create edge list for the name of columns
edges_col = []
for i in range(len(edges_fc)):
    edges_col.append("Edge"+str(edges_fc[i]))

# Modify ideal formation positions to one column vector
x_d1 = np.reshape(formation_positions1,number_robots*dim)
x_d2 = np.reshape(formation_positions2,number_robots*dim)

# Execute the simulation
x, u, cbf_oa, nom_controller = fc.execute_snnfc_simulation(number_robots, dim, max_time_size, x_max, freq, L_G, L_G2, x_d1, x_d2, oa, d_oa, edges_fc, alpha, network, snn_agent, True)

# Print plots
fc.print_plots(edges_fc, max_time_size, freq, cbf_oa, edges_col, u, nom_controller, number_robots)

# Play video
fc.play_video(winx, winy, r_robot, number_robots, edges, edges2, x, max_time_size, freq, save=False)

# Play video
fc.play_video(winx, winy, r_robot, number_robots, edges, edges2, x, max_time_size, freq, save=True)


### ------------------------------------------------------------------------------------------------------------ ###
## Save data

dataset_per_agent = {}
for i in range(number_robots):
    dataset_per_agent[i] = {"state":[x[2*i,:], x[2*i+1,:]],"control":[u[2*i,:], u[2*i+1,:]]}

dataset = {}
counter = 0
# saving the numpy arrays in folder 
for data in dataset_per_agent.values():
    dataset[counter] = data
    counter += 1

# directory of the dataset 
dataset_dir = "dataset"
os.makedirs(dataset_dir,exist_ok=True)

with open(dataset_dir + "/simulation.pkl", 'wb') as f:
    pkl.dump(dataset, f)