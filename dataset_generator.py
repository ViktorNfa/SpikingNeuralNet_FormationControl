import formation_control as fc
import numpy as np
from tqdm import tqdm
import pickle as pkl
import os


### ------------------------------------------------------------------------------------------------------------ ###
## Auxiliary functions

def relative_position_other_agents(i, number_robots, x, max_time_size, freq, x_d1, x_d2):
    """computes the relative position of all the other agents with respect to the agent i"""
    data = []
    for j in range(number_robots):
        if i != j:
            data_agent = []
            for t in range(max_time_size):
                data_agent.append([x[2*i,t]-x[2*j,t],x[2*i+1,t]-x[2*j+1,t]])
            data.append(data_agent)

    return data

def relative_position_from_goal(i, number_robots, x, max_time_size, freq, x_d1, x_d2):
    """Computes the relative position of the agent i with respect to its goal position"""
    data = []
    for t in range(max_time_size):
        secs = t/freq
        if secs < 10:
            data.append([x_d1[2*i]-x[2*i,t], x_d1[2*i+1]-x[2*i+1,t]])
        elif secs >= 10 and secs < 20:
            data.append([x_d2[2*i]-x[2*i,t], x_d2[2*i+1]-x[2*i+1,t]])
        else:
            data.append([x_d1[2*i]-x[2*i,t], x_d1[2*i+1]-x[2*i+1,t]])

    return [data]   

def prepare_dataset(number_robots, x, u, max_time_size, freq, x_d1, x_d2):
    combo = [relative_position_other_agents, relative_position_from_goal]
    
    data_set_per_agent = {}
    for i in range(number_robots):
        feature = []
        label   = []
        for function in combo:
            feature += function(i, number_robots, x, max_time_size, freq, x_d1, x_d2)
        
        for t in range(max_time_size):
            label.append([u[2*i,t],u[2*i+1,t]])
        
        feature = np.hstack([np.array(component) for component in feature]).T # time in the x direction and the state on the y
        label   = np.array(label).T
        
        data_set_per_agent[i] = {"feature":feature,"label":label}
    
    return data_set_per_agent


### ------------------------------------------------------------------------------------------------------------ ###
## Generate the dataset

# Number of simulations
num_sims  = 1


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

dataset = {}
counter = 0
for i in tqdm(range(num_sims)):
    x, u, _, _ = fc.execute_fc_simulation(number_robots, dim, max_time_size, x_max, freq, L_G, L_G2, x_d1, x_d2, oa, d_oa, edges_fc, alpha)

    shortcut = 10*freq
    dataset_per_agent = prepare_dataset(number_robots, x[:,:shortcut], u[:,:shortcut], shortcut-1, freq, x_d1, x_d2)
    
    # saving the numpy arrays in folder 
    for data in dataset_per_agent.values():
        dataset[counter] = data
        counter += 1

# directory of the dataset 
dataset_dir = "dataset"
os.makedirs(dataset_dir,exist_ok=True)

with open(dataset_dir + "/dataset.pkl", 'wb') as f:
    pkl.dump(dataset, f)