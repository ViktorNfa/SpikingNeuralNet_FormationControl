import formation_control as fc
import numpy as np


### ------------------------------------------------------------------------------------------------------------ ###
## Testing scenario

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
x, u, cbf_oa, nom_controller = fc.execute_fc_simulation(number_robots, dim, max_time_size, x_max, freq, L_G, L_G2, x_d1, x_d2, oa, d_oa, edges_fc, alpha)

# Print plots
fc.print_plots(edges_fc, max_time_size, freq, cbf_oa, edges_col, u, nom_controller, number_robots)

# Play video
fc.play_video(winx, winy, r_robot, number_robots, edges, edges2, x, max_time_size, freq, save=False)