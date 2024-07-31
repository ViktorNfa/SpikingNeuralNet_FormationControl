#=====================================
#        Python FC simulator
#          mobile 2D robots
#     Victor Nan Fernandez-Ayala 
#           (vnfa@kth.se)
#=====================================

import numpy as np
from scipy.optimize import minimize, LinearConstraint
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation
from tqdm import tqdm
import pickle as pkl
import os

#plt.style.use("seaborn-whitegrid")


### ------------------------------------------------------------------------------------------------------------ ###
## Auxiliary functions

def formationController(L_G, p, p_d):
    # Create extended laplacian
    I = np.identity(2)
    L_ext = np.kron(L_G,I)

    # Compute formation controller
    u = -np.dot(L_ext,p-p_d)
    return u

def cbf_h(p_i, p_j, safe_distance, dir):
    # Dir 1 corresponds to CM and -1 to OA
    return dir*(safe_distance**2 - np.linalg.norm(p_i - p_j)**2)

def cbf_gradh(p_i, p_j, dir):
    # Dir 1 corresponds to CM and -1 to OA
    return dir*(-2*np.array([[p_i[0]-p_j[0]], [p_i[1]-p_j[1]]]))

def cbfController(p, u_n, oa, d_oa, number_robots, edges, n, alpha):
    #Create CBF constraint matrices
    A_oa = np.zeros((len(edges), number_robots*n))
    b_oa = np.zeros((len(edges)))
    for i in range(len(edges)):
        aux_i = edges[i][0]-1
        aux_j = edges[i][1]-1
        p_i = np.array([p[2*aux_i],p[2*aux_i+1]])
        p_j = np.array([p[2*aux_j],p[2*aux_j+1]])

        b_oa[i] = alpha*cbf_h(p_i, p_j, d_oa, -1)

        grad_h_value_oa = np.transpose(cbf_gradh(p_i, p_j, -1))

        A_oa[i, 2*aux_i:2*aux_i+2] = grad_h_value_oa
        A_oa[i, 2*aux_j:2*aux_j+2] = -grad_h_value_oa

    #----------------------------
    # Solve minimization problem
    #----------------------------
    #Define linear constraints
    constraint_oa = LinearConstraint(A_oa*oa, lb=-b_oa*oa, ub=np.inf)
    
    #Define objective function
    def objective_function(u, u_n):
        return np.linalg.norm(u - u_n)**2
    
    #Construct the problem
    u = minimize(
        objective_function,
        x0=u_n,
        args=(u_n,),
        constraints=[constraint_oa],
    )

    return u.x, b_oa

def systemDynamics(p, u):
    # System dynamics parameters
    f = np.zeros(len(p))
    g = np.identity(len(u))

    # Update state vector derivative
    xdot = f+np.dot(g,u)
    return xdot


### ------------------------------------------------------------------------------------------------------------ ###
## Parameter setup

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


### ------------------------------------------------------------------------------------------------------------ ###
## Pre-calculations needed for controller and simulation

# Time size
max_time_size = max_T*freq

## For neighbor list 1
# Get the number of neighbours for each robot
number_neighbours = []
# Create edge list
edges = []
# Create Laplacian matrix for the graph
L_G = np.zeros((number_robots,number_robots))
# Setup controller output init output
controller = np.zeros((number_robots*dim,max_time_size-1))
nom_controller = np.zeros((number_robots*dim,max_time_size-1))
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
# Setup cbf function init output
cbf_oa = np.zeros((len(edges_fc),max_time_size-1))
for i in range(len(edges_fc)):
    edges_col.append("Edge"+str(edges_fc[i]))

# Modify ideal formation positions to one column vector
x_d1 = np.reshape(formation_positions1,number_robots*dim)
x_d2 = np.reshape(formation_positions2,number_robots*dim)


### ------------------------------------------------------------------------------------------------------------ ###
## Simulation and visualization loop

max_time_size = max_T*freq

# Initialize position matrix
x = np.zeros((number_robots*dim,max_time_size))

# Randomize initial position within a random circle
radius = x_max  # Assuming x_max == y_max
for i in range(number_robots):
    r = radius * np.sqrt(np.random.rand())  # Random radius
    theta = 2 * np.pi * np.random.rand()  # Random angle
    x[i, 0] = r * np.cos(theta)
    x[i, 1] = r * np.sin(theta)

# Start simulation loop
print("Computing evolution of the system...")
for t in tqdm(range(max_time_size-1)):
    secs = t/freq
    
    # Compute nominal controller - Time Varying
    if secs < 10:
        u_n = formationController(L_G, x[:,t], x_d1)
    elif secs >= 10 and secs < 20:
        u_n = formationController(L_G2, x[:,t], x_d2)
    else:
        u_n = formationController(L_G, x[:,t], x_d1)

    u, b_oa = cbfController(x[:,t], u_n, oa, d_oa, number_robots, edges_fc, dim, alpha)

    # Update the system using dynamics
    xdot = systemDynamics(x[:,t], u)
    x[:,t+1] = xdot*(1/freq) + x[:,t]

    # Save CBF functions
    for e in range(len(edges_fc)):
        aux_i = edges_fc[e][0]-1
        aux_j = edges_fc[e][1]-1
        x_i = np.array([x[2*aux_i,t],x[2*aux_i+1,t]])
        x_j = np.array([x[2*aux_j,t],x[2*aux_j+1,t]])
        cbf_oa[e,t] = cbf_h(x_i, x_j, d_oa, -1)
    
    # Save Final controller
    controller[:,t] = u

    # Save Nominal controller
    nom_controller[:,t] = u_n


### ------------------------------------------------------------------------------------------------------------ ###
## Visualize CBF conditions/plots & trajectories

print("Showing CBF function evolution...")

# Plot the CBF obstacle avoidance
fig_cbf_oa, ax_cbf_oa = plt.subplots()  # Create a figure and an axes.
for i in range(len(edges_fc)):
    ax_cbf_oa.plot(1/freq*np.arange(max_time_size-1), cbf_oa[i,:], label=edges_col[i])  # Plot some data on the axes.
ax_cbf_oa.set_xlabel('time')  # Add an x-label to the axes.
ax_cbf_oa.set_ylabel('h_ca')  # Add a y-label to the axes.
ax_cbf_oa.set_title("CBF functions for collision avoidance")  # Add a title to the axes.
ax_cbf_oa.legend(fontsize=13)  # Add a legend.
ax_cbf_oa.axhline(y=0, color='k', lw=1)

# Plot the normed difference between nominal and final controller
fig_norm, ax_norm = plt.subplots()  # Create a figure and an axes.
ax_norm.axis('on')
for i in range(number_robots):
    diff_x = controller[2*i,:] - nom_controller[2*i,:]
    diff_y = controller[2*i+1,:] - nom_controller[2*i+1,:]
    diff = np.array([diff_x, diff_y])
    normed_difference = np.sqrt(np.square(diff).sum(axis=0))
    ax_norm.plot(1/freq*np.arange(max_time_size-1), normed_difference, label="Robot"+str(i+1))  # Plot some data on the axes.

ax_norm.set_xlabel('time')  # Add an x-label to the axes.
ax_norm.set_ylabel('|u - u_nom|')  # Add a y-label to the axes.
ax_norm.set_title("Normed difference between u and nominal u")  # Add a title to the axes.
ax_norm.legend(fontsize=13)  # Add a legend.
ax_norm.axhline(y=0, color='k', lw=1)

# Plot the final controller values
fig_contr, ax_contr = plt.subplots()  # Create a figure and an axes.
ax_contr.axis('on')
for i in range(number_robots):
    ax_contr.plot(1/freq*np.arange(max_time_size-1), controller[2*i,:], label="RobotX"+str(i+1))  # Plot some data on the axes.
    ax_contr.plot(1/freq*np.arange(max_time_size-1), controller[2*i+1,:], label="RobotY"+str(i+1))  # Plot some data on the axes.
ax_contr.set_xlabel('time')  # Add an x-label to the axes.
ax_contr.set_ylabel('u')  # Add a y-label to the axes.
ax_contr.set_title("Final controller u")  # Add a title to the axes.
ax_contr.legend(fontsize=13)  # Add a legend.
ax_contr.axhline(y=0, color='k', lw=1)

plt.show()


print("Showing animation...")

# Start figure and axes with limits
fig = plt.figure()
ax = plt.axes(xlim=(-winx, winx), ylim=(-winy, winy))

time_txt = ax.text(0.475, 0.975,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)

# Add initial points
initials = []
for i in range(number_robots):
    initials.append(plt.Circle((x[2*i,0], x[2*i+1,0]), r_robot/2, fc='k', alpha=0.3))
    plt.gca().add_patch(initials[i])
for i in range(len(edges)):
    aux_i = edges[i][0]-1
    aux_j = edges[i][1]-1
    initials.append(plt.Line2D((x[2*aux_i,0], x[2*aux_j,0]), (x[2*aux_i+1,0], x[2*aux_j+1,0]), lw=0.5, color='k', alpha=0.1))
    plt.gca().add_line(initials[number_robots+i])

shapes = []
for i in range(number_robots):
    shapes.append(plt.Circle((x[2*i,0], x[2*i+1,0]), r_robot, fc='b'))

for i in range(len(edges)):
    aux_i = edges[i][0]-1
    aux_j = edges[i][1]-1
    shapes.append(plt.Line2D((x[2*aux_i,0], x[2*aux_j,0]), (x[2*aux_i+1,0], x[2*aux_j+1,0]), lw=0.5, color='b', alpha=0.3))
for i in range(len(edges2)):
    aux_i = edges2[i][0]-1
    aux_j = edges2[i][1]-1
    shapes.append(plt.Line2D((x[2*aux_i,0], x[2*aux_j,0]), (x[2*aux_i+1,0], x[2*aux_j+1,0]), lw=0.5, color='b', alpha=0.3))

def init():
    for i in range(number_robots):
        shapes[i].center = (x[2*i,0], x[2*i+1,0])
        ax.add_patch(shapes[i])

    for i in range(len(edges)):
        aux_i = edges[i][0]-1
        aux_j = edges[i][1]-1
        shapes[number_robots+i].set_xdata((x[2*aux_i,0], x[2*aux_j,0]))
        shapes[number_robots+i].set_ydata((x[2*aux_i+1,0], x[2*aux_j+1,0]))
        ax.add_line(shapes[number_robots+i])
    for i in range(len(edges2)):
        aux_i = edges2[i][0]-1
        aux_j = edges2[i][1]-1
        shapes[number_robots+len(edges)+i].set_xdata((0, 0))
        shapes[number_robots+len(edges)+i].set_ydata((0, 0))
        ax.add_line(shapes[number_robots+len(edges)+i])

    time_txt.set_text('T=0.0 s')

    return shapes + [time_txt,]

def animate(frame):

    secs = frame/freq

    for i in range(number_robots):
        shapes[i].center = (x[2*i,frame], x[2*i+1,frame])

    if secs < 10:
        for i in range(len(edges)):
            aux_i = edges[i][0]-1
            aux_j = edges[i][1]-1
            shapes[number_robots+i].set_xdata((x[2*aux_i,frame], x[2*aux_j,frame]))
            shapes[number_robots+i].set_ydata((x[2*aux_i+1,frame], x[2*aux_j+1,frame]))
        for i in range(len(edges2)):
            aux_i = edges2[i][0]-1
            aux_j = edges2[i][1]-1
            shapes[number_robots+len(edges)+i].set_xdata((0, 0))
            shapes[number_robots+len(edges)+i].set_ydata((0, 0))
    elif secs >= 10 and secs < 20:
        for i in range(len(edges)):
            aux_i = edges[i][0]-1
            aux_j = edges[i][1]-1
            shapes[number_robots+i].set_xdata((0, 0))
            shapes[number_robots+i].set_ydata((0, 0))
        for i in range(len(edges2)):
            aux_i = edges2[i][0]-1
            aux_j = edges2[i][1]-1
            shapes[number_robots+len(edges)+i].set_xdata((x[2*aux_i,frame], x[2*aux_j,frame]))
            shapes[number_robots+len(edges)+i].set_ydata((x[2*aux_i+1,frame], x[2*aux_j+1,frame]))
    else:
        for i in range(len(edges)):
            aux_i = edges[i][0]-1
            aux_j = edges[i][1]-1
            shapes[number_robots+i].set_xdata((x[2*aux_i,frame], x[2*aux_j,frame]))
            shapes[number_robots+i].set_ydata((x[2*aux_i+1,frame], x[2*aux_j+1,frame]))
        for i in range(len(edges2)):
            aux_i = edges2[i][0]-1
            aux_j = edges2[i][1]-1
            shapes[number_robots+len(edges)+i].set_xdata((0, 0))
            shapes[number_robots+len(edges)+i].set_ydata((0, 0))

    time_txt.set_text('T=%.1d s' % secs)

    return shapes + [time_txt,]

anim = animation.FuncAnimation(fig, animate, 
                               init_func=init, 
                               frames=max_time_size, 
                               interval=1/freq*1000,
                               blit=True,
                               repeat=False)

plt.show()

#anim.save('animation.mp4', fps=50, extra_args=['-vcodec', 'libx264'])

print("Completed!")


### ------------------------------------------------------------------------------------------------------------ ###
## Visualize CBF conditions/plots & trajectories

print("Saving dataset...")

# directory of the dataset 
dataset_dir = "dataset"
os.makedirs(dataset_dir,exist_ok=True)

with open(dataset_dir + "/dataset.pkl", 'wb') as f:
    pkl.dump(dataset, f)