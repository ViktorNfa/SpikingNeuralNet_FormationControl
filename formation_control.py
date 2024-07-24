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

def extraRobotDynamics(i, max_time_size, v_huil, division):
    # Leave some time at the start and the end to allow the robots to form
    max_time = max_time_size*(1-2/division)
    i = i - max_time_size/division

    u_huil_x = 0
    u_huil_y = 0
    # Simulated HuIL input to move in  a rectangle manner
    if i < max_time/4 and i > 0:
        u_huil_x = 0
        u_huil_y = -v_huil
    elif i > max_time/4 and i < max_time/2:
        u_huil_x = v_huil
        u_huil_y = 0
    elif i > max_time/2 and i < 3*max_time/4:
        u_huil_x = 0
        u_huil_y = v_huil
    elif i > 3*max_time/4 and i <= max_time:
        u_huil_x = -v_huil
        u_huil_y = 0
    else:
        u_huil_x = 0
        u_huil_y = 0

    return np.array([u_huil_x, u_huil_y])

def cbf_h(p_i, p_j, safe_distance, dir):
    # Dir 1 corresponds to CM and -1 to OA
    return dir*(safe_distance**2 - np.linalg.norm(p_i - p_j)**2)

def cbf_gradh(p_i, p_j, dir):
    # Dir 1 corresponds to CM and -1 to OA
    return dir*(-2*np.array([[p_i[0]-p_j[0]], [p_i[1]-p_j[1]]]))

def cbfController(p, u_n, cm, oa, d_cm, d_oa, number_robots, edges, n, alpha):
    #Create CBF constraint matrices
    A_cm = np.zeros((len(edges), number_robots*n))
    b_cm = np.zeros((len(edges)))
    A_oa = np.zeros((len(edges), number_robots*n))
    b_oa = np.zeros((len(edges)))
    for i in range(len(edges)):
        aux_i = edges[i][0]-1
        aux_j = edges[i][1]-1
        p_i = np.array([p[2*aux_i],p[2*aux_i+1]])
        p_j = np.array([p[2*aux_j],p[2*aux_j+1]])

        b_cm[i] = alpha*cbf_h(p_i, p_j, d_cm, 1)
        b_oa[i] = alpha*cbf_h(p_i, p_j, d_oa, -1)

        grad_h_value_cm = np.transpose(cbf_gradh(p_i, p_j, 1))
        grad_h_value_oa = np.transpose(cbf_gradh(p_i, p_j, -1))

        A_cm[i, 2*aux_i:2*aux_i+2] = grad_h_value_cm
        A_cm[i, 2*aux_j:2*aux_j+2] = -grad_h_value_cm
        A_oa[i, 2*aux_i:2*aux_i+2] = grad_h_value_oa
        A_oa[i, 2*aux_j:2*aux_j+2] = -grad_h_value_oa

    #----------------------------
    # Solve minimization problem
    #----------------------------
    #Define linear constraints
    constraint_cm = LinearConstraint(A_cm*cm, lb=-b_cm*cm, ub=np.inf)
    constraint_oa = LinearConstraint(A_oa*oa, lb=-b_oa*oa, ub=np.inf)
    
    #Define objective function
    def objective_function(u, u_n):
        return np.linalg.norm(u - u_n)**2
    
    #Construct the problem
    u = minimize(
        objective_function,
        x0=u_n,
        args=(u_n,),
        constraints=[constraint_cm, constraint_oa],
    )

    return u.x, b_cm, b_oa

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
x_max = winx-5
y_max = winy-5

# Robot size/radius (modelled as a circle with a directional arrow)
r_robot = 0.5

# Frequency of update of the simulation (in Hz)
freq = 50

# Maximum time of the simulation (in seconds)
max_T = 30

# Ideal formation positions
formation_positions = [[0, 2], [0, 0], [0, -2], [2, 2], [2, -2]]
#formation_positions = [[0, 10], [0, 8], [0, 6], [0, 4], [0, 2], [0, 0], [0, -2], [0, -4], [0, -6], [0, -8], [0, -10], 
#                        [10, 10], [8, 8], [6, 6], [4, 4], [2, 2], [2, -2], [4, -4], [6, -6], [8, -8], [10, -10]]

# Get the number of robots
number_robots = len(formation_positions)

# List of neighbours for each robot
#neighbours = [[2], [1, 3, 4, 5], [2], [2], [2]]
#neighbours = [[2, 4], [1, 3, 4, 5], [2, 5], [1, 2], [2, 3]]
neighbours = [[2, 4], [1, 3, 4, 5], [2, 5], [1, 2, 5], [2, 3, 4]]
#neighbours = [[2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7, 16, 17], [6, 8], [7, 9], [8, 10], [9, 11], [10], 
#               [13], [12, 14], [13, 15], [14, 16], [6, 15], [6, 18], [17, 19], [18, 20], [19, 21], [20]]
#neighbours = [[i+1 for i in range(number_robots) if i != j] for j in range(number_robots)]

# CBF Communication maintenance or obstacle avoidance activation 
# (1 is activated/0 is deactivated)
cm = 1
oa = 0

# Safe distance for communication maintenance and obstacle avoidance
d_cm = 2
d_oa = 1.1

# Linear alpha function with parameter
alpha = 1


### ------------------------------------------------------------------------------------------------------------ ###
## Pre-calculations needed for controller and simulation

# Time size
max_time_size = max_T*freq

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

# Create edge list for the name of columns
edges_col = []
# Setup cbf functions init output
cbf_cm = np.zeros((len(edges),max_time_size-1))
cbf_oa = np.zeros((len(edges),max_time_size-1))
for i in range(len(edges)):
    edges_col.append("Edge"+str(edges[i]))

# Modify ideal formation positions to one column vector
x_d = np.reshape(formation_positions,number_robots*dim)


### ------------------------------------------------------------------------------------------------------------ ###
## Simulation and visualization loop

max_time_size = max_T*freq

# Initialize position matrix
x = np.zeros((number_robots*dim,max_time_size))

# Randomize initial position
x[:,0] = [-(x_max-5), (y_max-5), 0, 0, -(x_max-5), -(y_max-5), (x_max-5), (y_max-5), (x_max-5), -(y_max-5)]

# Start simulation loop
print("Computing evolution of the system...")
for t in tqdm(range(max_time_size-1)):
    secs = t/freq
    
    # Compute nominal controller - Centralized and Distributed
    u_n = formationController(L_G, x[:,t], x_d)

    u, b_cm, b_oa = cbfController(x[:,t], u_n, cm, oa, d_cm, d_oa, number_robots, edges, dim, alpha)

    # Update the system using dynamics
    xdot = systemDynamics(x[:,t], u)
    x[:,t+1] = xdot*(1/freq) + x[:,t]

    # Save CBF functions
    for e in range(len(edges)):
        aux_i = edges[e][0]-1
        aux_j = edges[e][1]-1
        x_i = np.array([x[2*aux_i,t],x[2*aux_i+1,t]])
        x_j = np.array([x[2*aux_j,t],x[2*aux_j+1,t]])
        cbf_cm[e,t] = cbf_h(x_i, x_j, d_cm, 1)
        cbf_oa[e,t] = cbf_h(x_i, x_j, d_oa, -1)
    
    # Save Final controller
    controller[:,t] = u

    # Save Nominal controller
    nom_controller[:,t] = u_n


### ------------------------------------------------------------------------------------------------------------ ###
## Visualize CBF conditions/plots & trajectories

print("Showing CBF function evolution...")

# Plot the CBF comunication maintenance
fig_cbf_cm, ax_cbf_cm = plt.subplots()  # Create a figure and an axes.
for i in range(len(edges)):
    ax_cbf_cm.plot(1/freq*np.arange(max_time_size-1), cbf_cm[i,:], label=edges_col[i])  # Plot some data on the axes.
ax_cbf_cm.set_xlabel('time')  # Add an x-label to the axes.
ax_cbf_cm.set_ylabel('h_cm')  # Add a y-label to the axes.
ax_cbf_cm.set_title("CBF functions for connectivity maintenance")  # Add a title to the axes.
ax_cbf_cm.legend(fontsize=13)  # Add a legend.
ax_cbf_cm.axhline(y=0, color='k', lw=1)

# Plot the CBF obstacle avoidance
fig_cbf_oa, ax_cbf_oa = plt.subplots()  # Create a figure and an axes.
for i in range(len(edges)):
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

    time_txt.set_text('T=0.0 s')

    return shapes + [time_txt,]

def animate(frame):

    for i in range(number_robots):
        shapes[i].center = (x[2*i,frame], x[2*i+1,frame])

    for i in range(len(edges)):
        aux_i = edges[i][0]-1
        aux_j = edges[i][1]-1
        shapes[number_robots+i].set_xdata((x[2*aux_i,frame], x[2*aux_j,frame]))
        shapes[number_robots+i].set_ydata((x[2*aux_i+1,frame], x[2*aux_j+1,frame]))

    secs = frame/freq
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