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
import torch

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
## Simulation and visualization loop

def execute_fc_simulation(number_robots, dim, max_time_size, x_max, freq, L_G, L_G2, x_d1, x_d2, oa, d_oa, edges_fc, alpha):
    
    # Setup cbf function init output
    cbf_oa = np.zeros((len(edges_fc),max_time_size-1))
    # Setup controller output init output
    controller = np.zeros((number_robots*dim,max_time_size-1))
    nom_controller = np.zeros((number_robots*dim,max_time_size-1))

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
    for t in range(max_time_size-1):
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

    return x, controller, cbf_oa, nom_controller

def execute_snnfc_simulation(number_robots, dim, max_time_size, x_max, freq, L_G, L_G2, x_d1, x_d2, oa, d_oa, edges_fc, alpha, network, snn_agent):
    
    # Setup cbf function init output
    cbf_oa = np.zeros((len(edges_fc),max_time_size-1))
    # Setup controller output init output
    controller = np.zeros((number_robots*dim,max_time_size-1))
    nom_controller = np.zeros((number_robots*dim,max_time_size-1))

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
    for t in range(max_time_size-1):
        secs = t/freq
        
        # Compute nominal controller - Time Varying
        if secs < 10:
            u_n = formationController(L_G, x[:,t], x_d1)
        elif secs >= 10 and secs < 20:
            u_n = formationController(L_G2, x[:,t], x_d2)
        else:
            u_n = formationController(L_G, x[:,t], x_d1)

        u, b_oa = cbfController(x[:,t], u_n, oa, d_oa, number_robots, edges_fc, dim, alpha)

        # Utilize SNN for snn_agent
        input_data = []
        for j in range(number_robots):
            if snn_agent != j:
                input_data += [x[2*snn_agent,t] - x[2*j,t], x[2*snn_agent+1,t] - x[2*j+1,t],]
            
        if secs < 10:
            input_data += [x_d1[2*i]-x[2*snn_agent,t], x_d1[2*i+1]-x[2*snn_agent+1,t]]
        elif secs >= 10 and secs < 20:
            input_data += [x_d2[2*i]-x[2*snn_agent,t], x_d2[2*i+1]-x[2*snn_agent+1,t]]
        else:
            input_data += [x_d1[2*i]-x[2*snn_agent,t], x_d1[2*i+1]-x[2*snn_agent+1,t]]
        
        input = torch.tensor(input_data).float()
        input = input.unsqueeze(0)  # Adding a dimension for batch size

        u[2*snn_agent], u[2*snn_agent+1] = network(input).detach()

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

    return x, controller, cbf_oa, nom_controller


### ------------------------------------------------------------------------------------------------------------ ###
## Visualize CBF conditions/plots & trajectories

def print_plots(edges_fc, max_time_size, freq, cbf_oa, edges_col, controller, nom_controller, number_robots):

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

def play_video(winx, winy, r_robot, number_robots, edges, edges2, x, max_time_size, freq, save=False):

    if save == False:
        print("Showing animation...")
    else:
        print("Saving animation...")

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

    if save == False:
        plt.show()
    else:
        anim.save('animation.mp4', fps=50, extra_args=['-vcodec', 'libx264'])

    print("Completed!")