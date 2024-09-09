# Spiking Neural Network for Formation Control in Multi-Agent Systems

This repository contains code to implement a Spiking Neural Network (SNN) for solving a formation control problem in Multi-Agent Systems (MAS) using Norse. The SNN is trained to approximate a classical time-varying formation control algorithm while ensuring collision avoidance.

## What are Spiking Neural Networks (SNNs)?

Spiking Neural Networks (SNNs) are a type of artificial neural network that more closely mimic biological neurons compared to traditional neural networks. In SNNs, neurons communicate through discrete spikes or pulses, much like how biological neurons do in the brain. This temporally-rich communication method introduces a notion of time and allows SNNs to process temporal sequences more efficiently.

In this project, we utilize SNNs to learn and replicate classical formation control techniques, providing a neuro-inspired approach to solving control problems.

## What is Formation Control?

Formation control refers to the coordination strategy used in multi-agent systems (such as drones, robots, or autonomous vehicles) to maintain a desired formation during motion. The objective is to move multiple agents while ensuring they maintain certain relative positions to each other. This type of control is commonly used in applications such as robotics, autonomous vehicles, and swarm intelligence.

The goal of this project is to compare a traditional formation control algorithm with a learned formation controller using an SNN.

## What are Control Barrier Functions (CBFs)?

Control Barrier Functions (CBFs) are mathematical tools used to enforce safety constraints, such as collision avoidance in control systems. They ensure that agents in a multi-agent system avoid collisions with each other or obstacles by restricting certain unsafe control actions. In this project, CBFs are used alongside both the classical and SNN-based controllers to ensure collision-free movement of the agents.

## Project Overview

The project aims to compare the classical formation control algorithm with a learned SNN-based approach. The task involves agents maintaining a desired formation while avoiding collisions. This project demonstrates the potential of neuro-inspired computation for solving engineering problems, such as multi-agent formation control.

### Key Features

- **Classical Formation Control**: a traditional method for controlling the position and trajectory of agents in a multi-agent system.
- **Spiking Neural Network (SNN) Formation Control**: a neural-inspired algorithm trained to mimic the classical control while introducing biological plausibility through spiking neurons.
- **Collision Avoidance**: both the classical and learned approaches incorporate collision avoidance strategies using Control Barrier Functions (CBFs).

## To Do

- [x] Setup repository, requirements, README and main structure.
- [x] Program classical formation controller using my master thesis code. Save data to train SNN.
- [x] Create a SNN for formation control and train and test accordingly.
- [X] Simulate both classical and SNN FC. Record video and write a short report.

## Installation

### 1. Clone the Repository

To start using the project, clone the repository from GitHub:

```
git clone https://github.com/ViktorNfa/SpikingNeuralNet_FormationControl.git cd SpikingNeuralNet_FormationControl
```

### 2. Set Up a Virtual Environment

It is recommended to use a virtual environment to manage dependencies:

Using `venv`:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Or using `conda`:

```
conda create --name snn_env python=3.9
conda activate snn_env
```

### 3. Install Dependencies

Once inside the virtual environment, install the required dependencies:

```
pip install -r requirements.txt
```

## Results

### Video Demonstrations

Below are visual demonstrations of the formation control in action. These videos show the behavior of the multi-agent system using both the classical and SNN-based controllers:

- Classical Formation Control:

![Original algorithm animation](original_fc_control.gif)

- SNN-based Formation Control:

![Learnt algorithm animation](learnt_fc_control.gif)

### Collision Avoidance Comparison

Below are visual representations of the effectiveness of collision avoidance strategies in both the classical and SNN-based control systems:

- Classical CBF Collision Avoidance:

![Original algorithm CBF](original_cbf_collision_avoidance.png)

- Learnt SNN CBF Collision Avoidance:

![Leartn algorithm CBF](learnt_cbf_collision_avoidance.png)

The graphs show the effectiveness of Control Barrier Functions in maintaining safe distances between agents. Notably the learnt agent can keep the safe condition with respect to the rest of the network.

## Usage Instructions

- Classical Formation Control: run the classical formation control simulation with the following command:

```
python formation_control.py
```

- Dataset generator: run the dataset generator needed to train the network a:

```
python dataset_generator.py
```

- SNN Training: to train the Spiking Neural Network, execute:

```
python snn_training.py
```

- Simulation: once the SNN is trained, you can compare both controllers by running the simulation:

```
python simulation.py
```

## Future Work

- Extend the SNN to handle larger teams of agents.
- Investigate more complex formation control tasks.
- Explore different SNN architectures to improve learning efficiency.
  
## License

This project is licensed under the [LICENSE](LICENSE).

## Acknowledgments

This project was inspired by research into neuro-inspired computation and multi-agent systems, particularly those solving engineering problems such as formation control and collision avoidance.

