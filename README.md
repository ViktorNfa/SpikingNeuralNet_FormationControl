# SpikingNeuralNet_FormationControl
This repository holds the code for a Spiking Neural Network (SNN) using Norse to learn a time-varying Formation Controller (FC) for Multi-Agent Systems (MAS).

## Dependencies:
To install all the required packages you can create a virtual environment (venv, conda...) and execute: 
```
pip install -r requirements.txt
```

## To Do

- [x] Setup repository, requirements, README and main structure.
- [x] Program classical formation controller using my master thesis code. Save data to train SNN.
- [x] Create a SNN for formation control and train and test accordingly.
- [ ] Create simulator using Gazebo, ROS2 and Rosie robotic models.
- [ ] Simulate both classical and SNN FC. Record and upload video to my YouTube and write a short 1 page report.

## Summary

### documentation.pdf
This 1 page document summarizes the idea and reports the results of the SNN FC.

### formation_control.py
This file contains the code to create and test a time-varying formation controller using classic control and multiple agents in a MAS.

### snn_training.py
This file contains the code to train and test the SNN, as well as saving the trained model to use for simulation.

### simulation.py
This file contains the code to simulate both the original formation controller using classic control and the newer formation controller using the SNN.