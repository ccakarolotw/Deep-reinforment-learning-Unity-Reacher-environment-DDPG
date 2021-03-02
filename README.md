# Deep-reinforment-learning-Unity-Reacher-environment-DDPG
This project uses DDPG implemented in Pytorch to solve the Unity Reacher environment. The environment is solved in 238 episodes. 

## Unity Reacher Environment
The agent controls a double-jointed arm to move to goal locations and keep it there. The agent recieve +0.1 reward for every time stamp the arm is in the goal location. The observation space has 33 variables and the action space has 4 continuous variables. 

## Dependencies
Installation of dependencies follow https://github.com/udacity/deep-reinforcement-learning#dependencies
1. Create (and activate) a new environment with Python 3.6.

`conda create --name drlnd python=3.6 
activate drlnd`

2.  Install pytorch 0.4.0

`conda install pytorch=0.4.0 -c pytorch`

3. Download Unity environment and place it in the same folder as the jupyter notebook `Navigation.ipynb`

## Deep Deterministic Policy Gradient (DDPG) Agent

The agent consists of a critic network and an actor network. The critic network learns to approximate the state-action value function (Q). The actor network learns to choose actions based on input state to maximize the expected value given by the critic network. The critic network is updated through TD learning. 

The critic and target critic network are represented by neural network with three connected layers.
```
x_state = nn.Linear(state_dim,hidden_dim_state)(state)
x_state = F.leaky_relu(x_state)
x = torch.cat(x_state, action)
x = nn.Linear(hidden_dim_state, hidden_dim)(x)
x = F.relu(x)
x = nn.Linear(hidden_dim, action_dim)(x)
```
