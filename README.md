# Deep-reinforment-learning-Unity-Reacher-environment-DDPG
This project uses DDPG implemented in Pytorch to solve the Unity Reacher environment. The environment is considered solved if the score (average over 100 episodes and all the parallel arms) is >30. The code is in PyTorch (v0.4) and Python 3. 

## Unity Reacher Environment
The agent controls a double-jointed arm to move to goal locations and keep it there. The agent recieve +0.1 reward for every time stamp the arm is in the goal location. The observation space has 33 variables and the action space has 4 continuous variables. 

## Dependencies
Installation of dependencies follow https://github.com/udacity/deep-reinforcement-learning#dependencies
1. Create (and activate) a new environment with Python 3.6.

`conda create --name drlnd python=3.6 
activate drlnd`

2.  Install pytorch 0.4.0

`conda install pytorch=0.4.0 -c pytorch`

3. Download Unity environment and place it in the same folder as the jupyter notebook `Continuous_Control.ipynb`

## Deep Deterministic Policy Gradient (DDPG) Agent

The agent consists of a critic network and an actor network. The critic network learns to approximate the state-action value function (Q). The actor network learns to choose actions based on input state to maximize the expected value given by the critic network. 

The critic network is updated through TD learning. 

Q(state, action) = reward + GAMMA* Q'(next_state,next_action)

Q' is the target Q network. 
The critic and target critic network is represented by a neural network with three fully connected layers.
```x_state = nn.Linear(state_dim,hidden_dim)(state)
x_state = F.leaky_relu(x_state)
x = torch.cat(x_state, action)
x = nn.Linear(hidden_dim+action_dim, hidden_dim_1)(x)
x = F.relu(x)
x = nn.Linear(hidden_dim_1, 1)(x)
```

The actor network is represented by a neural network with three fully connected layers.
```x = nn.Linear(state_dim,hidden_dim)(state)
x = F.relu(x)
x = nn.Linear(hidden_dim, hidden_dim_1)(x)
x = F.relu(x)
x = nn.Linear(hidden_dim_1, action_dim)(x)
x = F.tanh(x)
```

### Hyperparameters
- GAMMA=0.99
- hidden_dim = 256
- hidden_dim_1 = 128
- Target critic and target actor network are are updated through soft update (target_ parameter = (1-tau)* target_ parameter + tau* local_ parameter)
- tau = 1E-3
- Batch_size = 64
- Critic optimizer: Adam with learning rate 5E-4
- Actor optimizer: Adam with learning rate 2E-4

## Results
The environment is solved in 283 episodes.

![Training score](https://github.com/ccakarolotw/Deep-reinforment-learning-Unity-Reacher-environment-DDPG/blob/main/scores.png)
