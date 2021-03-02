# Report

The code uses DDPG to solve the Unity Reacher environment. 

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

The actor and target actor network is represented by a neural network with three fully connected layers.
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
The environment is solved in 283 episodes. The neural network weights are saved in `checkpoint_actor.pth` and `checkpoint_critic.pth`

<img src="https://github.com/ccakarolotw/Deep-reinforment-learning-Unity-Reacher-environment-DDPG/blob/main/Unity-Reacher_trained.gif" width="500" >

![Training score](https://github.com/ccakarolotw/Deep-reinforment-learning-Unity-Reacher-environment-DDPG/blob/main/scores.png)

## Ideas for future work
The performance of the agent can be further improved by
- Tuning hyperparameters
- Try other deep reinforment techniques such as trust region policy optimization and proximal policy optimization 
