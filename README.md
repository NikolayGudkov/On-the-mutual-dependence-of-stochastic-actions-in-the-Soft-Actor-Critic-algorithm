# On the mutual dependence between stochastic actions in the Soft Actor-Critic algorithm
The analysis of the mutual dependence between actions of multivariate stochastic policies within the Soft Actor-Critic algorithm.

## Overview of the problem
The Soft Actor-Critic (SAC) is an off-policy deep reinforcement learning algorithm proposed in [Haarnoja et al.(2018)](https://arxiv.org/abs/1801.01290). The algorithm aims to maximize both the expected reward and the policy's entropy, thereby gaining the exploration and stability of the training process. The practical approach for the SAC algorithm is to seek the optimal policy within a set of parametrized distributions (e.g. multivariate Gaussian). 

This repository investigates the mutual dependence between random actions taken under stochastic policies within the Soft Actor-Critic algorithm. We want to answer the following two questions:
* Does the SAC agent with mutually dependent stochastic actions perform better than its counterpart with independent actions when learning a simple environment with continuous state-action space?
* What is the correlation structure of the stochastic actions under the multivariate Gaussian policies? Specifically, are the mutual correlations between the actions statistically different from zero?

For this study's purpose, we use 'HopperBulletEnv-v0', an open-source version of the [MuJuCo](http://www.mujoco.org) environment, powered by the [Bullet Physics](https://pybullet.org/wordpress/) engine. This environment features 15 continuous state variables and three continuous action variables. The goal of the agent is to move the hopper robot forward with minimal energy.

## Numerical solution

We extend the SAC algorithm implementation in [Miguel Morales (@mimoralea)](https://github.com/mimoralea) by adding a separate soft value function approximator. In the original paper, [Haarnoja et al.(2018)](https://arxiv.org/abs/1801.01290), this addition provides better stability of the training process. Moreover, we use the deque data structure for the replay buffer. It should improve the computational performance of storing observations and sampling batches of the past experiences from the buffer for training neural networks. Finally, we implement a stochastic policy based on multivariate Gaussian distribution. For this purpose, the neural network, which approximates the policy &pi;(a|s), returns a tuple of &mu;, log(&sigma;), l. Here, &mu; is the vector of actions' means, while &sigma; and l are two vectors of the size |A| and (|A|-1)*(|A|)/2, respectively. The vectors &sigma; and l are used to fill the non-zero elements of the lower triangular matrix **L**, which in the lower triangular matrix in the Cholesky decomposition of the covariance matrix **S=LL'**.

## Simulation results

To answer the two questions above, we run a series of numerical experiments of training the SAC algorithm with independent and dependent stochastic actions. In the both cases, in order to obtain comparable numerical results, we use the same random seed for the trainging process. The training ends when the agent achieves a moving average of 1500 across the previous 100 episodes.

![Figure 1](https://github.com/NikolayGudkov/On-the-mutual-dependence-of-stochastic-actions-in-the-Soft-Actor-Critic-algorithm/blob/main/Hopper_dependent_independent_actions.png)


From the figure above, we note that the SAC agent with dependent stochastic actions solves the environment substantially faster than the counterpart with independent actions. It takes the first agent less than 600 episodes to achieve the training target, while the second agent needs around 1000 episodes. Partially, this observation can be explained by the fact that the former agent seeks the optimal policy in a set of more parametrized Gaussian distributions. This set provides more flexible learning policies and allows the agent to perform better than the agent, whose actions are forces to be independent.

During the agent's training process with mutually dependent actions, we collect a batch of experiences and keep them in the replay buffer. These experiences contain a set of states that the agent has encountered during the training. Once the agent is considered fully trained (i.e., after the moving average of the cumulative rewards per episode exceeds 1500), the policy network is assumed to approximate the optimal policy. We pass the batch of all states visited by the agent to this network and constructs a set of correlations between stochastic Gaussian actions under the optimal policy in each state. Since there are three continuous actions in the 'HopperBulletEnv-v0' environment,(a<sub>1</sub>,a<sub>2</sub>,a<sub>3</sub>), we get a pair-vise correlation triplets batch,(&rho;(a<sub>1</sub>,a<sub>2</sub>), &rho;(a<sub>1</sub>,a<sub>3</sub>), &rho;(a<sub>2</sub>,a<sub>3</sub>)). In order to answer the second question above, we do the following.

We construct three histograms for all pair-vise correlations. From these histograms, we observe that under the optimal policy the correlation between the first two actions &rho;(a<sub>1</sub>,a<sub>2</sub>) is positive for the most of the states. While the histograms for the other two correlations &rho;(a<sub>1</sub>,a<sub>3</sub>) and &rho;(a<sub>2</sub>,a<sub>3</sub>) are centered close to zero, there are states for which the correlations under the optimal policies are highly non-negative. We compute the sample means for &rho;(a<sub>1</sub>,a<sub>2</sub>), &rho;(a<sub>1</sub>,a<sub>3</sub>), &rho;(a<sub>2</sub>,a<sub>3</sub>) to be 0.3757, -0.0565 and 0.0885, respectively.

Using the pair-vise correlations samples, we perform a t-test to hypothesize that the sample means of each correlation is zero. The zero hypotheses have been rejected for all three samples, with t-statistics being very large in absolute terms: 716.63, -126.23, and 182.79. 

