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

![Figure 1](https://github.com/NikolayGudkov/Some-analysis-of-the-Soft-Actor-Critic-algorithm/blob/main/SAC_plus_1.png)


From 

