# On-the-mutual-dependence-of-stochastic-actions-in-the-Soft-Actor-Critic-algorithm
The analysis of the mutual dependence between actions of multivariate stochastic policies within the Soft Actor-Critic algorithm.

## Overview of the problem
The Soft Actor-Critic is a deep reinforcement learning algorithm proposed in [Haarnoja et al.(2018)](https://arxiv.org/abs/1801.01290). The algorithms learners off-policy and aim to maximize both the expected reward and the entropy, which provides a gain in exploration and stability. Under this framework, the practical approach is to seek the optimal policy within a set of parametrized distributions such as Gaussian. This repository aims to investigate the mutual dependence between random actions taken under stochastic policies within the Soft Actor-Critic algorithm. We want to answer the following two questions:
* Does the SAC agent with mutually dependent stochastic actions perform better than its counterpart with independent actions when learning a simple continuous state-action environment?
* What is the correlation structure of the stochastic actions under the multivariate Gaussian policies? Specifically, are the mutual correlations between the actions statistically different from zero?

For this study's purpose, we use 'HopperBulletEnv-v0' environment, an open-source version of the [MuJuCo](http://www.mujoco.org) environment, powered by the [Bullet Physics](https://pybullet.org/wordpress/) engine. This environment features 15 continuous state variables and 3 continuous action variables. The goal of the agent is to move the hopper robot forward with minimal energy.

## Numerical solution

We extend the SAC algorithm of [Miguel Morales (@mimoralea)](https://github.com/mimoralea) by adding a separate soft value function approximator. In the original paper by [Haarnoja et al.(2018)](https://arxiv.org/abs/1801.01290), this addition is claimed to stabilize training process. Moreover, we change the realization of the replay buffer, to make it based on the deque data structure. Finally, we provide a stochastic policy based on multivariate Gaussina distribution. For this purpose, the neural network, which approximate the policy h<sub>&theta;</sub>(x) = &theta;<sub>o</sub> x + &theta;<sub>1</sub>x
