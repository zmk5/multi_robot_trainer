# multi_robot_trainer

Deep RL ROS2 package for synchronous or asynchronous off-line or on-line training a single or multiple robots.

## Introduction

My research work particularly revolves around swarm intelligence and multi-agent reinforcement learning in the context of herding robots or with chemical reaction networks. The hardest part I've noticed is the lack of tooks capable of linking to *real* robots. Since I'm a huge fan of ROS2, I decided to release some of my work as an open source template for people to be able to use for their own projects. Specifically, this pacakge is capable of training robots with user defined states and output actions using popular architectures out there:

- Deep Q-Learning
- Double Deep Q-Learning
- REINFORCE
- Actor-Critic with a shared network
  - n-step Returns
  - Generalized Advantage Estimation (*turning on soon!*)
  - Entropy Regularization (*turning on soon!*)
- Actor-Critic with a dual network
  - n-step Returns
  - Generalized Advantage Estimation (*turning on soon!*)
  - Entropy Regularization (*turning on soon!*)
- Proximal Policy Optimization (*Coming Soon*)

All of these algorithims are written in *TensorFlow* 2.0 giving users an easier time to modify the networks however you would like!

## Quick Guide

### How to Install

This package is more of a *template* than an actual package for people to use. So to use, simply fork the project for your own needs or clone it and modify it however you would like! Remember to put this within a ROS2 workspace that you can create with the following:

```bash
~$ mkdir -p my_deep_rl_ws/src
~$ cd my_deep_rl_ws/src
~$ git clone https://github.com/zmk5/multi_robot_trainer
```

### Modify the Package

Once cloned/forked, make sure to modify the following file with what consititues a *step* with your environment:

`mrt_worker/mrt_worker/base.py`

Add what you need and feel free to use the the `Experience` class located within the `mrt_worker/utils` directory to have a lower memory storage capability. Once you have added what your step will be you can test out your network with the test launch file, `mrt_server/launch/test.launch.py`. Change line 72 to either `server_sync_dual_node` for networks that require two networks or `server_async_node` to run asynchronous training.

### Build Your Package

Build and source the package with the colcon command:

```bash
~$ colcon build --symlink-install && . install/local_setup.bash
```

### Run the Test Launch file

Once built run the test launch file with the following:

```bash
~$ ros2 launch mrt_server test.launch.py 'policy_type:=DQN'
```

Future updates will provide further explanations on options and how to run with Docker and Gazebo!
