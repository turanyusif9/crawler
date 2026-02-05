## Project: Continuous Control (Crawler)   

### Introduction

For this project, we work with the [Crawler](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler) environment. A creature with 4 arms and 4 forearms.   

Original Agent Reward Function :
* +0.03 times body velocity in the goal direction.
* +0.01 times body direction alignment with goal direction.

Modified Agent Reward Function:

Agent reward function is updated using three different penalties based on smoothness, symmetry, and magnitude parameters. These penalties can be turned on and off in `Agent` object in `ppo_agent_combined.py` file.


![](images/crawler.gif)

### Prepare environment on the local machine

You need at least the following three packages:

1. **deep-reinforcement-learning  (DRLND)**        
   The instructions to set up the DRLND repository can be found [here](https://github.com/udacity/deep-reinforcement-learning#dependencies). This repository contains material related to Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.

2. **ml-agents  (ML-Agents Toolkit)**
   To configure the ML-Agents Toolkit for Windows you need to complete the following steps:
    
    2.1  Creating a new Conda environment:
    
       conda create -n ml-agents python=3.6
       
    2.2 Activating ml-agents by the following command:
    
       activate ml-agents
       
    2.3 Latest versions of TensorFlow won't work, so you will need to make sure that you install version 1.7.1:
    
       pip install tensorflow==1.7.1
       
    For details on installing the ML-Agents Toolkit, see the instructions [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation-Windows.md).     
    
3. **Unity environment _Crawler_**

    For this project, we not need to install Unity because the environment already built. For 20 agents, the environment     
    can be downloaded as follows:

   Windows (64-bit), [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86_64.zip)    
   Windows (32-bit), [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86.zip)     

   Download this environment zip into  **p2_continuous-control/** folder, and unzip the file.
   
### Update mechanism

Standard policy gradient methods perform one gradient update per data sample.     
In the [original paper](https://arxiv.org/abs/1707.06347) it was proposed a novel objective function that enables **multiple epochs**.   
This is  the **loss** function _L\_t(\\theta)_, which is (approximately) maximized each iteration:    

![](images/objective_function_07.png)

Parameters **c1**, **c2** and **epoch** are essential hyperparameters in the PPO algorithm.
In this agent, c1 = -0.5,   c2 = 0.01. 

                policy_loss = -torch.min(obj, obj_clipped) - 0.01 * entropy_loss
                value_loss = 0.5 * (sampled_returns - values).pow(2)
                loss = policy_loss + value_loss 

The update is performed in the function **agent.step()**.


### Train the Agent

   To use the original model, `Agent` from `ppo_agent.py` must be imported. 

   To use the updated model that includes penalties, `Agent` from `ppo_agent_combined.py` must be imported.


   
### Watch the Trained Agent

For both neural networks, the actor and the critic, we save the trained weights into checkpoint files   
with the extension pth.  For all cases, the corresponding files are saved into the directory checkpoints.    
Using notebook _WatchAgent.ipynb_ we can load the trained weights and replay them.


### Credit   

Most of the code is based on Udacity's PPO code and the repo by Rafael1s, Rafael Stekolshchik.
   
