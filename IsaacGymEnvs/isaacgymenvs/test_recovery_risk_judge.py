import isaacgym

import torch
import torch.nn as nn

import copy

import random
import numpy as np


# import the skrl components to build the RL system
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG,DDPG_Recovery_Risk_Judge,DDPG_Recovery_Long_Risk
from skrl.agents.torch.ddpg import DDPG_Recovery,DDPG_RECOVERY_DEFAULT_CONFIG,DDPG_Recovery_Pretrain,DDPG_Recovery_Same_Memory
from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# seed for reproducibility
set_seed(1000)  # e.g. `set_seed(42)` for fixed seed


# define models (deterministic models) using mixins
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, self.num_actions),
                                 nn.Tanh())

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}

class V_Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations , 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

# load and wrap the Isaac Gym environment
env = load_isaacgym_env_preview4(task_name="UAV_multi_obstacle_recovery_test")

env = wrap_env(env)

device = env.device


# instantiate a memory as experience replay
safe_memory = RandomMemory(memory_size=15625, num_envs=env.num_envs, device=device)
unsafe_memory = RandomMemory(memory_size=15625, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#models
models = {}
models["state_risk_critic_1"] = V_Critic(env.observation_space, env.action_space, device)
models["state_action_risk_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["state_risk_critic_2"] = V_Critic(env.observation_space, env.action_space, device)
models["state_action_risk_critic_2"] = Critic(env.observation_space, env.action_space, device)

models["reward_policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["target_reward_policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["reward_critic"] = Critic(env.observation_space, env.action_space, device)
models["target_reward_critic"] = Critic(env.observation_space, env.action_space, device)

models["recovery_policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["target_recovery_policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["recovery_critic"] = Critic(env.observation_space, env.action_space, device)
models["target_recovery_critic"] = Critic(env.observation_space, env.action_space, device)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#configuration-and-hyperparameters
cfg = DDPG_RECOVERY_DEFAULT_CONFIG.copy()
cfg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.5, device=device)
cfg["gradient_steps"] = 1
cfg["batch_size"] = 4096
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 5e-4
cfg["critic_learning_rate"] = 5e-4
cfg["random_timesteps"] = 1
cfg["learning_starts"] = 100
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 800
cfg["experiment"]["checkpoint_interval"] = 8000
cfg["experiment"]["directory"] = "runs/torch/UAV/ddpg_recovery/multi"


agent_1 = DDPG_Recovery_Risk_Judge(models=models,
             memory_safe=safe_memory,
             memory_unsafe=unsafe_memory,
             cfg=cfg,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=device)

agent_2 = DDPG_Recovery_Pretrain(models=models,
             memory_safe=safe_memory,
             memory_unsafe=unsafe_memory,
             cfg=cfg,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=device)

agent_3 = DDPG_Recovery_Long_Risk(models=models,
             memory_safe=safe_memory,
             memory_unsafe=unsafe_memory,
             cfg=cfg,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=device)


agent_2.load("./runs/torch/UAV/ddpg_recovery/multi/24-10-30_11-31-34-019125_DDPG_Recovery/checkpoints/best_agent.pt")

agent_3.load("./runs/torch/UAV/ddpg_recovery/multi/25-01-12_23-15-00-499547_DDPG_Recovery_Long_Risk/checkpoints/agent_100000.pt")

agent_1.load("./runs/torch/UAV/ddpg_recovery/multi/25-01-07_03-45-00-402145_DDPG_Recovery_Same_Memory/checkpoints/agent_104000.pt")


agent_1.models["state_risk_critic_1"].load_state_dict(agent_3.models["state_risk_critic_1"].state_dict())
agent_1.models["state_action_risk_critic_1"].load_state_dict(agent_3.models["state_action_risk_critic_1"].state_dict())
agent_1.models["state_risk_critic_2"].load_state_dict(agent_3.models["state_risk_critic_2"].state_dict())
agent_1.models["state_action_risk_critic_2"].load_state_dict(agent_3.models["state_action_risk_critic_2"].state_dict())

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 180000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_1)

# start training
trainer.eval()
