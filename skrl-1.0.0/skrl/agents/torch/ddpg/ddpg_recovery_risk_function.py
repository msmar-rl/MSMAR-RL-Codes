from typing import Any, Dict, Optional, Tuple, Union

import copy
import gym
import gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch import Agent,SafeAgent
from skrl.memories.torch import Memory
from skrl.models.torch import Model


# [start-config-dict-torch]
DDPG_RECOVERY_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size

    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.005,                # soft update hyperparameter (tau)

    "actor_learning_rate": 1e-3,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0,            # clipping coefficient for the norm of the gradients

    "exploration": {
        "noise": None,              # exploration noise
        "initial_scale": 1.0,       # initial scale for the noise
        "final_scale": 1e-3,        # final scale for the noise
        "timesteps": None,          # timesteps for the noise decay
    },

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": 250,      # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 1000,        # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]


class DDPG_Recovery_Risk_Function(SafeAgent):
    def __init__(self,
                 models: Dict[str, Model],
                 memory_safe: Optional[Union[Memory, Tuple[Memory]]] = None,
                 memory_unsafe: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Deep Deterministic Policy Gradient (DDPG)

        https://arxiv.org/abs/1509.02971

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(DDPG_RECOVERY_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models=models,
                         memory_safe=memory_safe,
                         memory_unsafe=memory_unsafe,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=_cfg)

        # models
        # state risk critic
        self.state_risk_critic = self.models.get("state_risk_critic",None)
        self.state_action_risk_critic = self.models.get("state_action_risk_critic",None)

        # self.target_state_risk_critic = self.models.get("target_state_risk_critic",None)
        # self.target_state_action_risk_critic = self.models.get("target_state_action_risk_critic",None)
        # reward network
        self.reward_policy = self.models.get("reward_policy", None)
        self.target_reward_policy = self.models.get("target_reward_policy", None)
        self.reward_critic = self.models.get("reward_critic", None)
        self.target_reward_critic = self.models.get("target_reward_critic", None)
        # recovery network
        self.recovery_policy = self.models.get("recovery_policy", None)
        self.target_recovery_policy = self.models.get("target_recovery_policy", None)
        self.recovery_critic = self.models.get("recovery_critic", None)
        self.target_recovery_critic = self.models.get("target_recovery_critic", None)

        # checkpoint models
        self.checkpoint_modules["state_risk_critic"] = self.state_risk_critic
        self.checkpoint_modules["state_action_risk_critic"] = self.state_action_risk_critic
        # self.checkpoint_modules["target_state_risk_critic"] = self.target_state_risk_critic
        # self.checkpoint_modules["target_state_action_risk_critic"] = self.target_state_action_risk_critic

        self.checkpoint_modules["reward_policy"] = self.reward_policy
        self.checkpoint_modules["target_reward_policy"] = self.target_reward_policy
        self.checkpoint_modules["reward_critic"] = self.reward_critic
        self.checkpoint_modules["target_reward_critic"] = self.target_reward_critic

        self.checkpoint_modules["recovery_policy"] = self.recovery_policy
        self.checkpoint_modules["target_recovery_policy"] = self.target_recovery_policy
        self.checkpoint_modules["recovery_critic"] = self.recovery_critic
        self.checkpoint_modules["target_recovery_critic"] = self.target_recovery_critic

        if self.target_reward_policy is not None and self.target_reward_critic is not None and self.target_recovery_critic is not None and self.target_recovery_critic is not None:
            # self.target_state_risk_critic.freeze_parameters(True)
            # self.target_state_action_risk_critic.freeze_parameters(True)
            
            self.target_reward_policy.freeze_parameters(True)
            self.target_reward_critic.freeze_parameters(True)

            self.target_recovery_policy.freeze_parameters(True)
            self.target_recovery_critic.freeze_parameters(True)

            # # freeze reward network
            # self.reward_policy.freeze_parameters(True)
            # self.reward_critic.freeze_parameters(True)

            # update target networks (hard update)
            # self.target_state_risk_critic.update_parameters(self.state_risk_critic,polyak=1)
            # self.target_state_action_risk_critic.update_parameters(self.state_action_risk_critic,polyak=1)

            self.target_reward_policy.update_parameters(self.reward_policy, polyak=1)
            self.target_reward_critic.update_parameters(self.reward_critic, polyak=1)

            self.target_recovery_policy.update_parameters(self.recovery_policy, polyak=1)
            self.target_recovery_critic.update_parameters(self.recovery_critic, polyak=1)

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

        self._actor_learning_rate = self.cfg["actor_learning_rate"]
        self._critic_learning_rate = self.cfg["critic_learning_rate"]
        
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._recovery_state_preprocessor = self.cfg["state_preprocessor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._grad_norm_clip = self.cfg["grad_norm_clip"]

        self._exploration_noise = self.cfg["exploration"]["noise"]
        self._exploration_initial_scale = self.cfg["exploration"]["initial_scale"]
        self._exploration_final_scale = self.cfg["exploration"]["final_scale"]
        self._exploration_timesteps = self.cfg["exploration"]["timesteps"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        # set up optimizers and learning rate schedulers
        if self.reward_policy is not None and self.reward_critic is not None and self.recovery_critic is not None and self.recovery_policy is not None:
            
            self.state_risk_critic_optimizer = torch.optim.Adam(self.state_risk_critic.parameters(),lr=self._critic_learning_rate)
            self.state_action_risk_critic_optimizer = torch.optim.Adam(self.state_action_risk_critic.parameters(),lr=self._critic_learning_rate)

            self.reward_policy_optimizer = torch.optim.Adam(self.reward_policy.parameters(), lr=self._actor_learning_rate)
            self.reward_critic_optimizer = torch.optim.Adam(self.reward_critic.parameters(), lr=self._critic_learning_rate)
            self.recovery_policy_optimizer = torch.optim.Adam(self.recovery_policy.parameters(), lr=self._actor_learning_rate)
            self.recovery_critic_optimizer = torch.optim.Adam(self.recovery_critic.parameters(), lr=self._critic_learning_rate)
            if self._learning_rate_scheduler is not None:
                self.state_risk_critic_scheduler = self._learning_rate_scheduler(self.state_risk_critic_optimizer, **self.cfg["learning_rate_scheduler_kwargs"])
                self.state_action_risk_critic_scheduler = self._learning_rate_scheduler(self.state_action_risk_critic_optimizer, **self.cfg["learning_rate_scheduler_kwargs"])
                self.reward_policy_scheduler = self._learning_rate_scheduler(self.reward_policy_optimizer, **self.cfg["learning_rate_scheduler_kwargs"])
                self.reward_critic_scheduler = self._learning_rate_scheduler(self.reward_critic_optimizer, **self.cfg["learning_rate_scheduler_kwargs"])
                self.recovery_policy_scheduler = self._learning_rate_scheduler(self.recovery_policy_optimizer, **self.cfg["learning_rate_scheduler_kwargs"])
                self.recovery_critic_scheduler = self._learning_rate_scheduler(self.recovery_critic_optimizer, **self.cfg["learning_rate_scheduler_kwargs"])

            self.checkpoint_modules["state_risk_critic_optimizer"] = self.state_risk_critic_optimizer
            self.checkpoint_modules["state_action_risk_critic_optimizer"] = self.state_action_risk_critic_optimizer
            self.checkpoint_modules["reward_policy_optimizer"] = self.reward_policy_optimizer
            self.checkpoint_modules["reward_critic_optimizer"] = self.reward_critic_optimizer
            self.checkpoint_modules["recovery_policy_optimizer"] = self.recovery_policy_optimizer
            self.checkpoint_modules["recovery_critic_optimizer"] = self.recovery_critic_optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor
            
        if self._recovery_state_preprocessor:
            self._recovery_state_preprocessor = self._recovery_state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["recovery_state_preprocessor"] = self._recovery_state_preprocessor
        else:
            self._recovery_state_preprocessor = self._empty_preprocessor


    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent
        """
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory_safe is not None:
            self.memory_safe.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory_safe.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory_safe.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory_safe.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory_safe.create_tensor(name="terminated", size=1, dtype=torch.bool)

            self.memory_safe.create_tensor(name="safe_value",size=1,dtype=torch.float32)

            self._tensors_names = ["states", "actions", "rewards", "next_states", "terminated","safe_value"]

        if self.memory_unsafe is not None:
            self.memory_unsafe.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory_unsafe.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory_unsafe.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory_unsafe.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory_unsafe.create_tensor(name="terminated", size=1, dtype=torch.bool)

            self.memory_unsafe.create_tensor(name="safe_value",size=1,dtype=torch.float32)

            self._tensors_names = ["states", "actions", "rewards", "next_states", "terminated","safe_value"]

        # clip noise bounds
        if self.action_space is not None:
            self.clip_actions_min = torch.tensor(self.action_space.low, device=self.device)
            self.clip_actions_max = torch.tensor(self.action_space.high, device=self.device)

    def act(self, states: torch.Tensor, safe_value:torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        safe_count = 0 
        # sample random actions
        if timestep < self._random_timesteps:
            return self.reward_policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        #
        # print(safe_value)
        # print(safe_value.shape)

        # actions = torch.zeros(100,3).to("cuda:2")


        actions, _, outputs = self.reward_policy.act({"states": self._state_preprocessor(states)}, role="policy")
        # print(actions)
        # print(self.reward_policy.act({"states": (states)}, role="policy"))

        # print(self._exploration_noise)
        # print(states)
        # print(self._state_preprocessor(states))
        # print(actions)
        # add exloration noise
        # if self._exploration_noise is not None:
        #     # sample noises
        #     noises = self._exploration_noise.sample(actions.shape)

        #     # define exploration timesteps
        #     scale = self._exploration_final_scale
        #     if self._exploration_timesteps is None:
        #         self._exploration_timesteps = timesteps

        #     # apply exploration noise
        #     if timestep <= self._exploration_timesteps:
        #         scale = (1 - timestep / self._exploration_timesteps) \
        #               * (self._exploration_initial_scale - self._exploration_final_scale) \
        #               + self._exploration_final_scale
        #         noises.mul_(scale)

        #         # modify actions
        #         actions.add_(noises)
        #         actions.clamp_(min=self.clip_actions_min, max=self.clip_actions_max)

        #         # record noises
        #         self.track_data("Exploration / Exploration noise (max)", torch.max(noises).item())
        #         self.track_data("Exploration / Exploration noise (min)", torch.min(noises).item())
        #         self.track_data("Exploration / Exploration noise (mean)", torch.mean(noises).item())

        #     else:
        #         # record noises
        #         self.track_data("Exploration / Exploration noise (max)", 0)
        #         self.track_data("Exploration / Exploration noise (min)", 0)
        #         self.track_data("Exploration / Exploration noise (mean)", 0)

        return actions, None, outputs

    def record_transition(self,
                          states: torch.Tensor,
                          actions: torch.Tensor,
                          rewards: torch.Tensor,
                          next_states: torch.Tensor,
                          terminated: torch.Tensor,
                          truncated: torch.Tensor,
                          infos: Any,
                          timestep: int,
                          timesteps: int) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

        safe_value = infos["safe_values"].clone()
        safe_reward= infos["safe_rewards"].clone()
        

        # positive_mask = safe_value>0
        # negative_mask = safe_value<0
        
        safe_reward= safe_reward.reshape(100,1)
        safe_value = safe_value.reshape(100,1)
        # print(safe_value)
        # print(states[positive_mask])
        # # # this part , we directly store all the data without aparting the safe with unsafe 
        if self.memory_safe is not None:

            self.memory_safe.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                    terminated=terminated,safe_value=safe_value ,truncated=truncated)
            
            for memory in self.secondary_memories_safe:
                memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                   terminated=terminated,safe_value = safe_value ,truncated=truncated)
        # if self.memory_unsafe is not None:

        #     self.memory_unsafe.add_samples(states=states, actions=actions, rewards=safe_reward, next_states=next_states,
        #                             terminated=terminated, safe_value = safe_value,truncated=truncated)
        #     for memory in self.secondary_memories_unsafe:
        #         memory.add_samples(states=states, actions=actions, rewards=safe_reward, next_states=next_states,
        #                            terminated=terminated, safe_value = safe_value, truncated=truncated)

        # # this part , we store the data into different memory 
        # if self.memory_safe is not None:

        #     self.memory_safe.add_samples(states=states[negative_mask], actions=actions[negative_mask], rewards=rewards[negative_mask], next_states=next_states[negative_mask],
        #                             terminated=terminated[negative_mask],safe_value=safe_value[negative_mask] ,truncated=truncated[negative_mask])
            
        #     for memory in self.secondary_memories_safe:
        #         memory.add_samples(states=states[negative_mask], actions=actions[negative_mask], rewards=rewards[negative_mask], next_states=next_states[negative_mask],
        #                             terminated=terminated[negative_mask],safe_value=safe_value[negative_mask] ,truncated=truncated[negative_mask])
        # # print(self.memory_unsafe is not  None)
        # if self.memory_unsafe is not None:

        #     self.memory_unsafe.add_samples(states=states[positive_mask], actions=actions[positive_mask], rewards=safe_reward[positive_mask], next_states=next_states[positive_mask],
        #                             terminated=terminated[positive_mask], safe_value = safe_value[positive_mask],truncated=truncated[positive_mask])
        #     for memory in self.secondary_memories_unsafe:
        #         memory.add_samples(states=states[positive_mask], actions=actions[positive_mask], rewards=safe_reward[positive_mask], next_states=next_states[positive_mask],
        #                             terminated=terminated[positive_mask], safe_value = safe_value[positive_mask],truncated=truncated[positive_mask])

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if timestep >= self._learning_starts:
            self.set_mode("train")
            self._update_safe(timestep, timesteps)
            # self._update_unsafe(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update_safe(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # sample a batch from memory
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones, sampled_safe_value = \
            self.memory_safe.sample(names=self._tensors_names, batch_size=self._batch_size)[0]
        gamma = 0.95
        # gradient steps
        for gradient_step in range(self._gradient_steps):

            sampled_states = self._state_preprocessor(sampled_states)
            sampled_next_states = self._state_preprocessor(sampled_next_states)

            
            # # This part  comes from the paper of machine dog
            # # with torch.no_grad():
            # #     print("safe value")
            # #     print(sampled_safe_value)

            # #     next_actions, _, _ = self.target_reward_policy.act({"states": sampled_next_states}, role="target_policy")

            # #     print("target q")
            # #     risk_target_q_values, _, _ = self.target_state_action_risk_critic.act({"states": sampled_next_states, "taken_actions": next_actions}, role="target_critic")
            # #     print(risk_target_q_values)
            # #     print("q")
            # #     risk_critic_q_values, _, _ = self.state_action_risk_critic.act({"states": sampled_states, "taken_actions": sampled_actions}, role="critic")
            # #     print(risk_critic_q_values)
            # #     print("target v")
            # #     risk_target_v_values, _, _ = self.target_state_risk_critic.act({"states": sampled_next_states}, role="target_critic")
            # #     print(risk_target_v_values)
            # #     print("v")
            # #     risk_critic_v_values, _, _ = self.state_risk_critic.act({"states": sampled_states}, role="critic")
            # #     print(risk_critic_v_values)

            # # compute Q values and target values
            # with torch.no_grad():
            #     next_actions, _, _ = self.target_reward_policy.act({"states": sampled_next_states}, role="target_policy")
                
            #     risk_target_q_values, _, _ = self.state_action_risk_critic.act({"states": sampled_next_states, "taken_actions": next_actions}, role="target_critic")
            #     risk_critic_q_values_t_1 = sampled_safe_value + self._discount_factor * sampled_dones.logical_not() * risk_target_q_values

            # # compute critic loss
            # risk_critic_q_values_t, _, _  = self.state_action_risk_critic.act({"states": sampled_states, "taken_actions": sampled_actions}, role="critic")
            # ground_truth_q = torch.where(sampled_safe_value>0,sampled_safe_value,torch.min(sampled_safe_value,risk_critic_q_values_t_1))

            # ground_truth_q = gamma*ground_truth_q + (1-gamma)*sampled_safe_value
            # risk_critic_q_loss = F.mse_loss(risk_critic_q_values_t, ground_truth_q)

            # # optimization step (critic)
            # self.state_action_risk_critic_optimizer.zero_grad()
            # risk_critic_q_loss.backward()
            # if self._grad_norm_clip > 0:
            #     nn.utils.clip_grad_norm_(self.state_action_risk_critic.parameters(), self._grad_norm_clip)
            # self.state_action_risk_critic_optimizer.step()

            # # compute V values and target values
            # with torch.no_grad():
            #     next_actions, _, _ = self.reward_policy.act({"states": sampled_next_states}, role="target_policy")
                
            #     risk_target_v_values, _, _ = self.state_risk_critic.act({"states": sampled_next_states}, role="target_critic")
            #     risk_critic_v_values_t_1 = sampled_safe_value + self._discount_factor * sampled_dones.logical_not() * risk_target_v_values

            # # compute critic loss
            # risk_critic_v_values_t, _, _ = self.state_risk_critic.act({"states": sampled_states}, role="critic")
            # ground_truth_v = torch.where(sampled_safe_value>0,sampled_safe_value,torch.min(sampled_safe_value,risk_critic_v_values_t_1))

            # ground_truth_v = gamma*ground_truth_v +(1-gamma)*sampled_safe_value

            # risk_critic_v_loss = F.mse_loss(risk_critic_v_values_t, ground_truth_v)

            # # optimization step (critic)
            # self.state_risk_critic_optimizer.zero_grad()
            # risk_critic_v_loss.backward()
            # if self._grad_norm_clip > 0:
            #     nn.utils.clip_grad_norm_(self.state_risk_critic.parameters(), self._grad_norm_clip)
            # self.state_risk_critic_optimizer.step()

            # update target networks
            # self.target_state_action_risk_critic.update_parameters(self.state_action_risk_critic, polyak=self._polyak)
            # self.target_state_risk_critic.update_parameters(self.state_risk_critic, polyak=self._polyak)



            # The second method is just a program method
            # just compute value by the network
            risk_critic_q_values_t, _, _  = self.state_action_risk_critic.act({"states": sampled_states, "taken_actions": sampled_actions}, role="critic")
            risk_critic_q_loss = F.mse_loss(risk_critic_q_values_t,sampled_safe_value)
            # optimization step (critic)
            self.state_action_risk_critic_optimizer.zero_grad()
            risk_critic_q_loss.backward()
            if self._grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(self.state_action_risk_critic.parameters(), self._grad_norm_clip)
            self.state_action_risk_critic_optimizer.step()


            risk_critic_v_values_t, _, _  = self.state_risk_critic.act({"states": sampled_states, "taken_actions": sampled_actions}, role="critic")
            risk_critic_v_loss = F.mse_loss(risk_critic_v_values_t,sampled_safe_value)
            # optimization step (critic)
            self.state_risk_critic_optimizer.zero_grad()
            risk_critic_v_loss.backward()
            if self._grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(self.state_risk_critic.parameters(), self._grad_norm_clip)
            self.state_risk_critic_optimizer.step()

            # here , we test the function of q vlaue and v value
            print("sampled safe value")
            print(sampled_safe_value)
            print("risk critic q value")
            print(risk_critic_q_values_t)
            print("risk critic v value")
            print(risk_critic_v_values_t)

            # update learning rate
            if self._learning_rate_scheduler:
                self.state_action_risk_critic_scheduler.step()
                self.state_risk_critic_scheduler.step()

            # record data
            self.track_data("Loss / risk_critic_q loss", risk_critic_q_loss.item())
            self.track_data("Loss / risk_critic_v loss", risk_critic_v_loss.item())

            if self._learning_rate_scheduler:
                self.track_data("Learning / Reward Policy learning rate", self.reward_policy_scheduler.get_last_lr()[0])
                self.track_data("Learning / Reward Critic learning rate", self.reward_critic_scheduler.get_last_lr()[0])
