from __future__ import annotations

import copy

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .architectures import EIIE
from .utils import apply_portfolio_noise
from .utils import PVM
from .utils import ReplayBuffer
from .utils import RLDataset


def _calc_metrics(portfolio_values: list) -> dict:
    """根据组合净值序列计算年化 Sharpe 和最大回撤。"""
    arr = np.array(portfolio_values, dtype=float)

    if len(arr) < 2:
        return {"sharpe": 0.0, "max_drawdown": 0.0}

    rets = np.diff(arr) / arr[:-1]
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    peak = np.maximum.accumulate(arr)
    max_dd = float(((arr - peak) / peak).min())
    return {"sharpe": sharpe, "max_drawdown": max_dd}


def _get_action_memory(env):
    """
    兼容不同版本环境中的动作历史字段。
    优先尝试:
      1) env._asset_memory["actions"]
      2) env._actions_memory / env._action_memory / ...
    """
    asset_memory = getattr(env, "_asset_memory", None)
    if isinstance(asset_memory, dict):
        actions = asset_memory.get("actions", None)
        if actions is not None:
            return actions

    candidate_names = [
        "_actions_memory",
        "_action_memory",
        "actions_memory",
        "action_memory",
    ]
    for name in candidate_names:
        if hasattr(env, name):
            memory = getattr(env, name)
            if memory is not None:
                return memory

    return []


class PolicyGradient:
    """Class implementing policy gradient algorithm to train portfolio
    optimization agents.

    Note:
        During testing, the agent can be optimized through online learning.
        To disable online learning, set learning rate to 0 or use a very
        large online_training_period.

    Attributes:
        train_env: Environment used to train the agent.
        train_policy: Policy used in training.
        test_env: Environment used to test the agent.
        test_policy: Policy after test online learning.
    """

    def __init__(
        self,
        env,
        policy=EIIE,
        policy_kwargs=None,
        validation_env=None,
        batch_size=100,
        lr=1e-3,
        action_noise=0,
        optimizer=AdamW,
        device="cpu",
    ):
        """Initializes Policy Gradient for portfolio optimization.

        Args:
          env: Training environment.
          policy: Policy architecture to be used.
          policy_kwargs: Arguments to be used in the policy network.
          validation_env: Validation environment.
          batch_size: Batch size to train neural network.
          lr: Policy neural network learning rate.
          action_noise: Noise parameter (between 0 and 1) to be applied
            during training.
          optimizer: Optimizer class of neural network.
          device: Device where neural network is run.
        """
        self.policy = policy
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.validation_env = validation_env
        self.batch_size = batch_size
        self.lr = lr
        self.action_noise = action_noise
        self.optimizer = optimizer
        self.device = device
        self._setup_train(env, self.policy, self.batch_size, self.lr, self.optimizer)

    def _setup_train(self, env, policy, batch_size, lr, optimizer):
        """Initializes algorithm before training."""
        self.train_env = env

        self.train_policy = policy(**self.policy_kwargs).to(self.device)
        self.train_optimizer = optimizer(self.train_policy.parameters(), lr=lr)

        self.train_batch_size = batch_size
        self.train_buffer = ReplayBuffer(capacity=batch_size)
        self.train_pvm = PVM(self.train_env.episode_length, env.portfolio_size)

        dataset = RLDataset(self.train_buffer)
        self.train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def train(self, episodes=100):
        """Training sequence.

        Args:
            episodes: Number of episodes to simulate.
        """
        for _ in tqdm(range(1, episodes + 1)):
            obs = self.train_env.reset()
            self.train_pvm.reset()
            done = False

            while not done:
                last_action = self.train_pvm.retrieve()
                obs_batch = np.expand_dims(obs, axis=0)
                last_action_batch = np.expand_dims(last_action, axis=0)

                action = apply_portfolio_noise(
                    self.train_policy(obs_batch, last_action_batch),
                    self.action_noise,
                )
                self.train_pvm.add(action)

                next_obs, reward, done, info = self.train_env.step(action)

                exp = (obs, last_action, info["price_variation"], info["trf_mu"])
                self.train_buffer.append(exp)

                if len(self.train_buffer) == self.train_batch_size:
                    self._gradient_ascent()

                obs = next_obs

            # episode 末尾剩余数据
            self._gradient_ascent()

            # 如果传了 validation_env，则按 demo 风格在每个 episode 末尾跑一次验证
            if self.validation_env is not None:
                self.test(
                    self.validation_env,
                    policy=None,
                    online_training_period=10**18,
                    batch_size=self.train_batch_size,
                    lr=0.0,
                    optimizer=None,
                )

    def _setup_test(self, env, policy, batch_size, lr, optimizer):
        """Initializes algorithm before testing."""
        self.test_env = env

        policy = self.train_policy if policy is None else policy
        lr = self.lr if lr is None else lr
        optimizer = self.optimizer if optimizer is None else optimizer

        self.test_policy = copy.deepcopy(policy).to(self.device)
        self.test_optimizer = optimizer(self.test_policy.parameters(), lr=lr)

        self.test_buffer = ReplayBuffer(capacity=batch_size)
        self.test_pvm = PVM(self.test_env.episode_length, env.portfolio_size)

        dataset = RLDataset(self.test_buffer)
        self.test_dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def test(
        self,
        env,
        policy=None,
        online_training_period=10,
        batch_size=None,
        lr=None,
        optimizer=None,
    ):
        """Tests the policy with optional online learning.

        Args:
          env: Environment to be used in testing.
          policy: Policy architecture to be used. If None, it will use the training
            architecture.
          online_training_period: Period in which an online training will occur. To
            disable online learning, use a very big value.
          batch_size: Batch size to train neural network. If None, it will use the
            training batch size.
          lr: Policy neural network learning rate. If None, it will use the training
            learning rate.
          optimizer: Optimizer of neural network. If None, it will use the training
            optimizer.
        """
        batch_size = self.train_batch_size if batch_size is None else batch_size
        self._setup_test(env, policy, batch_size, lr, optimizer)

        obs = self.test_env.reset()
        self.test_pvm.reset()
        done = False
        steps = 0

        while not done:
            steps += 1

            last_action = self.test_pvm.retrieve()
            obs_batch = np.expand_dims(obs, axis=0)
            last_action_batch = np.expand_dims(last_action, axis=0)
            action = self.test_policy(obs_batch, last_action_batch)
            self.test_pvm.add(action)

            next_obs, reward, done, info = self.test_env.step(action)

            exp = (obs, last_action, info["price_variation"], info["trf_mu"])
            self.test_buffer.append(exp)

            # lr <= 0 时直接跳过 online learning
            if lr is not None and lr <= 0:
                pass
            else:
                if steps % online_training_period == 0:
                    self._gradient_ascent(test=True)

            obs = next_obs

    def _gradient_ascent(self, test=False):
        """Performs the gradient ascent step in the policy gradient algorithm."""
        if test:
            if len(self.test_buffer) == 0:
                return None
        else:
            if len(self.train_buffer) == 0:
                return None

        obs, last_actions, price_variations, trf_mu = (
            next(iter(self.test_dataloader))
            if test
            else next(iter(self.train_dataloader))
        )

        obs = obs.to(self.device)
        last_actions = last_actions.to(self.device)
        price_variations = price_variations.to(self.device)
        trf_mu = trf_mu.unsqueeze(1).to(self.device)

        mu = (
            self.test_policy.mu(obs, last_actions)
            if test
            else self.train_policy.mu(obs, last_actions)
        )

        policy_loss = -torch.mean(
            torch.log(torch.sum(mu * price_variations * trf_mu, dim=1))
        )

        if test:
            self.test_optimizer.zero_grad()
            policy_loss.backward()
            self.test_optimizer.step()
        else:
            self.train_optimizer.zero_grad()
            policy_loss.backward()
            self.train_optimizer.step()

        return policy_loss.item()