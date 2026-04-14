# from __future__ import annotations

# import copy

# import numpy as np
# import torch
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# from .architectures_changed import EIIE
# from .utils import apply_portfolio_noise
# from .utils import PVM
# from .utils import ReplayBuffer
# from .utils import RLDataset


# class PolicyGradient:
#     """Class implementing policy gradient algorithm to train portfolio
#     optimization agents.

#     Note:
#         During testing, the agent can be optimized through online learning.
#         To disable online learning, set learning rate to 0 or use a very
#         large online_training_period.

#     Attributes:
#         train_env: Environment used to train the agent.
#         train_policy: Policy used in training.
#         test_env: Environment used to test the agent.
#         test_policy: Policy after test online learning.
#     """

#     def __init__(
#         self,
#         env,
#         policy=EIIE,
#         policy_kwargs=None,
#         validation_env=None,
#         batch_size=100,
#         lr=1e-3,
#         action_noise=0,
#         optimizer=AdamW,
#         device="cpu",
#     ):
#         """Initializes Policy Gradient for portfolio optimization.

#         Args:
#           env: Training environment.
#           policy: Policy architecture to be used.
#           policy_kwargs: Arguments to be used in the policy network.
#           validation_env: Validation environment.
#           batch_size: Batch size to train neural network.
#           lr: Policy neural network learning rate.
#           action_noise: Noise parameter (between 0 and 1) to be applied
#             during training.
#           optimizer: Optimizer class of neural network.
#           device: Device where neural network is run.
#         """
#         self.policy = policy
#         self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
#         self.validation_env = validation_env
#         self.batch_size = batch_size
#         self.lr = lr
#         self.action_noise = action_noise
#         self.optimizer = optimizer
#         self.device = device
#         self._setup_train(env, self.policy, self.batch_size, self.lr, self.optimizer)

#     def _setup_train(self, env, policy, batch_size, lr, optimizer):
#         """Initializes algorithm before training."""
#         self.train_env = env

#         self.train_policy = policy(**self.policy_kwargs).to(self.device)
#         self.train_optimizer = optimizer(self.train_policy.parameters(), lr=lr)

#         self.train_batch_size = batch_size
#         self.train_buffer = ReplayBuffer(capacity=batch_size)
#         self.train_pvm = PVM(self.train_env.episode_length, env.portfolio_size)

#         dataset = RLDataset(self.train_buffer)
#         self.train_dataloader = DataLoader(
#             dataset=dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             pin_memory=True,
#         )

#     def train(self, episodes=100):
#         """Training sequence.

#         Args:
#             episodes: Number of episodes to simulate.
#         """
#         for _ in tqdm(range(1, episodes + 1)):
#             obs = self.train_env.reset()
#             self.train_pvm.reset()
#             done = False

#             while not done:
#                 last_action = self.train_pvm.retrieve()
#                 obs_batch = np.expand_dims(obs, axis=0)
#                 last_action_batch = np.expand_dims(last_action, axis=0)

#                 action = apply_portfolio_noise(
#                     self.train_policy(obs_batch, last_action_batch),
#                     self.action_noise,
#                 )
#                 self.train_pvm.add(action)

#                 next_obs, reward, done, info = self.train_env.step(action)

#                 exp = (obs, last_action, info["price_variation"], info["trf_mu"])
#                 self.train_buffer.append(exp)

#                 if len(self.train_buffer) == self.train_batch_size:
#                     self._gradient_ascent()

#                 obs = next_obs

#             # episode 末尾剩余数据
#             self._gradient_ascent()

#             # 如果传了 validation_env，则按 demo 风格在每个 episode 末尾跑一次验证
#             if self.validation_env is not None:
#                 self.test(
#                     self.validation_env,
#                     policy=None,
#                     online_training_period=10**18,
#                     batch_size=self.train_batch_size,
#                     lr=0.0,
#                     optimizer=None,
#                 )

#     def _setup_test(self, env, policy, batch_size, lr, optimizer):
#         """Initializes algorithm before testing."""
#         self.test_env = env

#         policy = self.train_policy if policy is None else policy
#         lr = self.lr if lr is None else lr
#         optimizer = self.optimizer if optimizer is None else optimizer

#         self.test_policy = copy.deepcopy(policy).to(self.device)
#         self.test_optimizer = optimizer(self.test_policy.parameters(), lr=lr)

#         self.test_buffer = ReplayBuffer(capacity=batch_size)
#         self.test_pvm = PVM(self.test_env.episode_length, env.portfolio_size)

#         dataset = RLDataset(self.test_buffer)
#         self.test_dataloader = DataLoader(
#             dataset=dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             pin_memory=True,
#         )

#     def test(
#         self,
#         env,
#         policy=None,
#         online_training_period=10,
#         batch_size=None,
#         lr=None,
#         optimizer=None,
#     ):
#         """Tests the policy with optional online learning.

#         Args:
#           env: Environment to be used in testing.
#           policy: Policy architecture to be used. If None, it will use the training
#             architecture.
#           online_training_period: Period in which an online training will occur. To
#             disable online learning, use a very big value.
#           batch_size: Batch size to train neural network. If None, it will use the
#             training batch size.
#           lr: Policy neural network learning rate. If None, it will use the training
#             learning rate.
#           optimizer: Optimizer of neural network. If None, it will use the training
#             optimizer.
#         """
#         batch_size = self.train_batch_size if batch_size is None else batch_size
#         self._setup_test(env, policy, batch_size, lr, optimizer)

#         obs = self.test_env.reset()
#         self.test_pvm.reset()
#         done = False
#         steps = 0

#         while not done:
#             steps += 1

#             last_action = self.test_pvm.retrieve()
#             obs_batch = np.expand_dims(obs, axis=0)
#             last_action_batch = np.expand_dims(last_action, axis=0)
#             action = self.test_policy(obs_batch, last_action_batch)
#             self.test_pvm.add(action)

#             next_obs, reward, done, info = self.test_env.step(action)

#             exp = (obs, last_action, info["price_variation"], info["trf_mu"])
#             self.test_buffer.append(exp)

#             # lr <= 0 时直接跳过 online learning
#             if lr is not None and lr <= 0:
#                 pass
#             else:
#                 if steps % online_training_period == 0:
#                     self._gradient_ascent(test=True)

#             obs = next_obs

#     def _gradient_ascent(self, test=False):
#         """Performs the gradient ascent step in the policy gradient algorithm.

#         改动说明：
#         1) 保留原始 log-growth 主目标；
#         2) 加 entropy regularization，防止权重过于 one-hot；
#         3) 加 turnover regularization，惩罚和上一步权重差太大；
#         4) 对 log 输入做 clamp，避免数值不稳定；
#         5) 加梯度裁剪，避免训练后期越来越激进。
#         """
#         if test:
#             if len(self.test_buffer) == 0:
#                 return None
#         else:
#             if len(self.train_buffer) == 0:
#                 return None

#         obs, last_actions, price_variations, trf_mu = (
#             next(iter(self.test_dataloader))
#             if test
#             else next(iter(self.train_dataloader))
#         )

#         obs = obs.to(self.device)
#         last_actions = last_actions.to(self.device)
#         price_variations = price_variations.to(self.device)
#         trf_mu = trf_mu.unsqueeze(1).to(self.device)

#         # 当前策略输出的目标权重
#         mu = (
#             self.test_policy.mu(obs, last_actions)
#             if test
#             else self.train_policy.mu(obs, last_actions)
#         )

#         # -----------------------------
#         # 1) 原始 growth 项：安全版 log
#         # -----------------------------
#         portfolio_growth = torch.sum(mu * price_variations * trf_mu, dim=1)
#         portfolio_growth = torch.clamp(portfolio_growth, min=1e-8)
#         growth_loss = -torch.mean(torch.log(portfolio_growth))

#         # -----------------------------
#         # 3) 调仓平滑正则：惩罚和上一期权重差太大
#         #    注意：这里用的是目标权重 mu 与 last_actions 的差
#         # -----------------------------
#         turnover = torch.sum(torch.abs(mu - last_actions), dim=1).mean()
#         turnover_coef = 5e-3

#         # -----------------------------
#         # 4) 总 loss
#         #    最小化 growth_loss；
#         #    同时鼓励更高熵、减少过猛调仓
#         # -----------------------------
#         policy_loss = (
#             growth_loss
#           + turnover_coef * turnover
#         )

#         if test:
#             self.test_optimizer.zero_grad()
#             policy_loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.test_policy.parameters(), max_norm=1.0)
#             self.test_optimizer.step()
#         else:
#             self.train_optimizer.zero_grad()
#             policy_loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.train_policy.parameters(), max_norm=1.0)
#             self.train_optimizer.step()

#         return policy_loss.item()



from __future__ import annotations

import copy

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .architectures_changed import EIIE
from .utils import apply_portfolio_noise
from .utils import PVM
from .utils import ReplayBuffer
from .utils import RLDataset


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
        dsr_coef=0.05,
        mdd_coef=0.10,
        dsr_eta=1 / 252,
        reward_eps=1e-8,
        mdd_use_max=True,
        grad_clip=1.0,
        debug_reward=False,
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
          dsr_coef: Coefficient of DSR loss term.
          mdd_coef: Coefficient of MDD loss term.
          dsr_eta: Adaptation rate in DSR recursion. Common choice: 1/252.
          reward_eps: Small epsilon for numerical stability.
          mdd_use_max: If True, use max drawdown in current batch;
                       otherwise use mean drawdown in current batch.
          grad_clip: Gradient clipping max norm.
          debug_reward: Whether to print reward component values.
        """
        self.policy = policy
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.validation_env = validation_env
        self.batch_size = batch_size
        self.lr = lr
        self.action_noise = action_noise
        self.optimizer = optimizer
        self.device = device

        # reward-related hyperparameters
        self.dsr_coef = dsr_coef
        self.mdd_coef = mdd_coef
        self.dsr_eta = dsr_eta
        self.reward_eps = reward_eps
        self.mdd_use_max = mdd_use_max
        self.grad_clip = grad_clip
        self.debug_reward = debug_reward

        # path-dependent reward states
        self._reset_train_reward_state()
        self._reset_test_reward_state()

        self._setup_train(env, self.policy, self.batch_size, self.lr, self.optimizer)

    def _reset_train_reward_state(self):
        """Reset train-side path-dependent reward state."""
        self.train_reward_state = {
            "X": 0.0,              # EMA of first moment for DSR
            "Y": 0.0,              # EMA of second moment for DSR
            "portfolio_value": 1.0,
            "peak_value": 1.0,
        }

    def _reset_test_reward_state(self):
        """Reset test-side path-dependent reward state."""
        self.test_reward_state = {
            "X": 0.0,
            "Y": 0.0,
            "portfolio_value": 1.0,
            "peak_value": 1.0,
        }

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
            shuffle=False,   # IMPORTANT: DSR / MDD are path-dependent
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
            self._reset_train_reward_state()
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

            # 如果传了 validation_env，则在每个 episode 末尾跑一次验证
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
            shuffle=False,   # IMPORTANT: DSR / MDD are path-dependent
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
        self._reset_test_reward_state()
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

    def _compute_dsr_loss(self, log_returns, state):
        """Compute DSR-based loss over an ordered batch.

        DSR recursion:
            X_t = X_{t-1} + eta * (r_t - X_{t-1})
            Y_t = Y_{t-1} + eta * (r_t^2 - Y_{t-1})
            DSR_t = [Y_{t-1} * ΔX_t - 0.5 * X_{t-1} * ΔY_t] / (Y_{t-1} - X_{t-1}^2)^(3/2)

        We minimize negative mean DSR to maximize risk-adjusted return.
        """
        X_prev = log_returns.new_tensor(float(state["X"]))
        Y_prev = log_returns.new_tensor(float(state["Y"]))

        dsr_values = []

        for r_t in log_returns:
            delta_x = r_t - X_prev
            delta_y = r_t * r_t - Y_prev

            variance_proxy = torch.clamp(
                Y_prev - X_prev * X_prev,
                min=self.reward_eps,
            )
            raw_dsr = (
                Y_prev * delta_x - 0.5 * X_prev * delta_y
            ) / torch.pow(variance_proxy, 1.5)

            # clip to avoid huge instability in early recursion
            dsr_t = torch.clamp(raw_dsr, min=-10.0, max=10.0)
            dsr_values.append(dsr_t)

            X_prev = X_prev + self.dsr_eta * delta_x
            Y_prev = Y_prev + self.dsr_eta * delta_y

        dsr_values = torch.stack(dsr_values)
        dsr_loss = -torch.mean(dsr_values)  # minimize negative DSR => maximize DSR

        new_state = {
            "X": float(X_prev.detach().item()),
            "Y": float(Y_prev.detach().item()),
        }
        return dsr_loss, dsr_values.detach(), new_state

    def _compute_mdd_loss(self, portfolio_growth, state):
        """Compute MDD-based penalty over an ordered batch.

        We maintain running portfolio value and running peak value across batches.
        MDD penalty is either:
            - max drawdown within current batch (default), or
            - mean drawdown within current batch.
        """
        current_value = portfolio_growth.new_tensor(float(state["portfolio_value"]))
        peak_value = portfolio_growth.new_tensor(float(state["peak_value"]))

        drawdowns = []

        for g_t in portfolio_growth:
            current_value = current_value * g_t
            peak_value = torch.maximum(peak_value, current_value)

            dd_t = (peak_value - current_value) / torch.clamp(
                peak_value, min=self.reward_eps
            )
            dd_t = torch.clamp(dd_t, min=0.0, max=1.0)
            drawdowns.append(dd_t)

        drawdowns = torch.stack(drawdowns)

        if self.mdd_use_max:
            mdd_loss = torch.max(drawdowns)
        else:
            mdd_loss = torch.mean(drawdowns)

        new_state = {
            "portfolio_value": float(current_value.detach().item()),
            "peak_value": float(peak_value.detach().item()),
        }
        return mdd_loss, drawdowns.detach(), new_state

    def _gradient_ascent(self, test=False):
        """Performs the gradient ascent step in the policy gradient algorithm.

        当前版本的总 loss:
            policy_loss = growth_loss + dsr_coef * dsr_loss + mdd_coef * mdd_loss

        其中：
            growth_loss = -mean(log(portfolio_growth))
            dsr_loss    = -mean(DSR_t)
            mdd_loss    = max_drawdown_in_batch (or mean_drawdown_in_batch)
        """
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

        # -------------------------------------------------
        # 1) growth term
        # -------------------------------------------------
        portfolio_growth = torch.sum(mu * price_variations * trf_mu, dim=1)
        portfolio_growth = torch.clamp(portfolio_growth, min=self.reward_eps)

        log_returns = torch.log(portfolio_growth)
        growth_loss = -torch.mean(log_returns)

        # choose correct path-dependent state
        reward_state = self.test_reward_state if test else self.train_reward_state

        # -------------------------------------------------
        # 2) DSR term
        # -------------------------------------------------
        dsr_loss, dsr_values, dsr_state_update = self._compute_dsr_loss(
            log_returns,
            reward_state,
        )

        # -------------------------------------------------
        # 3) MDD term
        # -------------------------------------------------
        mdd_loss, drawdowns, mdd_state_update = self._compute_mdd_loss(
            portfolio_growth,
            reward_state,
        )

        # -------------------------------------------------
        # 4) total loss
        # -------------------------------------------------
        policy_loss = (
            growth_loss
            + self.dsr_coef * dsr_loss
            + self.mdd_coef * mdd_loss
        )

        # optimizer step
        if test:
            self.test_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.test_policy.parameters(), max_norm=self.grad_clip)
            self.test_optimizer.step()

            # update reward state after backward
            self.test_reward_state.update(dsr_state_update)
            self.test_reward_state.update(mdd_state_update)

        else:
            self.train_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.train_policy.parameters(), max_norm=self.grad_clip)
            self.train_optimizer.step()

            # update reward state after backward
            self.train_reward_state.update(dsr_state_update)
            self.train_reward_state.update(mdd_state_update)

        if self.debug_reward:
            dsr_mean = float(dsr_values.mean().item())
            mdd_batch = float(mdd_loss.item())
            print(
                f"[Reward Debug] "
                f"growth_loss={growth_loss.item():.6f}, "
                f"dsr_loss={dsr_loss.item():.6f}, "
                f"mdd_loss={mdd_loss.item():.6f}, "
                f"mean_dsr={dsr_mean:.6f}, "
                f"batch_mdd={mdd_batch:.6f}, "
                f"total_loss={policy_loss.item():.6f}"
            )

        return policy_loss.item()