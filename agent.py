import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np
import math
from torch.distributions import Normal

# Q-Network Definition
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        # Concatenate state and action, then produce a single Q-value
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Policy Network Definition
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, action_dim)
        self.log_std_linear = nn.Linear(256, action_dim)
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        mu = torch.tanh(self.mu(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)

        return action, log_prob

class CQLAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 alpha=0.2,
                 cql_weight=5.0,
                 num_random_actions=10):

        # Detect GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # CQL penalty coefficient
        self.alpha = alpha
        self.cql_weight = cql_weight

        # Number of random actions to sample for the conservative penalty
        self.num_random_actions = num_random_actions

        # Initialize networks
        self.q1 = QNetwork(state_dim, action_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)

        # Initialize optimizers
        self.q_optim = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr / 10)  # Lower policy LR (heuristic)

        # Logging
        self.q_loss = 0
        self.policy_loss = 0

    def get_action(self, state, deterministic=False):
        """Select action from current policy; add a bit of noise if not deterministic."""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_pi = self.policy(state)

            if not deterministic:
                noise = 0.1 * torch.randn_like(action).to(self.device)
                action = torch.clamp(action + noise, -1.0, 1.0)

            return action.squeeze(0).cpu().numpy()

    def _compute_policy_values(self, obs_pi, obs_q):
        #with torch.no_grad():
        actions_pred, log_pis = self.policy.evaluate(obs_pi)

        qs1 = self.q1(obs_q, actions_pred)
        qs2 = self.q2(obs_q, actions_pred)

        return qs1 - log_pis.detach(), qs2 - log_pis.detach()

    def _compute_random_values(self, obs, actions, critic):
        random_values = critic(obs, actions)
        random_log_probs = math.log(0.5 ** self.action_dim)
        return random_values - random_log_probs

    def get_q_loss(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, new_log_pi = self.policy(next_states)
            next_q1 = self.q1_target(next_states, next_actions)
            next_q2 = self.q2_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2).squeeze(-1)  # Ensure proper shape
            target_q = rewards + (1 - dones) * self.gamma * next_q.unsqueeze(-1)  # Match q1/q2 shape

        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        bellman_error = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # random_actions = torch.FloatTensor(q1.shape[0] * 10, actions.shape[-1]).uniform_(-1, 1).to(self.device)
        # num_repeat = int (random_actions.shape[0] / states.shape[0])
        # temp_states = states.unsqueeze(1).repeat(1, num_repeat, 1).view(states.shape[0] * num_repeat, states.shape[1])
        # temp_next_states = next_states.unsqueeze(1).repeat(1, num_repeat, 1).view(next_states.shape[0] * num_repeat, next_states.shape[1])

        # current_pi_values1, current_pi_values2  = self._compute_policy_values(temp_states, temp_states)
        # next_pi_values1, next_pi_values2 = self._compute_policy_values(temp_next_states, temp_states)

        # random_values1 = self._compute_random_values(temp_states, random_actions, self.q1).reshape(states.shape[0], num_repeat, 1)
        # random_values2 = self._compute_random_values(temp_states, random_actions, self.q2).reshape(states.shape[0], num_repeat, 1)

        # current_pi_values1 = current_pi_values1.reshape(states.shape[0], num_repeat, 1)
        # current_pi_values2 = current_pi_values2.reshape(states.shape[0], num_repeat, 1)

        # next_pi_values1 = next_pi_values1.reshape(states.shape[0], num_repeat, 1)
        # next_pi_values2 = next_pi_values2.reshape(states.shape[0], num_repeat, 1)

        # cat_q1 = torch.cat([random_values1, current_pi_values1, next_pi_values1], 1)
        # cat_q2 = torch.cat([random_values2, current_pi_values2, next_pi_values2], 1)

        # cql1_scaled_loss = ((torch.logsumexp(cat_q1, dim=1).mean() * self.cql_weight) - q1.mean()) * self.cql_weight
        # cql2_scaled_loss = ((torch.logsumexp(cat_q2, dim=1).mean() * self.cql_weight) - q2.mean()) * self.cql_weight
        # cql_loss = cql1_scaled_loss + cql2_scaled_loss

        q_loss = bellman_error # + cql_loss
        return q_loss

    def get_policy_loss(self, states):
        """
        Simple policy loss for CQL:
          maximize Q(s, pi(s)) => minimize -Q(s, pi(s))
        """
        actions, log_pi = self.policy(states)
        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        q_values = torch.min(q1, q2)
        policy_loss = (self.alpha * log_pi - q_values).mean()
        return policy_loss

    def update(self, states, actions, rewards, next_states, dones):
        # ---- Update Q-function ----
        self.q_optim.zero_grad()
        q_loss = self.get_q_loss(states, actions, rewards, next_states, dones)
        q_loss.backward()
        self.q_optim.step()
        self.q_loss = q_loss.item()

        # ---- Update Policy ----
        self.policy_optim.zero_grad()
        policy_loss = self.get_policy_loss(states)
        policy_loss.backward()
        self.policy_optim.step()
        self.policy_loss = policy_loss.item()

        # ---- Soft-update target networks ----
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
