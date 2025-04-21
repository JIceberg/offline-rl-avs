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
        self.log_std_linear = nn.Linear(hidden_size, action_dim)
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
        log_prob = (dist.log_prob(e) - \
                    torch.log((1 - action.pow(2)) + epsilon))
        # log_prob = (dist.log_prob(e) - torch.log(torch.clamp(1 - action.pow(2), min=epsilon))).sum(1, keepdim=True)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

class CQLAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 q_lr=1e-4,
                 policy_lr=1e-4,
                 alpha_multiplier=1.0,
                 gamma=0.99,
                 tau=5e-3,
                 cql_weight=1.0,
                 importance_sampling=True,
                 num_random_actions=10,
                 temperature=1.0
                 ):

        # Detect GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # CQL multipliers
        self.cql_weight = cql_weight
        self.alpha_multiplier = alpha_multiplier
        self.temp = temperature
        self.log_alpha = torch.zeros(1, requires_grad=True)

        # Number of random actions to sample for the conservative penalty
        self.num_random_actions = num_random_actions
        
        self.target_entropy = -action_dim
        self.importance_sampling = importance_sampling

        # Initialize networks
        self.q1 = QNetwork(state_dim, action_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)

        # Initialize optimizers
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=q_lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=q_lr)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=policy_lr)
        # self.cql_alpha_optim = optim.Adam(self.log_cql_alpha.parameters(), lr=lr)

        # Logging
        self.q_loss = 0
        self.policy_loss = 0

    def save(self, filepath):
        torch.save({
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'policy': self.policy.state_dict(),
            'q1_optim': self.q1_optim.state_dict(),
            'q2_optim': self.q2_optim.state_dict(),
            'policy_optim': self.policy_optim.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.q1_optim.load_state_dict(checkpoint['q1_optim'])
        self.q2_optim.load_state_dict(checkpoint['q2_optim'])
        self.policy_optim.load_state_dict(checkpoint['policy_optim'])

    def get_action(self, state, deterministic=False):
        """Select action from current policy; add a bit of noise if not deterministic."""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_pi = self.policy.evaluate(state)

            if not deterministic:
                noise = 0.1 * torch.randn_like(action).to(self.device)
                action = torch.clamp(action + noise, -1.0, 1.0)

            return action.squeeze(0).cpu().numpy()

    def get_q_loss(self, states, actions, rewards, next_states, dones):
        batch_size = states.shape[0]
        with torch.no_grad():
            next_actions, new_log_pi = self.policy.evaluate(next_states)
            next_q1 = self.q1_target(next_states, next_actions)
            next_q2 = self.q2_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        # CQL Loss computation

        random_actions = torch.FloatTensor(batch_size * self.num_random_actions, self.action_dim).uniform_(-1, 1)
        states_repeat = states.unsqueeze(1).repeat(1, self.num_random_actions, 1).view(-1, states.shape[-1])
        next_states_repeat = next_states.unsqueeze(1).repeat(1, self.num_random_actions, 1).view(-1, states.shape[-1])
        curr_actions, curr_log_pis = self.policy.evaluate(states_repeat)
        next_actions, next_log_pis = self.policy.evaluate(next_states_repeat)
        q1_rand = self.q1(states_repeat, random_actions).view(batch_size, self.num_random_actions, 1)
        q2_rand = self.q2(states_repeat, random_actions).view(batch_size, self.num_random_actions, 1)
        q1_current = self.q1(states_repeat, curr_actions).view(batch_size, self.num_random_actions, 1)
        q2_current = self.q2(states_repeat, curr_actions).view(batch_size, self.num_random_actions, 1)
        q1_next = self.q1(next_states_repeat, next_actions).view(batch_size, self.num_random_actions, 1)
        q2_next = self.q2(next_states_repeat, next_actions).view(batch_size, self.num_random_actions, 1)
        curr_log_pis = curr_log_pis.view(batch_size, self.num_random_actions, 1)
        next_log_pis = next_log_pis.view(batch_size, self.num_random_actions, 1)

        # random_density = np.log(0.5 ** current_actions.shape[-1])
        cat_q1 = torch.cat(
            [q1_rand, q1_pred.unsqueeze(1), q1_next, q1_current], 1
        )
        cat_q2 = torch.cat(
            [q2_rand, q2_pred.unsqueeze(1), q2_next, q2_current], 1
        )

        if self.importance_sampling:
            random_density = np.log(0.5 ** curr_actions.shape[-1])
            cat_q1 = torch.cat(
                [q1_rand - random_density,
                 q1_next - next_log_pis.detach(),
                 q1_current - curr_log_pis.detach()], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density,
                 q2_next - next_log_pis.detach(),
                 q2_current - curr_log_pis.detach()], 1
            )

        if self.temp != 0:
            min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1,).mean() * self.temp
            min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1,).mean() * self.temp
        else:
            min_qf1_loss = 0
            min_qf2_loss = 0
                    
        """Subtract the log likelihood of data"""
        cql_q1_loss = min_qf1_loss - q1_pred.mean()
        cql_q2_loss = min_qf2_loss - q2_pred.mean()

        q1_loss += cql_q1_loss * self.cql_weight
        q2_loss += cql_q2_loss * self.cql_weight

        return q1_loss, q2_loss

    def get_policy_loss(self, states):
        """
        Simple policy loss for CQL:
          maximize Q(s, pi(s)) => minimize -Q(s, pi(s))
        """
        actions, log_pi = self.policy.evaluate(states)
        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        q_values = torch.min(q1, q2)
        alpha = self.alpha_multiplier * self.log_alpha.exp() 
        policy_loss = (alpha * log_pi - q_values).mean()
        return policy_loss

    def update(self, states, actions, rewards, next_states, dones):
        # ---- Update Q-function ----
        q1_loss, q2_loss = self.get_q_loss(states, actions, rewards, next_states, dones)
        self.q1_optim.zero_grad()
        q1_loss.backward(retain_graph=True)
        self.q1_optim.step()
        self.q2_optim.zero_grad()
        q2_loss.backward(retain_graph=True)
        self.q2_optim.step()
        self.q_loss = q1_loss.item() + q2_loss.item()

        # ---- Update Alpha ----
        _, log_pi = self.policy.evaluate(states)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # ---- Update Policy ----
        policy_loss = self.get_policy_loss(states)
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        self.policy_loss = policy_loss.item()

        # ---- Soft-update target networks ----
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)