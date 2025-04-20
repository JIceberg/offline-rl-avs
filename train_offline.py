# %%
import pickle
from replay_buffer import ReplayBuffer
from agent import CQLAgent
from offline_gym import OfflineRL
import numpy as np
import matplotlib.pyplot as plt
from offline_gym import record_gif

with open('offline_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

offline_data = {
    'observations': data['observations'],
    'next_observations': data['next_states'],
    'actions': data['actions'],
    'rewards': data['rewards'],
    'terminals': data['dones']
}

env = OfflineRL()
env.reset()

# Assuming ReplayBuffer is already defined and initialized
replay_buffer = ReplayBuffer()

for obs, next_obs, action, reward, terminal in zip(
    offline_data['observations'],
    offline_data['next_observations'],
    offline_data['actions'],
    offline_data['rewards'],
    offline_data['terminals']
):
    replay_buffer.add(obs, action, reward, next_obs, terminal)

# %%
NUM_EPISODES = 3
NUM_STEPS = 15000
NUM_TRAJS = 10
NUM_TRAJ_STEPS = 110
BATCH_SIZE = 64

cql_weight = 1.0
alpha_multiplier = 1.0
agent = CQLAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    cql_weight=cql_weight,
    alpha_multiplier=alpha_multiplier,
    temperature=1.0,
    lr=1e-4
)

# Initialize tracking variables
episode_rewards = []
q_losses = []
policy_losses = []

for episode in range(NUM_EPISODES):
    # train
    episode_q_loss = []
    episode_policy_loss = []
    for step in range(NUM_STEPS):
        # Sample a batch of data from the replay buffer
        batch = replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = batch

        # Update the agent
        agent.update(states, actions, rewards, next_states, dones)

        print(f"Episode: {episode+1}/{NUM_EPISODES}, Step: {step+1}/{NUM_STEPS}, Q Loss: {agent.q_loss:.4f}, Policy Loss: {agent.policy_loss:.4f}")
        episode_q_loss.append(agent.q_loss)
        episode_policy_loss.append(agent.policy_loss)
    # eval
    traj_reward = []
    len_traj = []
    num_crashes = 0
    for traj in range(NUM_TRAJS):
        # Reset the environment
        state, _ = env.reset()
        total_reward = 0

        for step in range(NUM_TRAJ_STEPS):
            # Select action from the agent
            action = agent.get_action(state, deterministic=True)
            # print(action)

            # Step the environment
            next_state, reward, done, collision = env.step(action)
            total_reward += reward

            # Update state
            state = next_state

            if done:
                if collision:
                    num_crashes += 1
                len_traj.append(step)
                traj_reward.append(total_reward)
                break
        
        # print(f"Trajectory: {traj+1}/{NUM_TRAJS}, Reward: {total_reward:.4f}, Steps: {step}/{NUM_TRAJ_STEPS}")

    avg_reward = sum(traj_reward) / len(traj_reward)
    avg_length = sum(len_traj) / len(len_traj)

    q_losses.append(np.mean(episode_q_loss))
    policy_losses.append(np.mean(episode_policy_loss))
    episode_rewards.append(avg_reward)
<<<<<<< Updated upstream
    
    print(f"Episode: {episode+1}/{NUM_EPISODES}, Average Reward: {avg_reward:.4f}, Crashes: {num_crashes}/{NUM_TRAJS}, Average Length: {avg_length:.2f}")

# Plot the losses
fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].plot(q_losses)
ax[0].set_xlabel('Episodes')
ax[0].set_title('Q-Loss')
ax[1].plot(policy_losses)
ax[1].set_xlabel('Episodes')
ax[1].set_title('Policy Loss')
ax[2].plot(episode_rewards)
ax[2].set_xlabel('Episodes')
ax[2].set_title('Rewards')
if cql_weight == 0.0 and alpha_multiplier == 0.0:
    fig.suptitle('QL')
    fig.savefig('ql_offline.jpg')
else:
    fig.suptitle('CQL')
    fig.savefig('cql_offline.jpg')

results = {}
results['q_losses'] = q_losses
results['policy_losses'] = policy_losses
results['rewards'] = episode_rewards
if cql_weight == 0.0 and alpha_multiplier == 0.0:
    with open('ql_results_offline.pkl', 'wb') as f:
        pickle.dump(results,f)
else:
    with open('cql_results_offline.pkl', 'wb') as f:
        pickle.dump(results,f)
=======

    ql_q_losses.append(np.mean(ql_episode_q_loss))
    ql_policy_losses.append(np.mean(ql_episode_policy_loss))
    ql_episode_rewards.append(ql_avg_reward)

    print(f"Episode: {episode+1}/{NUM_EPISODES}, Agent Avg Reward: {avg_reward:.4f}, Crashes: {num_crashes}/{NUM_TRAJS}, Avg Length: {avg_length:.2f}")
    print(f"Episode: {episode+1}/{NUM_EPISODES}, QL Agent Avg Reward: {ql_avg_reward:.4f}, Crashes: {ql_num_crashes}/{NUM_TRAJS}, Avg Length: {ql_avg_length:.2f}")

# Plot the losses and rewards for both agents
fig, ax = plt.subplots(2, 3, figsize=(18, 10))
ax[0, 0].plot(q_losses, label='CQL Agent')
ax[0, 0].plot(ql_q_losses, label='QL Agent')
ax[0, 0].set_xlabel('Episodes')
ax[0, 0].set_title('Q-Loss')
ax[0, 0].legend()

ax[0, 1].plot(policy_losses, label='CQL Agent')
ax[0, 1].plot(ql_policy_losses, label='QL Agent')
ax[0, 1].set_xlabel('Episodes')
ax[0, 1].set_title('Policy Loss')
ax[0, 1].legend()

ax[0, 2].plot(episode_rewards, label='CQL Agent')
ax[0, 2].plot(ql_episode_rewards, label='QL Agent')
ax[0, 2].set_xlabel('Episodes')
ax[0, 2].set_title('Rewards')
ax[0, 2].legend()

# Save the plots
fig.suptitle('CQL vs QL Comparison')
fig.savefig('cql_vs_ql_comparison.jpg')

#%%
# make the simulation video 
record_gif(
    env,
    agent,
    'cql_agent_test.gif',
)

record_gif(
    env,
    ql_agent,
    'ql_agent_test.gif',
)
# Save results for both agents
results = {
    'cql': {
        'q_losses': q_losses,
        'policy_losses': policy_losses,
        'rewards': episode_rewards
    },
    'ql': {
        'q_losses': ql_q_losses,
        'policy_losses': ql_policy_losses,
        'rewards': ql_episode_rewards
    }
}
with open('cql_vs_ql_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# save the agent weights
agent.save('cql_agent_weights_test.pth')
ql_agent.save('ql_agent_weights_test.pth')

# %%
cql_weight = 1.0
alpha_multiplier = 1.0
agent = CQLAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    cql_weight=cql_weight,
    alpha_multiplier=alpha_multiplier,
    temperature=1.0,
    lr=1e-4
)

print(agent.state_dim, agent.action_dim)

ql_agent = CQLAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    cql_weight=0.0,
    alpha_multiplier=0.0,
    temperature=1.0,
    lr=1e-4
)

agent.load('cql_agent_weights_test.pth')
ql_agent.load('ql_agent_weights_test.pth')

record_gif(
    env,
    agent,
    'cql_agent_test.gif',
)

record_gif(
    env,
    ql_agent,
    'ql_agent_test.gif',
)
# %%
>>>>>>> Stashed changes
