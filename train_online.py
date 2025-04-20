# %%
import pickle
from offline_gym import OfflineRL
from agent import CQLAgent
from replay_buffer import ReplayBuffer
import numpy as np
import matplotlib.pyplot as plt

# Initialize environment, model, and replay buffer
env = OfflineRL()
env.reset()
env.render()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print(state_dim, action_dim)
agent = CQLAgent(state_dim, action_dim, cql_weight=0, alpha_multiplier=0)
replay_buffer = ReplayBuffer()

# Training parameters
num_episodes = 1000
max_steps_per_episode = 110
BATCH_SIZE = 64
NUM_EVAL_TRAJS = 10

# Initialize tracking variables
episode_rewards = []
q_losses = []
policy_losses = []

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    episode_q_loss = []
    episode_policy_loss = []
    for step in range(max_steps_per_episode):
        # Select action using the model
        action = agent.get_action(state)
        
        # Take action in the environment
        next_state, reward, done, collision = env.step(action)
        total_reward += reward

        # Store transition in replay buffer
        replay_buffer.add(state, action, [reward], next_state, [done])
        
        # Update state
        state = next_state

        # Update the agent if the replay buffer has enough samples
        if len(replay_buffer) >= BATCH_SIZE:
            batch = replay_buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = batch
            agent.update(states, actions, rewards, next_states, dones)
            print(f"Episode: {episode+1}/{num_episodes}, Step: {step+1}/{max_steps_per_episode}, Q Loss: {agent.q_loss:.4f}, Policy Loss: {agent.policy_loss:.4f}")
            episode_q_loss.append(agent.q_loss)
            episode_policy_loss.append(agent.policy_loss)
                
        # end if done or if we collided
        if done or collision:
            # print(f"Episode {episode + 1}/{num_episodes} completed in {step + 1} steps with total reward {total_reward}")
            break
    
    q_losses.append(np.mean(episode_q_loss))
    policy_losses.append(np.mean(episode_policy_loss))
    episode_rewards.append(total_reward)

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

# evaluate
traj_reward = []
len_traj = []
num_crashes = 0
for traj in range(NUM_EVAL_TRAJS):
    # Reset the environment
    state, _ = env.reset()
    total_reward = 0

    for step in range(max_steps_per_episode):
        # Select action from the agent
        action = agent.get_action(state, deterministic=True)

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

avg_reward = sum(traj_reward) / len(traj_reward)
avg_length = sum(len_traj) / len(len_traj)
print(f"Average Reward: {avg_reward:.4f}, Crashes: {num_crashes}/{NUM_EVAL_TRAJS}, Average Length: {avg_length:.2f}")
    
print(len(replay_buffer))

replay_buffer.save_to_file("expert_dataset.pkl")

print("Training completed and replay buffer saved to expert_dataset.pkl.")

# %%
