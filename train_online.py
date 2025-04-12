import pickle
from offline_gym import OfflineRL
from agent import CQLAgent
from replay_buffer import ReplayBuffer

# Initialize environment, model, and replay buffer
env = OfflineRL()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print(state_dim, action_dim)
agent = CQLAgent(state_dim, action_dim)
replay_buffer = ReplayBuffer()

# Training parameters
num_episodes = 10000
max_steps_per_episode = 300
BATCH_SIZE = 32

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    for step in range(max_steps_per_episode):
        # Select action using the model
        action = agent.get_action(state)
        
        # Take action in the environment
        next_state, reward, done, collision = env.step(action)
        total_reward += reward

        # Store transition in replay buffer
        replay_buffer.add(state, action, reward, next_state, done)
        
        # Update state
        state = next_state

        # Update the agent if the replay buffer has enough samples
        if len(replay_buffer) >= BATCH_SIZE:
            batch = replay_buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = batch
            agent.update(states, actions, rewards, next_states, dones)
                
        # end if done or if we collided
        if done or collision:
            print(f"Episode {episode + 1}/{num_episodes} completed in {step + 1} steps with total reward {total_reward}")
            break
    
print(len(replay_buffer))

# Save the replay buffer to a file
with open("expert_dataset.pkl", "wb") as f:
    pickle.dump(replay_buffer, f)

print("Training completed and replay buffer saved to expert_dataset.pkl.")