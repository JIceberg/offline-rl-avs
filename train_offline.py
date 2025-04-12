import pickle
from replay_buffer import ReplayBuffer
from agent import CQLAgent
from offline_gym import OfflineRL

with open('offline_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

offline_data = {
    'observations': data['observations'],
    'next_observations': data['next_observations'],
    'actions': data['actions'],
    'rewards': data['rewards'],
    'terminals': data['terminals']
}

env = OfflineRL()

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

NUM_EPISODES = 100
NUM_STEPS = 15000
NUM_TRAJS = 5
BATCH_SIZE = 64

agent = CQLAgent(
    state_dim=offline_data['observations'][0].shape[0],
    action_dim=offline_data['actions'][0].shape[0]
)

for episode in range(NUM_EPISODES):
    # train
    for step in range(NUM_STEPS):
        # Sample a batch of data from the replay buffer
        batch = replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = batch

        # Update the agent
        agent.update(states, actions, rewards, next_states, dones)

        print(f"Episode: {episode}/{NUM_EPISODES}, Step: {step}/{NUM_STEPS}, Q Loss: {agent.q_loss}, Policy Loss: {agent.policy_loss}")

    # eval
    traj_reward = []
    for traj in range(NUM_TRAJS):
        # Reset the environment
        state, _ = env.reset()

        for step in range(NUM_STEPS):
            # Select action from the agent
            action = agent.get_action(state, deterministic=True)

            # Step the environment
            next_state, reward, done, _ = env.step(action)
            traj_reward.append(reward)

            # Update state
            state = next_state

            if done:
                break
    avg_reward = sum(traj_reward) / len(traj_reward)
    
    print(f"Episode: {episode}/{NUM_EPISODES}, Reward: {avg_reward}")