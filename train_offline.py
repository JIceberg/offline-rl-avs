import pickle
from replay_buffer import ReplayBuffer
from agent import CQLAgent
from offline_gym import OfflineRL

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

NUM_EPISODES = 100
NUM_STEPS = 15000
NUM_TRAJS = 10
NUM_TRAJ_STEPS = 110
BATCH_SIZE = 64

agent = CQLAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0]
)

for episode in range(NUM_EPISODES):
    # train
    for step in range(NUM_STEPS):
        # Sample a batch of data from the replay buffer
        batch = replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = batch

        # Update the agent
        agent.update(states, actions, rewards, next_states, dones)

        print(f"Episode: {episode+1}/{NUM_EPISODES}, Step: {step+1}/{NUM_STEPS}, Q Loss: {agent.q_loss:.4f}, Policy Loss: {agent.policy_loss:.4f}")

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
    
    print(f"Episode: {episode+1}/{NUM_EPISODES}, Average Reward: {avg_reward:.4f}, Crashes: {num_crashes}/{NUM_TRAJS}, Average Length: {avg_length:.2f}")