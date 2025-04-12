import math
import numpy as np
import pickle
from pathlib import Path
from offline_gym import get_reward

argoverse_scenario_dir = Path(
        'data_for_simulator/train/')
all_scenario_files = sorted(argoverse_scenario_dir.rglob("*.pkl"))
scenario_file_lists = (all_scenario_files[:1000])
scenarios = []
for scenario_file_list in scenario_file_lists:
    scenario = pickle.load(open(scenario_file_list, 'rb'))
    scenarios.append(scenario)
print(len(scenarios))

observations = []
next_observations = []
actions = []
rewards = []
terminals = []
ego_v_state = []

for scenario in scenarios:
    states_of_scenario = scenario['states']
    for i in range(len(states_of_scenario)):
        state = states_of_scenario[i]
        action = 0.0

        ego = state[0]
        object_front = state[1]
        object_behind = state[2]
        object_left_front = state[3]
        object_right_front = state[4]
        object_left_behind = state[5]
        object_right_behind = state[6]
        done = 0
        reach = 0

        ego_v = math.sqrt(state[0][3] ** 2 + state[0][4] ** 2)
        object_front_v = math.sqrt(state[1][5] ** 2 + state[1][6] ** 2)
        object_behind_v = math.sqrt(state[2][5] ** 2 + state[2][6] ** 2)
        object_left_front_v = math.sqrt(state[3][5] ** 2 + state[3][6] ** 2)
        object_right_front_v = math.sqrt(state[4][5] ** 2 + state[4][6] ** 2)
        object_left_behind_v = math.sqrt(state[5][5] ** 2 + state[5][6] ** 2)
        object_right_behind_v = math.sqrt(state[6][5] ** 2 + state[6][6] ** 2)
        if abs(state[1][0]) > 100:
            state[1][0] = 0
        if abs(state[2][0]) > 100:
            state[2][0] = 0
        if abs(state[3][0]) > 100:
            state[3][0] = 0
        if abs(state[4][0]) > 100:
            state[4][0] = 0
        if abs(state[5][0]) > 100:
            state[5][0] = 0
        if abs(state[6][0]) > 100:
            state[6][0] = 0
        if abs(state[1][1]) > 100:
            state[1][1] = 0
        if abs(state[2][1]) > 100:
            state[2][1] = 0
        if abs(state[3][1]) > 100:
            state[3][1] = 0
        if abs(state[4][1]) > 100:
            state[4][1] = 0
        if abs(state[5][1]) > 100:
            state[5][1] = 0
        if abs(state[6][1]) > 100:
            state[6][1] = 0
        if object_front_v > 100:
            object_front_v = 0
        if object_behind_v > 100:
            object_behind_v = 0
        if object_left_front_v > 100:
            object_left_front_v = 0
        if object_right_front_v > 100:
            object_right_front_v = 0
        if object_left_behind_v > 100:
            object_left_behind_v = 0
        if object_right_behind_v > 100:
            object_right_behind_v = 0
        observation = np.array([ego_v, state[1][0], state[1][1], object_front_v,
                       state[2][0], state[2][1], object_behind_v,
                       state[3][0], state[3][1], object_left_front_v,
                       state[4][0], state[4][1], object_right_front_v,
                       state[5][0], state[5][1], object_left_behind_v,
                       state[6][0], state[6][1], object_right_behind_v])
        ego_v_state.append(ego_v)

        collision = 0  # Initialize collision to 0
        if abs(observation[1]) <= 4 and abs(observation[2]) <= 2:
            done = 1
            collision = 1
        if abs(observation[4]) <= 4 and abs(observation[5]) <= 2:
            done = 1
            collision = 1
        if abs(observation[7]) <= 4 and abs(observation[8]) <= 2:
            done = 1
            collision = 1
        if abs(observation[10]) <= 4 and abs(observation[11]) <= 2:
            done = 1
            collision = 1
        if abs(observation[13]) <= 4 and abs(observation[14]) <= 2:
            done = 1
            collision = 1
        if abs(observation[16]) <= 4 and abs(observation[17]) <= 2:
            done = 1
            collision = 1

        if i != len(states_of_scenario) - 1:
            next_state = states_of_scenario[i + 1]
            next_ego_v = math.sqrt(next_state[0][3] ** 2 + next_state[0][4] ** 2)
            action = min(next_ego_v - 10, 10) * 0.1
        else:
            action = actions[-1][0]
            done = 1
            reach = 1

        reward = get_reward(observation, done, collision, ego_v, reach)

        rewards.append(reward)
        observations.append(observation)
        actions.append([action])
        terminals.append([done])

for i in range(1, len(observations)):
    next_observations.append(observations[i])
next_observations.append(observations[0])

observations = np.array(observations, dtype='float32')
next_observations = np.array(next_observations, dtype='float32')
actions = np.array(actions, dtype='float32')
rewards = np.array(rewards, dtype='float32')
terminals = np.array(terminals, dtype='float32')

b = rewards.shape[0]

dataset = {}
dataset['observations'] = observations
dataset['next_observations'] = next_observations
dataset['actions'] = actions
dataset['rewards'] = rewards
dataset['terminals'] = terminals

pickle.dump(dataset, open('offline_dataset.pkl', 'wb'))