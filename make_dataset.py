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

observations = []
next_observations = []
actions = []
rewards = []
terminals = []

for scenario in scenarios:
    states_of_scenario = scenario['states']
    for i in range(len(states_of_scenario)):
        state = states_of_scenario[i]
        action = [0] # [0, 0]
        done = 0
        reach = 0

        ego = state[0]
        object_front = state[1]
        object_behind = state[2]
        object_left_front = state[3]
        object_right_front = state[4]
        object_left_behind = state[5]
        object_right_behind = state[6]

        ego_v = math.sqrt(ego[3] ** 2 + ego[4] ** 2)
        object_front_v = math.sqrt(object_front[5] ** 2 + object_front[6] ** 2)
        object_behind_v = math.sqrt(object_behind[5] ** 2 + object_behind[6] ** 2)
        object_left_front_v = math.sqrt(object_left_front[5] ** 2 + object_left_front[6] ** 2)
        object_right_front_v = math.sqrt(object_right_front[5] ** 2 + object_right_front[6] ** 2)
        object_left_behind_v = math.sqrt(object_left_behind[5] ** 2 + object_left_behind[6] ** 2)
        object_right_behind_v = math.sqrt(object_right_behind[5] ** 2 + object_right_behind[6] ** 2)

        if abs(object_front[0]) > 100:
            object_front[0] = 0
        if abs(object_behind[0]) > 100:
            object_behind[0] = 0
        if abs(object_left_front[0]) > 100:
            object_left_front[0] = 0
        if abs(object_right_front[0]) > 100:
            object_right_front[0] = 0
        if abs(object_left_behind[0]) > 100:
            object_left_behind[0] = 0
        if abs(object_right_behind[0]) > 100:
            object_right_behind[0] = 0
        if abs(object_front[1]) > 100:
            object_front[1] = 0
        if abs(object_behind[1]) > 100:
            object_behind[1] = 0
        if abs(object_left_front[1]) > 100:
            object_left_front[1] = 0
        if abs(object_right_front[1]) > 100:
            object_right_front[1] = 0
        if abs(object_left_behind[1]) > 100:
            object_left_behind[1] = 0
        if abs(object_right_behind[1]) > 100:
            object_right_behind[1] = 0
            
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
        
        observation = np.array([ego_v,
                                object_front[0], object_front[1], object_front_v, object_front[4],
                                object_behind[0], object_behind[1], object_behind_v, object_behind[4],
                                object_left_front[0], object_left_front[1], object_left_front_v, object_left_front[4],
                                object_right_front[0], object_right_front[1], object_right_front_v, object_right_front[4],
                                object_left_behind[0], object_left_behind[1], object_left_behind_v, object_left_behind[4],
                                object_right_behind[0], object_right_behind[1], object_right_behind_v, object_right_behind[4]
                                ], dtype='float32')

        collision = 0  # Initialize collision to 0
        if abs(observation[1]) <= 4 and abs(observation[2]) <= 2:
            done = 1
            collision = 1
        if abs(observation[5]) <= 4 and abs(observation[6]) <= 2:
            done = 1
            collision = 1
        if abs(observation[9]) <= 4 and abs(observation[10]) <= 2:
            done = 1
            collision = 1
        if abs(observation[13]) <= 4 and abs(observation[14]) <= 2:
            done = 1
            collision = 1
        if abs(observation[17]) <= 4 and abs(observation[18]) <= 2:
            done = 1
            collision = 1
        if abs(observation[21]) <= 4 and abs(observation[22]) <= 2:
            done = 1
            collision = 1

        if i != len(states_of_scenario) - 1:
            next_state = states_of_scenario[i + 1]
            next_ego = next_state[0]
            next_ego_v = math.sqrt(next_ego[3] ** 2 + next_ego[4] ** 2)
            accel = (next_ego_v - ego_v) * 10
            ego_heading = ego[2]

            action[0] = accel
            # action[1] = ego_heading
        else:
            action = actions[-1]
            done = 1
            reach = 1

        reward = get_reward(observation, ego_v, action[0], collision, done, reach)

        rewards.append([reward])
        observations.append(observation)
        actions.append(action)
        terminals.append([done])

for i in range(1, len(observations)):
    next_observations.append(observations[i])
next_observations.append(observations[-1])

observations = np.array(observations, dtype='float32')
next_observations = np.array(next_observations, dtype='float32')
actions = np.array(actions, dtype='float32')
rewards = np.array(rewards, dtype='float32')
terminals = np.array(terminals, dtype='float32')

b = rewards.shape[0]

dataset = {}
dataset['observations'] = observations
dataset['next_states'] = next_observations
dataset['actions'] = actions
dataset['rewards'] = rewards
dataset['dones'] = terminals

pickle.dump(dataset, open('offline_dataset.pkl', 'wb'))