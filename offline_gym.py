import math
import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
from math import sqrt
import pickle
from pathlib import Path
import random

def get_reward(observation, ego_speed, ego_accel, collision, done, reach):
    r_terminal = 10 if reach else 0
    r_collision = -100 if collision else 0
    r_speed = ego_speed / 20
    # r_smooth = -0.1 * abs(ego_accel)

    total_reward = r_terminal + r_collision + r_speed # + r_smooth
    return float(total_reward) / 20.

def normalize_angle(angle_rad):
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

def object_to_ego(x, y, yaw):
    res_x = math.cos(yaw) * x - math.sin(yaw) * y
    res_y = math.sin(yaw) * x + math.cos(yaw) * y
    return res_x, res_y

class OfflineRL(gym.Env):
    def __init__(self):
        argoverse_scenario_dir = Path(
            'data_for_simulator/')
        all_scenario_files = sorted(argoverse_scenario_dir.rglob("*.pkl"))
        scenario_file_lists = (all_scenario_files[:20])
        self.scenarios = []
        for scenario_file_list in scenario_file_lists:
            scenario = pickle.load(open(scenario_file_list, 'rb'))
            self.scenarios.append(scenario)

        self.time = 0
        self.dt = 0.1

        self.x_threshold = 100000
        self.v_threshold = 100000
        self.max_speed = 20
        self.max_a = 2
        self.max_angle = np.pi / 2
        high = np.array([self.v_threshold,
                         self.x_threshold,
                         self.x_threshold,
                         self.v_threshold,
                         np.pi,
                         self.x_threshold,
                         self.x_threshold,
                         self.v_threshold,
                         np.pi,
                         self.x_threshold,
                         self.x_threshold,
                         self.v_threshold,
                         np.pi,
                         self.x_threshold,
                         self.x_threshold,
                         self.v_threshold,
                         np.pi,
                         self.x_threshold,
                         self.x_threshold,
                         self.v_threshold,
                         np.pi,
                         self.x_threshold,
                         self.x_threshold,
                         self.v_threshold,
                         np.pi
                         ],
                        dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-self.max_a], dtype=np.float32),
            high=np.array([self.max_a], dtype=np.float32)
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            shape=(25,),
            dtype=np.float32
        )

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)

    def step(self, action):
        done = 0
        reach = 0

        # find next ego position
        accel = action[0]
        accel = np.clip(accel, -1, 1) * self.max_a
        # steering_angle = action[1]
        # steering_angle = np.clip(steering_angle, -1, 1) * self.max_angle
        self.ego_v += accel * self.dt
        self.ego_v = np.clip(self.ego_v, 0, self.max_speed)
        self.ego_yaw = self.ego_track.object_states[self.time].heading
        self.ego_yaw = normalize_angle(self.ego_yaw)

        dx = self.ego_v * np.cos(self.ego_yaw) * self.dt
        dy = self.ego_v * np.sin(self.ego_yaw) * self.dt

        self.ego_x += dx
        self.ego_y += dy

        # get nearby objects
        object_front = [0.0 for _ in range(5)]
        object_behind = [0.0 for _ in range(5)]
        object_left_front = [0.0 for _ in range(5)]
        object_right_front = [0.0 for _ in range(5)]
        object_left_behind = [0.0 for _ in range(5)]
        object_right_behind = [0.0 for _ in range(5)]

        for other_track in self.object_tracks:
            object_state = None
            for other_state in other_track.object_states:
                if other_state.timestep == self.time:
                    object_state = other_state
                    break
            if object_state is None:
                continue

            object_x = object_state.position[0]
            object_y = object_state.position[1]
            object_yaw = object_state.heading
            object_v_x = object_state.velocity[0]
            object_v_y = object_state.velocity[1]

            object_v = math.sqrt(object_v_x ** 2 + object_v_y ** 2)

            x_to_ego, y_to_ego = object_to_ego(object_x - self.ego_x,
                                                object_y - self.ego_y,
                                                -self.ego_yaw)                    
            dist_to_ego = sqrt(x_to_ego ** 2 + y_to_ego ** 2)

            # ignore objects far away
            if abs(x_to_ego) > 30 or abs(y_to_ego) > 30:
                continue

            # front
            if x_to_ego > 0 and y_to_ego <= 2 and y_to_ego >= -2:
                if object_front[0] == 0:
                    object_front[0] = x_to_ego
                    object_front[1] = y_to_ego
                    object_front[2] = object_v
                    object_front[3] = object_yaw
                    object_front[4] = 1
                elif x_to_ego < object_front[0]:
                    object_front[0] = x_to_ego
                    object_front[1] = y_to_ego
                    object_front[2] = object_v
                    object_front[3] = object_yaw
                else:
                    continue

            # behind
            if x_to_ego < 0 and y_to_ego <= 2 and y_to_ego >= -2:
                if object_behind[0] == 0:
                    object_behind[0] = x_to_ego
                    object_behind[1] = y_to_ego
                    object_behind[2] = object_v
                    object_behind[3] = object_yaw
                    object_behind[4] = 1
                elif x_to_ego > object_behind[0]:
                    object_behind[0] = x_to_ego
                    object_behind[1] = y_to_ego
                    object_behind[2] = object_v
                    object_behind[3] = object_yaw
                else:
                    continue
            
            # left front
            if x_to_ego > 0 and y_to_ego > 2:
                if object_left_front[0] == 0:
                    object_left_front[0] = x_to_ego
                    object_left_front[1] = y_to_ego
                    object_left_front[2] = object_v
                    object_left_front[3] = object_yaw
                    object_left_front[4] = 1
                elif dist_to_ego < sqrt(object_left_front[0] ** 2 + object_left_front[1] ** 2):
                    object_left_front[0] = x_to_ego
                    object_left_front[1] = y_to_ego
                    object_left_front[2] = object_v
                    object_left_front[3] = object_yaw
                else:
                    continue
            
            # right front
            if x_to_ego > 0 and y_to_ego < -2:
                if object_right_front[0] == 0:
                    object_right_front[0] = x_to_ego
                    object_right_front[1] = y_to_ego
                    object_right_front[2] = object_v
                    object_right_front[3] = object_yaw
                    object_right_front[4] = 1
                elif dist_to_ego < sqrt(object_right_front[0] ** 2 + object_right_front[1] ** 2):
                    object_right_front[0] = x_to_ego
                    object_right_front[1] = y_to_ego
                    object_right_front[2] = object_v
                    object_right_front[3] = object_yaw
                else:
                    continue

            # left behind
            if x_to_ego < 0 and y_to_ego > 2:
                if object_left_behind[0] == 0:
                    object_left_behind[0] = x_to_ego
                    object_left_behind[1] = y_to_ego
                    object_left_behind[2] = object_v
                    object_left_behind[3] = object_yaw
                    object_left_behind[4] = 1
                elif dist_to_ego < sqrt(object_left_behind[0] ** 2 + object_left_behind[1] ** 2):
                    object_left_behind[0] = x_to_ego
                    object_left_behind[1] = y_to_ego
                    object_left_behind[2] = object_v
                    object_left_behind[3] = object_yaw
                else:
                    continue
            
            # right behind
            if x_to_ego < 0 and y_to_ego < -2:
                if object_right_behind[0] == 0:
                    object_right_behind[0] = x_to_ego
                    object_right_behind[1] = y_to_ego
                    object_right_behind[2] = object_v
                    object_right_behind[3] = object_yaw
                    object_right_behind[4] = 1
                elif dist_to_ego < sqrt(object_right_behind[0] ** 2 + object_right_behind[1] ** 2):
                    object_right_behind[0] = x_to_ego
                    object_right_behind[1] = y_to_ego
                    object_right_behind[2] = object_v
                    object_right_behind[3] = object_yaw
                else:
                    continue

        # print object locations
        # print('ego_x:', self.ego_x)
        # print('ego_y:', self.ego_y)
        # print('ego_yaw:', self.ego_yaw)
        # print('ego_v:', self.ego_v)
        # print('object_front:', object_front)
        # print('object_behind:', object_behind)
        # print('object_left_front:', object_left_front)
        # print('object_right_front:', object_right_front)
        # print('object_left_behind:', object_left_behind)
        # print('object_right_behind:', object_right_behind)

        observation = np.array([self.ego_v,
                                object_front[0], object_front[1], object_front[2], object_front[3],
                                object_behind[0], object_behind[1], object_behind[2], object_behind[3],
                                object_left_front[0], object_left_front[1], object_left_front[2], object_left_front[3],
                                object_right_front[0], object_right_front[1], object_right_front[2], object_right_front[3],
                                object_left_behind[0], object_left_behind[1], object_left_behind[2], object_left_behind[3],
                                object_right_behind[0], object_right_behind[1], object_right_behind[2], object_right_behind[3]])
        
        # check collision
        collision = 0
        if abs(observation[1]) <= 4 and abs(observation[2]) <= 2 and object_front[4]:
            done = 1
            collision = 1
        if abs(observation[5]) <= 4 and abs(observation[6]) <= 2 and object_behind[4]:
            done = 1
            collision = 1
        if abs(observation[9]) <= 4 and abs(observation[10]) <= 2 and object_left_front[4]:
            done = 1
            collision = 1
        if abs(observation[13]) <= 4 and abs(observation[14]) <= 2 and object_right_front[4]:
            done = 1
            collision = 1
        if abs(observation[17]) <= 4 and abs(observation[18]) <= 2 and object_left_behind[4]:
            done = 1
            collision = 1
        if abs(observation[21]) <= 4 and abs(observation[22]) <= 2 and object_right_behind[4]:
            done = 1
            collision = 1

        self.time += 1
        if self.time == len(self.scenario['states']):
            done = 1
            reach = 1

        reward = get_reward(observation, self.ego_v, action[0], collision, done, reach)
        return observation, float(reward), float(done), collision
    
    def _get_initial_state(self):
        state = self.scenario['states'][0]
        ego_v = math.sqrt(state[0][3] ** 2 + state[0][4] ** 2)
        object_front_v = math.sqrt(state[1][5] ** 2 + state[1][6] ** 2)
        object_behind_v = math.sqrt(state[2][5] ** 2 + state[2][6] ** 2)
        object_left_front_v = math.sqrt(state[3][5] ** 2 + state[3][6] ** 2)
        object_right_front_v = math.sqrt(state[4][5] ** 2 + state[4][6] ** 2)
        object_left_behind_v = math.sqrt(state[5][5] ** 2 + state[5][6] ** 2)
        object_right_behind_v = math.sqrt(state[6][5] ** 2 + state[6][6] ** 2)
        observation = np.array([ego_v,
                                state[1][0], state[1][1], object_front_v, state[1][4],
                                state[2][0], state[2][1], object_behind_v, state[2][4],
                                state[3][0], state[3][1], object_left_front_v, state[3][4],
                                state[4][0], state[4][1], object_right_front_v, state[4][4],
                                state[5][0], state[5][1], object_left_behind_v, state[5][4],
                                state[6][0], state[6][1], object_right_behind_v, state[6][4]])
        return observation

    def reset(self, *, seed=None, options=None):
        self.scenario = random.choice(self.scenarios)
        self.ego_track = self.scenario['EGO']
        self.object_tracks = self.scenario['others']

        self.ego_x = self.ego_track.object_states[0].position[0]
        self.ego_y = self.ego_track.object_states[0].position[1]

        self.ego_yaw = self.ego_track.object_states[0].heading
        self.ego_v = sqrt(self.ego_track.object_states[0].velocity[0] ** 2 +
                          self.ego_track.object_states[0].velocity[1] ** 2)

        self.time = 0
        self.viewer = None
        self.steps_beyond_done = None
        return self._get_initial_state(), {}

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        ego_x = 300
        ego_y = 200
        car_width = 20
        car_length = 40

        # number_of_car = len(self.object_position_for_view)
        l, r, t, b = -car_width / 2, car_width / 2, car_length / 2, -car_length / 2
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            ego_car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            ego_car.add_attr(self.carttrans)
            ego_car.set_color(1, 0, 0)
            self.viewer.add_geom(ego_car)
            for i in range(len(self.object_tracks)):
                globals()['object_' + str(i)] = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                globals()['object_trans' + str(i)] = rendering.Transform()
                globals()['object_' + str(i)].add_attr(globals()['object_trans' + str(i)])
                self.viewer.add_geom(globals()['object_' + str(i)])

        # Edit the pole polygon vertex

        self.carttrans.set_translation(ego_x, ego_y)
        self.carttrans.set_rotation(self.ego_yaw - math.pi / 2)
        for i in range(len(self.object_tracks)):
            if self.object_tracks[i].object_states[0].timestep <= self.time <= self.object_tracks[i].object_states[
                -1].timestep:
                for object_state in self.object_tracks[i].object_states:
                    if object_state.timestep == self.time:
                        object_x_to_ego = ego_x + (object_state.position[0] - self.ego_x) * 10
                        object_y_to_ego = ego_y + (object_state.position[1] - self.ego_y) * 10
                        globals()['object_trans' + str(i)].set_translation(object_x_to_ego, object_y_to_ego)
                        globals()['object_trans' + str(i)].set_rotation(object_state.heading - math.pi / 2)
                        break
            else:
                globals()['object_trans' + str(i)].set_translation(10000, 10000)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None