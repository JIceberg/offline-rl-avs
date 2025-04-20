import math
import gym
from gym import spaces, logger
from gym.utils import seeding
# from gym.envs.classic_control import rendering
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import pickle
from pathlib import Path
import random
import matplotlib
from matplotlib import transforms
from matplotlib.patches import Rectangle, Circle, Polygon
from io import BytesIO
from PIL import Image
import imageio
from tqdm import tqdm
import numpy as np
# from collections import namedtuple
# EgoState = namedtuple("EgoState", ["position", "heading", "velocity"])

def record_gif(env, policy, gif_filename='simulation.gif', fps=10, duration=None):
    """
    Record a GIF of the policy acting in the environment
    
    Args:
        env: Your environment
        policy: Function that takes observations and returns actions
        gif_filename: Output filename (should end with .gif)
        fps: Frames per second
        duration: Optional duration per frame (ms), overrides fps if set
    """
    frames = []
    obs, _ = env.reset()
    done = False
    
    print("Recording GIF...")
    
    with tqdm(total=1000, desc="Capturing frames") as pbar:
        while not done:
            action = policy.get_action(obs)
            obs, reward, done, truncated = env.step(action)
            
            frame = env.render(mode='rgb_array')
            if frame is not None:
                # Ensure correct format (H, W, 3) uint8
                if frame.ndim == 3 and frame.shape[2] == 3:
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    frames.append(frame)
            
            pbar.update(1)
            if done or truncated:
                break
    
    if len(frames) == 0:
        print("Warning: No frames were captured!")
        return
    
    print(f"Saving GIF with {len(frames)} frames...")
    
    try:
        # Calculate duration per frame if not specified
        if duration is None:
            duration = 1000 / fps  # Convert fps to ms per frame
        
        # Save as GIF
        imageio.mimsave(
            gif_filename,
            frames,
            format='GIF',
            duration=duration,
            loop=0  # 0 means infinite loop
        )
        print(f"GIF successfully saved to {gif_filename}")
    except Exception as e:
        print(f"Failed to save GIF: {e}")
    
    env.close()

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
            'data_for_simulator/train/')
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

        dist_traveled = self.ego_v * self.dt
        self._s_arc = min(self._path_len, self._s_arc + dist_traveled)
        # interpolate and set heading from path
        interp_heading = np.interp(self._s_arc, self._s_vals, self._headings_unwrapped)
        self.ego_yaw = normalize_angle(interp_heading)

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
        if seed is not None:
            self.seed(seed)

        self.scenario = random.choice(self.scenarios)
        self.ego_track = self.scenario['EGO']
        self.object_tracks = self.scenario['others']

        self.ego_x = self.ego_track.object_states[0].position[0]
        self.ego_y = self.ego_track.object_states[0].position[1]

        self.ego_yaw = self.ego_track.object_states[0].heading
        self.ego_v = sqrt(self.ego_track.object_states[0].velocity[0] ** 2 +
                          self.ego_track.object_states[0].velocity[1] ** 2)

        ego_states = self.ego_track.object_states
        xs = np.array([s.position[0] for s in ego_states], dtype=np.float32)
        ys = np.array([s.position[1] for s in ego_states], dtype=np.float32)
        deltas = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        self._s_vals = np.concatenate(([0.0], np.cumsum(deltas)))
        headings = np.array([s.heading for s in ego_states], dtype=np.float32)
        self._headings_unwrapped = np.unwrap(headings)
        self._path_len = float(self._s_vals[-1])
        self._s_arc = 0.0

        self.time = 0
        self.viewer = None
        self.steps_beyond_done = None
        return self._get_initial_state(), {}

    def transform_point(self,x, y, dx, dy, yaw):
        """Translate dx, dy in the local frame to world coordinates"""
        world_x = x + dx * np.cos(yaw) - dy * np.sin(yaw)
        world_y = y + dx * np.sin(yaw) + dy * np.cos(yaw)
        return world_x, world_y
    
    def draw_vehicle(self, ax, x, y, yaw, color='blue', alpha=1.0, zorder=3):
        # Assumes position (x, y) and yaw are in ego frame already
        length = 4.5
        width = 2.0
        corners = np.array([
            [ length/2,  width/2],
            [ length/2, -width/2],
            [-length/2, -width/2],
            [-length/2,  width/2],
        ])
        rotation = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])
        rotated = corners @ rotation.T
        translated = rotated + np.array([x, y])
        vehicle_poly = Polygon(translated, closed=True, color=color, alpha=alpha, zorder=zorder)
        ax.add_patch(vehicle_poly)

    def world_to_ego(self, x, y):
        dx = x - self.ego_x
        dy = y - self.ego_y
        cos_yaw = np.cos(-self.ego_yaw)
        sin_yaw = np.sin(-self.ego_yaw)
        ex = dx * cos_yaw - dy * sin_yaw
        ey = dx * sin_yaw + dy * cos_yaw
        return ex, ey

    def render(self, mode='human'):
        
        if not hasattr(self, 'fig') or not hasattr(self, 'ax'):
            matplotlib.use('Agg')
            plt.ioff()
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            plt.axis('off')
            self.fig.tight_layout(pad=0)

        self.ax.clear()

        # View window
        view_distance = 50
        self.ax.set_xlim(-view_distance, view_distance)
        self.ax.set_ylim(-view_distance / 2, view_distance / 2)
        self.ax.set_aspect('equal')
        # self.ax.set_xlim(self.ego_x - view_distance, self.ego_x + view_distance)
        # self.ax.set_ylim(self.ego_y - view_distance / 2, self.ego_y + view_distance / 2)

        # Road drawing - rotated to match ego vehicle's orientation
        road_length = 100  # Total length of visible road
        road_width = 40    # Total width of visible road
        
        # Road corners in ego-aligned coordinates (before rotation)
        road_half_length = road_length / 2
        road_half_width = road_width / 2
        road_corners_local = np.array([
            [-road_half_length, -road_half_width],
            [ road_half_length, -road_half_width],
            [ road_half_length,  road_half_width],
            [-road_half_length,  road_half_width]
        ])
        # Rotate road according to ego yaw
        rotation_matrix = np.array([
            [np.cos(self.ego_yaw), -np.sin(self.ego_yaw)],
            [np.sin(self.ego_yaw),  np.cos(self.ego_yaw)]
        ])
        road_corners_rotated = road_corners_local @ rotation_matrix.T
        
        # Create road polygon
        road_poly = Polygon(road_corners_rotated, color='lightgray', zorder=0)
        self.ax.add_patch(road_poly)
        # Lane markings (rotated to match ego orientation)
        lane_width = 3.7
        for lane_offset in [-lane_width, 0, lane_width]:
            # Create lane segment in local coordinates
            lane_x = np.array([-road_half_length, road_half_length])
            lane_y = np.array([lane_offset, lane_offset])
            
            # Rotate lane markings
            lane_points = np.column_stack((lane_x, lane_y)) @ rotation_matrix.T
            self.ax.plot(lane_points[:,0], lane_points[:,1], 'w--', linewidth=1, alpha=0.5, zorder=1)
        

        # Draw other vehicles
        for obj in self.object_tracks:
            if self.time < len(obj.object_states):
                state = obj.object_states[self.time]
                x, y = state.position
                yaw = state.heading
                vx, vy = state.velocity

                relative_yaw = (yaw - self.ego_yaw + np.pi) % (2 * np.pi) - np.pi

                # Determine if vehicle is moving forward or backward based on velocity
                local_vx = vx * np.cos(yaw) + vy * np.sin(yaw)
                is_moving_forward = local_vx >= 0  # Assuming positive velocity means forward
                # Place headlights - front if moving forward, back if moving backward
                headlight_x_local = 2.25 if is_moving_forward else -2.25
                headlight_y_local = 0.8

                ex, ey = self.world_to_ego(x, y)

                # Display distance on vehicle
                # self.ax.text(
                #     x, y,
                #     f"{distance:.1f} m",
                #     color='white',
                #     fontsize=8,
                #     ha='center',
                #     va='center',
                #     zorder=6,
                #     bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2')
                # )

                self.draw_vehicle(self.ax, ex, ey, relative_yaw, color='blue', alpha=0.9, zorder=3)

                # Rotate headlight positions by the vehicle's yaw
                for y_sign in [1, -1]:
                    hx_rotated = headlight_x_local * np.cos(relative_yaw) - y_sign * headlight_y_local * np.sin(relative_yaw)
                    hy_rotated = headlight_x_local * np.sin(relative_yaw) + y_sign * headlight_y_local * np.cos(relative_yaw)
                    self.ax.add_patch(Circle((ex + hx_rotated, ey + hy_rotated), 0.3, color='yellow'))

                # Add normalized direction arrow (same length for all vehicles)
                if (vx != 0 or vy != 0):  # Only draw if vehicle is moving
                    # Use relative yaw directly for arrow direction
                    arrow_length = 3.0
                    arrow_dx = arrow_length * np.cos(relative_yaw)
                    arrow_dy = arrow_length * np.sin(relative_yaw)
                    
                    # Draw arrow
                    self.ax.arrow(ex, ey, 
                                arrow_dx, 
                                arrow_dy,
                                color='green', width=0.2, zorder=4)

        self.draw_vehicle(self.ax, 0, 0, 0, color='red', alpha=1.0, zorder=4)

        for offset in [-1, 1]:
            # # Headlight in ego vehicle local frame
            # hx_local, hy_local = self.transform_point(0, 0, 4, offset * 1, self.ego_yaw)
            # self.ax.add_patch(Circle((hx_local, hy_local), 0.3, color='yellow'))
            # Headlight in ego vehicle local frame (front of the vehicle)
            hx_local = 2.25  # Half of vehicle length (4.5/2) to place at front
            hy_local = offset * 0.8  # Slightly inset from the sides
            self.ax.add_patch(Circle((hx_local, hy_local), 0.3, color='yellow'))

        self.ax.arrow(0, 0, 3, 0, color='green', width=0.1, zorder=5)
        # Ego taillights (black)
        # self.draw_tail_light_bar(self.ax, self.ego_x, self.ego_y, self.ego_yaw)

        # Ego info panel
        info_text = (
            f"Time: {self.time * self.dt:.1f}s\n"
            f"Speed: {self.ego_v:.1f} m/s\n"
            f"Position: ({self.ego_x:.1f}, {self.ego_y:.1f})"
        )
        self.ax.text(
            0.02, 0.98, info_text,
            transform=self.ax.transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'),
            fontsize=10
        )

        # Render to array or display
        if mode == 'rgb_array':
            try:
                self.fig.canvas.draw()
                width, height = self.fig.canvas.get_width_height()
                img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                return img.reshape(height, width, 3)
            except Exception:
                return np.zeros((480, 640, 3), dtype=np.uint8)
        elif mode == 'human':
            try:
                plt.draw()
                plt.pause(0.01)
            except:
                pass
            return None


   