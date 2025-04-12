from enum import Enum, unique
from pathlib import Path
from random import choices
from typing import Final

import click
import numpy as np
from joblib import Parallel, delayed
from rich.progress import track
import math
import json
import pickle

from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.viz.scenario_visualization import visualize_scenario
from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting.data_schema import ObjectType, TrackCategory

import matplotlib.pyplot as plt
import gym
import gc
import numpy as np
gc.enable()
from typing import Final, List, Optional, Sequence, Set, Tuple


_STATIC_OBJECT_TYPES: Set[ObjectType] = {
    ObjectType.STATIC,
    ObjectType.BACKGROUND,
    ObjectType.CONSTRUCTION,
    ObjectType.RIDERLESS_BICYCLE,
}

def generate_scenario_visualization(scenario_path: Path):
    """Generate and save dynamic visualization for a single Argoverse scenario.

    NOTE: This function assumes that the static map is stored in the same directory as the scenario file.

    Args:
        scenario_path: Path to the parquet file corresponding to the Argoverse scenario to visualize.
    """
    scenario_id = scenario_path.stem.split("_")[-1]
    static_map_path = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"

    scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
    static_map = ArgoverseStaticMap.from_json(static_map_path)
    # scenario_vector_drivable_areas = static_map.get_scenario_vector_drivable_areas()
    # scenario_lane_segment_ids = static_map.get_scenario_lane_segment_ids()
    # print(scenario_lane_segment_ids)

    return scenario, static_map, scenario_id


def object_to_ego(x, y, yaw):
    res_x = math.cos(yaw) * x - math.sin(yaw) * y
    res_y = math.sin(yaw) * x + math.cos(yaw) * y
    return res_x, res_y


if __name__ == "__main__":
    argoverse_scenario_dir = Path(
        'data_of_argo/train')
    all_scenario_files = sorted(argoverse_scenario_dir.rglob("*.parquet"))
    scenario_file_list = (
        all_scenario_files[:1000])


    for scenario_file in scenario_file_list:
        states = []
        scenario_offline = {}
        ego_car_track = []
        else_car_tracks = []

        scenario, static_map, scenario_id = generate_scenario_visualization(scenario_file)

        for track in scenario.tracks:
            if track.category == TrackCategory.FOCAL_TRACK:
                ego_car_track.append(track)
            else:
                else_car_tracks.append(track)

        scenario_offline['EGO'] = ego_car_track[0]
        for i in range(len(ego_car_track[0].object_states)):
            state = []
            ego_car_x = ego_car_track[0].object_states[i].position[0]
            ego_car_y = ego_car_track[0].object_states[i].position[1]
            ego_car_yaw = ego_car_track[0].object_states[i].heading
            ego_car_v_x = ego_car_track[0].object_states[i].velocity[0]
            ego_car_v_y = ego_car_track[0].object_states[i].velocity[1]

            ego_state = [ego_car_x, ego_car_y, ego_car_yaw, ego_car_v_x, ego_car_v_y]

            object_front = [10000, 10000, 10000, 10000, 10000, 10000,
                            10000]
            object_behind = [-10000, 10000, 10000, 10000, 10000, 10000,
                             10000]
            object_left_front = [10000, 10000, 10000, 10000, 10000, 10000,
                                 10000]
            object_right_front = [10000, -10000, 10000, 10000, 10000, 10000,
                                  10000]
            object_left_behind = [-10000, 10000, 10000, 10000, 10000, 10000,
                                  10000]
            object_right_behind = [-10000, -10000, 10000, 10000, 10000, 10000,
                                   10000]
            # 从其他轨迹中筛选有用数据
            for else_car_track in else_car_tracks:
                # if track.category != TrackCategory.FOCAL_TRACK:
                #     continue
                # 判断是否使用此轨迹
                useful = False
                # 时间框定
                if else_car_track.object_states[0].timestep <= i <= else_car_track.object_states[-1].timestep:
                    for object_state in else_car_track.object_states:
                        if object_state.timestep == i:
                            object_x = object_state.position[0]
                            object_y = object_state.position[1]
                            object_yaw = object_state.heading
                            object_v_x = object_state.velocity[0]
                            object_v_y = object_state.velocity[1]

                            x_to_ego, y_to_ego = object_to_ego(object_x - ego_car_x, object_y - ego_car_y, -ego_car_yaw)

                            if abs(object_yaw - ego_car_yaw) > 90 / 180 * math.pi:
                                continue
                            if x_to_ego > 30 or x_to_ego < -12 or y_to_ego < -5.25 or y_to_ego > 5.25:
                                continue
                            # 选择前车
                            if 0 < x_to_ego and -1.75 <= y_to_ego <= 1.75:
                                if x_to_ego < object_front[0]:
                                    object_front[0] = x_to_ego
                                    object_front[1] = y_to_ego
                                    object_front[2] = object_x
                                    object_front[3] = object_y
                                    object_front[4] = object_yaw
                                    object_front[5] = object_v_x
                                    object_front[6] = object_v_y
                                    useful = True
                                else:
                                    continue
                            # 选择后车
                            if x_to_ego < 0 and -1.75 <= y_to_ego <= 1.75:
                                if x_to_ego > object_behind[0]:
                                    object_behind[0] = x_to_ego
                                    object_behind[1] = y_to_ego
                                    object_behind[2] = object_x
                                    object_behind[3] = object_y
                                    object_behind[4] = object_yaw
                                    object_behind[5] = object_v_x
                                    object_behind[6] = object_v_y
                                    useful = True
                                else:
                                    continue
                            # 选择右前车
                            if y_to_ego < -1.75 and x_to_ego >= 0:
                                if math.sqrt(x_to_ego ** 2 + y_to_ego ** 2) < math.sqrt(
                                        object_right_front[0] ** 2 + object_right_front[1] ** 2):
                                    object_right_front[0] = x_to_ego
                                    object_right_front[1] = y_to_ego
                                    object_right_front[2] = object_x
                                    object_right_front[3] = object_y
                                    object_right_front[4] = object_yaw
                                    object_right_front[5] = object_v_x
                                    object_right_front[6] = object_v_y
                                    useful = True
                                else:
                                    continue
                            # 选择左前车
                            if y_to_ego > 1.75 and x_to_ego >= 0:
                                if math.sqrt(x_to_ego ** 2 + y_to_ego ** 2) < math.sqrt(
                                        object_left_front[0] ** 2 + object_left_front[1] ** 2):
                                    object_left_front[0] = x_to_ego
                                    object_left_front[1] = y_to_ego
                                    object_left_front[2] = object_x
                                    object_left_front[3] = object_y
                                    object_left_front[4] = object_yaw
                                    object_left_front[5] = object_v_x
                                    object_left_front[6] = object_v_y
                                    useful = True
                                else:
                                    continue
                            # 选择右后车
                            if y_to_ego < -1.75 and x_to_ego < 0:
                                if math.sqrt(x_to_ego ** 2 + y_to_ego ** 2) < math.sqrt(
                                        object_right_behind[0] ** 2 + object_right_behind[1] ** 2):
                                    object_right_behind[0] = x_to_ego
                                    object_right_behind[1] = y_to_ego
                                    object_right_behind[2] = object_x
                                    object_right_behind[3] = object_y
                                    object_right_behind[4] = object_yaw
                                    object_right_behind[5] = object_v_x
                                    object_right_behind[6] = object_v_y
                                    useful = True
                                else:
                                    continue
                            # 选择左后车
                            if y_to_ego > 1.75 and x_to_ego < 0:
                                if math.sqrt(x_to_ego ** 2 + y_to_ego ** 2) < math.sqrt(
                                        object_left_behind[0] ** 2 + object_left_behind[1] ** 2):
                                    object_left_behind[0] = x_to_ego
                                    object_left_behind[1] = y_to_ego
                                    object_left_behind[2] = object_x
                                    object_left_behind[3] = object_y
                                    object_left_behind[4] = object_yaw
                                    object_left_behind[5] = object_v_x
                                    object_left_behind[6] = object_v_y
                                    useful = True
                                else:
                                    continue
                        else:
                            continue
                    if not else_car_track.track_id in scenario_offline.keys() and useful:
                        scenario_offline[else_car_track.track_id] = else_car_track
                else:
                    continue
            state.append(ego_state)
            state.append(object_front)
            state.append(object_behind)
            state.append(object_left_front)
            state.append(object_right_front)
            state.append(object_left_behind)
            state.append(object_right_behind)
            states.append(state)
        scenario_offline['states'] = states
        save_path = "data_for_simulator/train/" + scenario_id + ".pkl"
        pickle.dump(scenario_offline, open(save_path, 'wb'))  # 序列化