'''
Author: Maonan Wang
Date: 2025-04-21 19:04:38
LastEditTime: 2025-05-08 21:00:15
LastEditors: Maonan Wang
Description: 对场景进行可视化, 并存储每一个 step 的信息, 包括以下内容:
-> 1. 路口每一个方向的摄像机视角
-> 2. 路口的俯视角的视图
-> 3. 当前场景内车辆的信息
'''
import os
import cv2
import random
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.tshub_env3d.tshub_env3d import Tshub3DEnvironment

from _config import SCENARIO_CONFIGS
from utils.tools import save_to_json

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'), terminal_log_level='INFO')

def convert_rgb_to_bgr(image):
    # Convert an RGB image to BGR
    return image[:, :, ::-1]

# 读取场景配置
SCENARIO_NAME = "SouthKorea_Songdo" # 可视化场景
config = SCENARIO_CONFIGS.get(SCENARIO_NAME) # 获取特定场景的配置
NETFILE = config["NETFILE"]
JUNCTION_NAME = config["JUNCTION_NAME"]
CENTER_COORDINATES = config["CENTER_COORDINATES"]
PHASE_NUMBER = config["PHASE_NUMBER"] # 绿灯相位数量

# 初始化场景飞行器位置, 获得俯视角图像
aircraft_inits = {
    'a1': {
        "aircraft_type": "drone",
        "action_type": "stationary", # 水平移动
        "position": CENTER_COORDINATES, "speed":3, "heading":(1,1,0), # 初始位置
        "communication_range":100, 
        "if_sumo_visualization":True, "img_file":None,
        "custom_update_cover_radius":None # 使用自定义的计算
    },
}


if __name__ == '__main__':
    sumo_cfg = path_convert(f"../sim_envs/{SCENARIO_NAME}/{NETFILE}.sumocfg")
    scenario_glb_dir = path_convert(f"../sim_envs/{SCENARIO_NAME}/3d_assets/")
    tshub_env3d = Tshub3DEnvironment(
        sumo_cfg=sumo_cfg,
        scenario_glb_dir=scenario_glb_dir,
        is_aircraft_builder_initialized=True,
        is_map_builder_initialized=False,
        is_vehicle_builder_initialized=True, 
        is_traffic_light_builder_initialized=True, # 需要打开信号灯才会有路口的摄像头
        aircraft_inits=aircraft_inits,
        tls_ids=[JUNCTION_NAME],
        tls_action_type='choose_next_phase',
        vehicle_action_type='lane_continuous_speed',
        use_gui=True, 
        num_seconds=500,
        collision_action="warn",
        # 下面是用于渲染的参数
        preset="1080P",
        render_mode="offscreen", # 将结果保存即可
        should_count_vehicles=True, # 额外返回场景内车辆位置信息, 用于渲染
        debuger_print_node=False,
        debuger_spin_camera=False,
        sensor_config={
            'tls': {
                JUNCTION_NAME:{
                    'sensor_types': ['junction_front_all'],
                    'tls_camera_height': 15,
                } # 路口传感器
            },
            'aircraft': {
                "a1": {"sensor_types": ['aircraft_all']}
            }, # 飞行器传感器
        }
    )

    obs = tshub_env3d.reset()
    done = False
    i_steps = 0
    while not done:
        random_phase = random.randint(0, PHASE_NUMBER-1)
        actions = {
            'vehicle': dict(),
            'tls': {JUNCTION_NAME: random_phase},
        }
        obs, reward, info, done, sensor_datas = tshub_env3d.step(actions=actions)
        sensor_data = sensor_datas['image'] # 获得图片数据
        vehicle_elements = sensor_datas['veh_elements'] # 车辆数据
        i_steps += 1

        try:
            # 存储每一个时刻的数据
            base_path = path_convert(f"./sensor_outputs/{SCENARIO_NAME}/{i_steps}")
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            
            # -> 存储车辆数据
            save_to_json(vehicle_elements, os.path.join(base_path, "vehs.json"))
            # -> 存储传感器数据
            for element_id, cameras in sensor_data.items():
                # Iterate over each camera type
                for camera_type, image_array in cameras.items():
                    # Save the numpy array as an image
                    image_path = os.path.join(base_path, f"{element_id}_{camera_type}.png")
                    cv2.imwrite(image_path, convert_rgb_to_bgr(image_array))
        except:
            pass

    tshub_env3d.close()