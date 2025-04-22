'''
Author: Maonan Wang
Date: 2025-04-21 19:04:38
LastEditTime: 2025-04-22 12:37:21
LastEditors: Maonan Wang
Description: 不同场景的 3D 可视化 (所有 Image 全部保存)
FilePath: /VLM-CloseLoop-TSC/vis_3d_env.py
'''
import os
import cv2
import random
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.tshub_env3d.tshub_env3d import Tshub3DEnvironment
path_convert = get_abs_path(__file__)
set_logger(path_convert('./'), terminal_log_level='INFO')

def convert_rgb_to_bgr(image):
    # Convert an RGB image to BGR
    return image[:, :, ::-1]


SCENARIO_NAME = "Hongkong_YMT" # 可视化场景
NETFILE = "ymt_eval"
JUNCTION_NAME = "J1"
CENTER_COORDINATES = (172, 201, 100)
PHASE_NUMBER = 4 # 绿灯相位数量

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
    sumo_cfg = path_convert(f"./sim_envs/{SCENARIO_NAME}/{NETFILE}.sumocfg")
    scenario_glb_dir = path_convert(f"./sim_envs/{SCENARIO_NAME}/3d_assets/")
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
        obs, reward, info, done, sensor_data = tshub_env3d.step(actions=actions)
        i_steps += 1

        try:
            # 将 sensor_data 的数据保存为图片
            base_path = path_convert(f"./sensor_outputs/{SCENARIO_NAME}/")
            for element_id, cameras in sensor_data.items():
                # Iterate over each camera type
                for camera_type, image_array in cameras.items():
                    # Create directory for the camera_type if it doesn't exist
                    camera_dir = os.path.join(base_path, element_id)
                    if not os.path.exists(camera_dir):
                        os.makedirs(camera_dir)

                    # Construct the image file path
                    image_path = os.path.join(camera_dir, f"{i_steps}.png")
                    # Save the numpy array as an image
                    cv2.imwrite(image_path, convert_rgb_to_bgr(image_array))
        except:
            pass

    tshub_env3d.close()