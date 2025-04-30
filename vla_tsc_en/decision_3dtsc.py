'''
Author: Maonan Wang
Date: 2025-04-23 15:13:54
LastEditTime: 2025-04-30 17:53:52
LastEditors: Maonan Wang
Description: VLA TSC (根据传感器图像进行决策)
FilePath: /VLM-CloseLoop-TSC/vla_tsc_en/decision_3dtsc.py
'''
# AI Agents
import json
from qwen_agent.agents import GroupChat
from qwen_agent.utils.output_beautify import typewriter_print
from vlm_agent import (
    scene_understanding_agent, # 图像理解 Agent
    scene_analysis_agent, # 场景分析 Agent
    rl_agent,
    ConcernCaseAgent,
    llm_cfg
)


# 3D TSC ENV
import re
import cv2
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from utils.make_tsc_env import make_env
from _config import SCENARIO_CONFIGS

def convert_rgb_to_bgr(image):
    # Convert an RGB image to BGR
    return image[:, :, ::-1]

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))

def extract_action(response):
    """将回复中的文本转换为 Traffic Phase 进行下发
    """
    match = re.search(r'\d+', response)
    if match:
        return np.array([int(match.group())])
    raise ValueError("No number found in the given string.")

# 全局变量
scenario_key = "Hongkong_YMT" # Hongkong_YMT, SouthKorea_Songdo, France_Massy
config = SCENARIO_CONFIGS.get(scenario_key) # 获取特定场景的配置
SCENARIO_NAME = config["SCENARIO_NAME"] # 场景名称
NETFILE = config["NETFILE"] # sumocfg 文件, 加载 eval 文件
JUNCTION_NAME = config["JUNCTION_NAME"] # sumo net 对应的路口 ID
PHASE_NUMBER = config["PHASE_NUMBER"] # 绿灯相位数量
SENSOR_INDEX_2_PHASE_INDEX = config["SENSOR_INDEX_2_PHASE_INDEX"] # 传感器与 Traffic Phase 的对应关系

# Init Agents
concer_case_decision_agent = ConcernCaseAgent(
    name='concer case decision agent',
    description='You play the role of a traffic police officer directing traffic at an intersection. When there are special vehicles at the intersection, such as police cars, ambulances, and fire engines, etc., it is up to you to make decisions.',
    llm_cfg=llm_cfg,
    phase_num=PHASE_NUMBER, # 当前路口存在的相位数量
) 

agents = [rl_agent, concer_case_decision_agent]
group_decision_bots = GroupChat(
    llm=llm_cfg, 
    agents=agents,
    agent_selection_method='auto',
)


if __name__ == '__main__':
    # #########
    # Init Env
    # #########
    sumo_cfg = path_convert(f"../sim_envs/{SCENARIO_NAME}/{NETFILE}.sumocfg")
    scenario_glb_dir = path_convert(f"../sim_envs/{SCENARIO_NAME}/3d_assets/")
    trip_info = path_convert(f"./results/{SCENARIO_NAME}/tripinfo.out.xml")
    tls_add = [
        path_convert(f'../sim_envs/{SCENARIO_NAME}/add/e2.add.xml'), # 探测器
        path_convert(f'../sim_envs/{SCENARIO_NAME}/add/tls_programs.add.xml'), # 信号灯
    ]
    params = {
        'tls_id':JUNCTION_NAME,
        'num_seconds':600,
        'number_phases':PHASE_NUMBER,
        'sumo_cfg':sumo_cfg,
        'scenario_glb_dir': scenario_glb_dir, # 场景 3D 素材
        'trip_info': trip_info, # 车辆统计信息
        'tls_state_add': tls_add, # 信号灯策略
        'use_gui':True,
    }
    env = SubprocVecEnv([make_env(**params) for _ in range(1)])
    env = VecNormalize.load(load_path=path_convert(f'../rl_tsc/results/{SCENARIO_NAME}/models/last_vec_normalize.pkl'), venv=env)
    env.training = False # 测试的时候不要更新
    env.norm_reward = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = path_convert(f'../rl_tsc/results/{SCENARIO_NAME}/models/last_rl_model.zip')
    model = PPO.load(model_path, env=env, device=device)

    # Simulation with environment
    obs = env.reset()
    seneor_data = None # 初始传感器数据为空, 
    dones = False # 默认是 False

    while not dones:
        action, _state = model.predict(obs, deterministic=True) # RL 获得的动作
        rl_agent.update_rl_traffic_phase(new_phase=action) # 更新当前 RL 推荐的动作

        if seneor_data is None:
            obs, rewards, dones, infos = env.step(action)

            # ######################
            # 获得并保存传感器的数据
            # ######################
            seneor_data = infos[0]['pixel']
            for phase_index in range(PHASE_NUMBER):
                image_path = path_convert(f"./{JUNCTION_NAME}_{phase_index}.jpg") # 保存的图像数据
                camera_data = seneor_data[f"{JUNCTION_NAME}_{phase_index}"]['junction_front_all']
                cv2.imwrite(image_path, convert_rgb_to_bgr(camera_data))
                
        else:
            # (1) 保存传感器数据; (2) 场景图片理解; (3) 将图像询问 agents; (3) 使用这里的 decision 与环境交互

            # ########################
            # (1) 获得并保存传感器的数据
            # ########################
            seneor_data = infos[0]['pixel']
            for phase_index in range(PHASE_NUMBER):
                image_path = path_convert(f"./{JUNCTION_NAME}_{phase_index}.jpg") # 保存的图像数据
                camera_data = seneor_data[f"{JUNCTION_NAME}_{phase_index}"]['junction_front_all']
                cv2.imwrite(image_path, convert_rgb_to_bgr(camera_data))
            
            # ###############
            # (2) 场景图片理解
            # ###############
            junction_mem = {} # 分别记录多个路口的信息 (一个路口不同方向的信息)
            for scene_index in range(PHASE_NUMBER):
                messages = [] # 对话的历史信息, 这里不同方向是独立的
                image_path = path_convert(f"./{JUNCTION_NAME}_{scene_index}.jpg") # 保存的图像数据
                # 构造多模态输入
                content = [{"text": "The picture shows a traffic intersection camera. Please describe the current road conditions, including the degree of congestion, and whether there are any special vehicles. If there are no obvious signs, they are ordinary vehicles. Only point out special vehicles when you are absolutely certain. All other vehicles are considered ordinary. If the vehicles are far from the intersection, they do not need to be considered. If there are special vehicles, you need to further determine whether the vehicle is entering the intersection or exiting it, and whether it has passed the stop line. Only vehicles that are entering and are within the intersection need to be considered."}]
                content.append({'image': image_path})
                messages.append({
                    'role': 'user',
                    'content': content  # 多模态的输入
                })  # Append the user query to the chat history.

                # AI Agents 对图片进行解析
                response = []
                response_plain_text = ''
                print(f'-->Scene Understand Agent response for Scene-{scene_index}:')
                for response in scene_understanding_agent.run(messages=messages):
                    response_plain_text = typewriter_print(response, response_plain_text)
                print('\n---------\n')
                # Append the bot responses to the chat history.
                junction_mem[scene_index] = response[0]['content'] # 每个方向的信息

            # ##################
            # (3) 场景信息分析
            # ##################
            junction_description = ""
            for scene_index, scene_text in junction_mem.items():
                traffic_phase_index = SENSOR_INDEX_2_PHASE_INDEX[scene_index] # 传感器 Index 转换为 Traffic Phase Index
                junction_description += f"The traffic situation corresponding to Traffic Phase-{traffic_phase_index} is: {scene_text}\n"
            
            # 构建询问的信息
            messages = []
            messages.append({
                'role': 'user',
                'content': f"The following are the descriptions of {PHASE_NUMBER} Traffic Phases at an intersection respectively. Please summarize the information of each Traffic Phase according to the detailed descriptions, focusing only on summarizing the congestion situation and whether there are special vehicles, and make it concise. If the vehicle type cannot be identified, it is defaulted to be an ordinary vehicle. Suspected special vehicles are not needed. " + \
                    f"The following are the detailed descriptions of each Traffic Phase currently: \n{junction_description}"
            })
            response = []
            response_plain_text = ''
            print(f'-->Scene Analysis Agent response:')
            for response in scene_analysis_agent.run(messages=messages):
                response_plain_text = typewriter_print(response, response_plain_text)

            # ###################
            # (4) 根据场景进行决策
            # ###################
            messages = []
            messages.append({
                'role': 'user',
                'content': f"Please make decisions based on the status of each current Traffic Phase to control the traffic lights. The information of the current Traffic Phase is as follows: \n{response[0]['content']}。"
            })
            response = []
            response_plain_text = ''
            print(f'\n-->Group Decision Agent response:')
            for response in group_decision_bots.run(messages=messages, max_round=1):
                response_plain_text = typewriter_print(response, response_plain_text)

            # (5) response -> action
            response_dict = json.loads(response[-1]['content'])
            decision, explanation = response_dict['decision'], response_dict['explanation']

            action = extract_action(decision) # 决策转换为动作
            obs, rewards, dones, infos = env.step(action)

    env.close()