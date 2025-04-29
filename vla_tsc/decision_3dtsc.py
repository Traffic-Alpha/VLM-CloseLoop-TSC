'''
Author: Maonan Wang
Date: 2025-04-23 15:13:54
LastEditTime: 2025-04-29 13:08:41
LastEditors: Maonan Wang
Description: VLA TSC (根据传感器图像进行决策)
FilePath: /VLM-CloseLoop-TSC/vla_tsc/decision_3dtsc.py
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
SCENARIO_NAME = "Hongkong_YMT" # 可视化场景
NETFILE = "ymt_eval" # sumocfg 文件, 加载 eval 文件
JUNCTION_NAME = "J1" # sumo net 对应的路口 ID
PHASE_NUMBER = 4 # 绿灯相位数量
SENSOR_INDEX_2_PHASE_INDEX = {
    0:2, 1:3, 2:0, 3:1
} # 传感器编号和 Traffic Phase 的对应关系

# Init Agents
concer_case_decision_agent = ConcernCaseAgent(
    name='concer case decision agent',
    description='你扮演一个在路口指挥交通的警察，当路口存在特殊车辆，例如警车、救护车和消防车等情况需要你来决策。',
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
                content = [{"text": "图片为路口摄像头，请你对当前道路进行描述，包括拥堵程度，以及是否存在特殊车辆。如果无法辨别车辆类型，默认是普通车辆。如果存在特殊车辆，你需要进一步判断车辆正在驶入路口还是驶出，是否已经经过了停车线。"}]
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
                junction_description += f"Traffic Phase-{traffic_phase_index} 对应的交通情况为：{scene_text}；\n"
            
            # 构建询问的信息
            messages = []
            messages.append({
                'role': 'user',
                'content': f"下面分别是一个十字路口中 {PHASE_NUMBER} 个 Traffic Phase 的描述，请你根据详细的描述总结每个 Traffic Phase 的信息，只需要总结拥堵情况和是否有特殊车辆，需要概况。如果无法辨别车辆类型，默认是普通车辆。不需要疑似特殊车辆。" + \
                    f"下面是当前每一个 Traffic Phase 详细的描述：\n{junction_description}"
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
                'content': f"请你根据当前每一个 Traffic Phase 的状态，做出决策，从而来控制信号灯。当前 Traffic Phase 的信息如下：\n{response[0]['content']}。"
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