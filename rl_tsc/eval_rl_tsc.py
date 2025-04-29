'''
Author: Maonan Wang
Date: 2025-04-22 14:20:29
LastEditTime: 2025-04-23 18:28:27
LastEditors: Maonan Wang
Description: 测试单路口智能体并保存 SUMO Outpuut
FilePath: /VLM-CloseLoop-TSC/rl_tsc/eval_rl_tsc.py
'''
import torch
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

from utils.make_tsc_env import make_env

path_convert = get_abs_path(__file__)
logger.remove()

# 全局变量
SCENARIO_NAME = "Hongkong_YMT" # 可视化场景
NETFILE = "ymt_eval" # sumocfg 文件, 加载 eval 文件
JUNCTION_NAME = "J1" # sumo net 对应的路口 ID
PHASE_NUMBER = 4 # 绿灯相位数量

if __name__ == '__main__':
    # #########
    # Init Env
    # #########
    sumo_cfg = path_convert(f"../sim_envs/{SCENARIO_NAME}/{NETFILE}.sumocfg")
    trip_info = path_convert(f"./results/{SCENARIO_NAME}/sumo_outputs/tripinfo.out.xml")
    tls_add = [
        path_convert(f'../sim_envs/{SCENARIO_NAME}/add/e2.add.xml'), # 探测器
        path_convert(f'../sim_envs/{SCENARIO_NAME}/add/tls_programs.add.xml'), # 信号灯
    ]
    params = {
        'tls_id':JUNCTION_NAME,
        'num_seconds':600,
        'number_phases':PHASE_NUMBER,
        'sumo_cfg':sumo_cfg,
        'trip_info': trip_info, # 车辆统计信息
        'tls_state_add': tls_add, # 信号灯策略
        'use_gui':True,
        'log_file':path_convert(f"./eval_{SCENARIO_NAME}.log"),
    }
    env = SubprocVecEnv([make_env(env_index=f'{i}', **params) for i in range(1)])
    env = VecNormalize.load(load_path=path_convert(f'./results/{SCENARIO_NAME}/models/last_vec_normalize.pkl'), venv=env)
    env.training = False # 测试的时候不要更新
    env.norm_reward = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = path_convert(f'./results/{SCENARIO_NAME}/models/last_rl_model.zip')
    model = PPO.load(model_path, env=env, device=device)

    # 使用模型进行测试
    obs = env.reset()
    dones = False # 默认是 False
    total_reward = 0

    while not dones:
        action, _state = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        total_reward += rewards
        
    env.close()
    print(f'累积奖励为, {total_reward}.')