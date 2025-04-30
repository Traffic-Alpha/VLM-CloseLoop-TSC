'''
Author: Maonan Wang
Date: 2025-04-29 15:42:24
LastEditTime: 2025-04-30 13:07:08
LastEditors: Maonan Wang
Description: FixTime Control
FilePath: /VLM-CloseLoop-TSC/rl_tsc/ft_tsc.py
'''
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path

from utils.make_tsc_env import make_env
from _config import SCENARIO_CONFIGS

path_convert = get_abs_path(__file__)
logger.remove()

# 全局变量
scenario_key = "Hongkong_YMT" # Hongkong_YMT, SouthKorea_Songdo, France_Massy
config = SCENARIO_CONFIGS.get(scenario_key) # 获取特定场景的配置
SCENARIO_NAME = config["SCENARIO_NAME"] # 场景名称
NETFILE = config["NETFILE"] # sumocfg 文件, 加载 eval 文件
JUNCTION_NAME = config["JUNCTION_NAME"] # sumo net 对应的路口 ID
PHASE_NUMBER = config["PHASE_NUMBER"] # 绿灯相位数量
REPEAT_NUMBER = 6 # 相位重复次数

if __name__ == '__main__':
    # #########
    # Init Env
    # #########
    sumo_cfg = path_convert(f"../sim_envs/{SCENARIO_NAME}/{NETFILE}.sumocfg")
    trip_info = path_convert(f"./tripinfo.out.xml") # 保存 TripInfo 在当前目录下
    tls_add = [
        path_convert(f'../sim_envs/{SCENARIO_NAME}/add/e2.add.xml'), # 探测器
        path_convert(f'../sim_envs/{SCENARIO_NAME}/add/tls_programs.add.xml'), # 信号灯
    ]
    params = {
        'tls_id':JUNCTION_NAME,
        'num_seconds':800,
        'number_phases':PHASE_NUMBER,
        'sumo_cfg':sumo_cfg,
        'trip_info': trip_info, # 车辆统计信息
        'tls_state_add': tls_add, # 信号灯策略
        'use_gui':True,
        'log_file':path_convert(f"./eval_{SCENARIO_NAME}.log"),
    }
    env = make_env(env_index="1", **params)()

    # 使用模型进行测试
    obs = env.reset()
    dones = False # 默认是 False
    total_reward = 0

    index = 0
    while not dones:
        action = (index // REPEAT_NUMBER) % PHASE_NUMBER # 每间隔 4 个进行相位的切换
        obs, rewards, truncated, dones, infos = env.step(action)
        index += 1
        
    env.close()