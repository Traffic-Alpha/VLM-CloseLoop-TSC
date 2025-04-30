'''
Author: Maonan Wang
Date: 2025-04-29 15:58:39
LastEditTime: 2025-04-30 12:46:12
LastEditors: Maonan Wang
Description: 最大占有率策略
FilePath: /VLM-CloseLoop-TSC/rl_tsc/max_occ.py
'''
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path

from utils.make_tsc_env import make_env
from _config import SCENARIO_CONFIGS

path_convert = get_abs_path(__file__)
logger.remove()

# 全局变量
scenario_key = "France_Massy" # Hongkong_YMT, SouthKorea_Songdo, France_Massy
config = SCENARIO_CONFIGS.get(scenario_key) # 获取特定场景的配置
SCENARIO_NAME = config["SCENARIO_NAME"] # 场景名称
NETFILE = config["NETFILE"] # sumocfg 文件, 加载 eval 文件
JUNCTION_NAME = config["JUNCTION_NAME"] # sumo net 对应的路口 ID
PHASE_NUMBER = config["PHASE_NUMBER"] # 绿灯相位数量

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
    infos = {'step_time': 5.0, 'phase_occ': {_key:0 for _key in range(PHASE_NUMBER)}} # 初始化 info

    while not dones:
        phase_occ = infos['phase_occ']
        action = max(phase_occ, key=phase_occ.get) # 找出最大占有的相位
        obs, rewards, truncated, dones, infos = env.step(action)
        
    env.close()