'''
@Author: WANG Maonan
@Date: 2023-09-08 17:45:54
@Description: 创建 TSC Env + Wrapper
LastEditTime: 2025-04-23 12:50:52
'''
import gymnasium as gym
from typing import List
from utils.tsc_env import TSCEnvironment
from utils.tsc_wrapper import TSCEnvWrapper
from stable_baselines3.common.monitor import Monitor

def make_env(
        tls_id:str, num_seconds:int, number_phases:int, 
        sumo_cfg:str, use_gui:bool,
        log_file:str, env_index:int, 
        trip_info:str=None, tls_state_add:List=None,
    ):
    def _init() -> gym.Env: 
        tsc_scenario = TSCEnvironment(
            sumo_cfg=sumo_cfg, 
            num_seconds=num_seconds,
            trip_info=trip_info,
            tls_state_add=tls_state_add,
            tls_ids=[tls_id], 
            tls_action_type='choose_next_phase',
            use_gui=use_gui,
        )
        tsc_wrapper = TSCEnvWrapper(tsc_scenario, tls_id=tls_id, number_phases=number_phases)
        return Monitor(tsc_wrapper, filename=f'{log_file}/{env_index}')
    
    return _init