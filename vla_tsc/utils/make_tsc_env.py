'''
@Author: WANG Maonan
@Date: 2023-09-08 17:45:54
@Description: 创建 TSC Env + Wrapper
LastEditTime: 2025-04-23 15:34:21
'''
import gymnasium as gym
from typing import List
from utils.tsc_env import TSC3DEnvironment
from utils.tsc_wrapper import TSCEnvWrapper

def make_env(
        tls_id:str, num_seconds:int, number_phases:int, 
        sumo_cfg:str, scenario_glb_dir:str, use_gui:bool,
        trip_info:str=None, tls_state_add:List=None,
    ):
    def _init() -> gym.Env: 
        tsc_scenario = TSC3DEnvironment(
            sumo_cfg=sumo_cfg, 
            scenario_glb_dir=scenario_glb_dir,
            num_seconds=num_seconds,
            trip_info=trip_info,
            tls_state_add=tls_state_add,
            tls_ids=[tls_id], 
            tls_action_type='choose_next_phase',
            use_gui=use_gui,
        )
        tsc_wrapper = TSCEnvWrapper(tsc_scenario, tls_id=tls_id, number_phases=number_phases)
        return tsc_wrapper
    
    return _init