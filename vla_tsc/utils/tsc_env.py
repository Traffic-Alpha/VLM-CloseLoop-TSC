'''
@Author: WANG Maonan
@Date: 2023-09-04 20:43:53
@Description: 信号灯控制环境 (3D)
LastEditTime: 2025-04-23 17:49:26
'''
import gymnasium as gym

from typing import List, Dict
from tshub.tshub_env3d.tshub_env3d import Tshub3DEnvironment

class TSC3DEnvironment(gym.Env):
    def __init__(self, 
                 sumo_cfg:str, scenario_glb_dir:str,
                 num_seconds:int, tls_ids:List[str], 
                 tls_action_type:str, use_gui:bool=False, 
                 trip_info:str=None, tls_state_add:List=None,
                ) -> None:
        super().__init__()

        self.tsc_env = Tshub3DEnvironment(
            sumo_cfg=sumo_cfg,
            is_aircraft_builder_initialized=False, 
            is_vehicle_builder_initialized=True, # 用于获得 vehicle 的 waiting time 来计算 reward
            is_traffic_light_builder_initialized=True,
            tls_ids=tls_ids, 
            trip_info=trip_info, # 输出 tripinfo
            tls_state_add=tls_state_add, # 输出信号灯变化
            num_seconds=num_seconds,
            tls_action_type=tls_action_type,
            use_gui=use_gui,
            is_libsumo=(not use_gui), # 如果不开界面, 就是用 libsumo
            # 用于渲染的参数
            scenario_glb_dir=scenario_glb_dir, # 场景 3D 素材
            render_mode="offscreen", # 如果设置了 use_render_pipeline, 此时只能是 onscreen
            debuger_print_node=False,
            debuger_spin_camera=False,
            sensor_config={
                'tls': {
                    tls_ids[0]:{
                        'sensor_types': ['junction_front_all'],
                        'tls_camera_height': 15,
                    } # 路口传感器
                }
            }
        )

    def reset(self):
        state_infos = self.tsc_env.reset()
        return state_infos
        
    def step(self, action:Dict[str, Dict[str, int]]):
        action = {'tls': action} # 这里只控制 tls 即可
        states, rewards, infos, dones, sensor_data = self.tsc_env.step(action)
        truncated = dones

        return states, rewards, truncated, dones, infos, sensor_data
    
    def close(self) -> None:
        self.tsc_env.close()