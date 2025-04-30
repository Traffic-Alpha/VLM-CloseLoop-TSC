'''
Author: Maonan Wang
Date: 2024-12-30 16:00:33
LastEditTime: 2025-04-21 15:36:14
LastEditors: Maonan Wang
Description: Generate Routes for Hongkong YMT
FilePath: /VLM-CloseLoop-TSC/sim_envs/Hongkong_YMT/generate_routes.py
'''
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.sumo_tools.generate_routes import generate_route

# 初始化日志
current_file_path = get_abs_path(__file__)
set_logger(current_file_path('./'), file_log_level='WARNING', terminal_log_level='INFO')

# 开启仿真 --> 指定 net 文件
sumo_net = current_file_path("./env/YMT.net.xml")

generate_route(
    sumo_net=sumo_net,
    interval=[1,1,1,1,1,1,1], 
    edge_flow_per_minute={
        '102454134#0': [10, 8, 15, 7, 5, 12, 9],
        '1200878753#0': [10, 15, 8, 10, 9, 11, 10],
        '30658263#0': [15, 15, 10, 2, 10, 6, 10],
        '960661806#0': [6, 5, 8, 15, 12, 11, 5],
    }, # 每分钟每个 edge 有多少车
    edge_turndef={},
    veh_type={
        'background': {'color':'220,220,220', 'length': 5, 'probability':0.97},
        'police': {'color':'0,0,255', 'length': 5, 'probability':0.02},
        'emergency': {'color':'255,165,0', 'length': 6.5, 'probability':0.01},
    },
    output_trip=current_file_path('./testflow.trip.xml'),
    output_turndef=current_file_path('./testflow.turndefs.xml'),
    output_route=current_file_path('./env/YMT.rou.xml'),
    interpolate_flow=False,
    interpolate_turndef=False,
)