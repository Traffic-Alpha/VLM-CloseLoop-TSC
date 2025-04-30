'''
Author: Maonan Wang
Date: 2025-04-30 10:07:40
LastEditTime: 2025-04-30 12:30:05
LastEditors: Maonan Wang
Description: Configs for TSC Envs
FilePath: /VLM-CloseLoop-TSC/rl_tsc/_config.py
'''
# 存储不同场景的配置信息
SCENARIO_CONFIGS = {
    "Hongkong_YMT": {
        "SCENARIO_NAME": "Hongkong_YMT",
        "NETFILE": "ymt_eval",
        "JUNCTION_NAME": "J1",
        "PHASE_NUMBER": 4
    },
    "SouthKorea_Songdo": {
        "SCENARIO_NAME": "SouthKorea_Songdo",
        "NETFILE": "songdo_eval",
        "JUNCTION_NAME": "J2",
        "PHASE_NUMBER": 4
    },
    "France_Massy": {
        "SCENARIO_NAME": "France_Massy",
        "NETFILE": "massy_eval",
        "JUNCTION_NAME": "INT1",
        "PHASE_NUMBER": 3
    }
}