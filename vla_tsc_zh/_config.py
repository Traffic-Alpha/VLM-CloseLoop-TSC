'''
Author: Maonan Wang
Date: 2025-04-30 16:32:45
LastEditTime: 2025-04-30 17:31:09
LastEditors: Maonan Wang
Description: Configs for VLA
FilePath: /VLM-CloseLoop-TSC/vla_tsc/_config.py
'''
# 存储不同场景的配置信息
SCENARIO_CONFIGS = {
    "Hongkong_YMT": {
        "SCENARIO_NAME": "Hongkong_YMT",
        "NETFILE": "ymt_eval",
        "JUNCTION_NAME": "J1",
        "PHASE_NUMBER": 4,
        "CENTER_COORDINATES": (172, 201, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:0, 3:1}
    },
    "SouthKorea_Songdo": {
        "SCENARIO_NAME": "SouthKorea_Songdo",
        "NETFILE": "songdo_eval",
        "JUNCTION_NAME": "J2",
        "PHASE_NUMBER": 4,
        "CENTER_COORDINATES": (900, 1641, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:1, 3:0}
    },
    "France_Massy": {
        "SCENARIO_NAME": "France_Massy",
        "NETFILE": "massy_eval",
        "JUNCTION_NAME": "INT1",
        "PHASE_NUMBER": 3,
        "CENTER_COORDINATES": (173, 244, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:1, 2:0}
    }
}