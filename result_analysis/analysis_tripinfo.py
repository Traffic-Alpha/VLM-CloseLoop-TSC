'''
Author: Maonan Wang
Date: 2025-04-29 15:23:30
LastEditTime: 2025-05-13 16:01:55
LastEditors: Maonan Wang
Description: 分析 TripInfo 的结果
FilePath: /VLM-CloseLoop-TSC/result_analysis/analysis_tripinfo.py
'''
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.sumo_tools.analysis_output.tripinfo_analysis import TripInfoAnalysis

# 初始化日志
current_file_path = get_abs_path(__file__)
set_logger(current_file_path('./'), file_log_level="INFO")

METHOD = 'VLM'
SCENARIO_NAME = 'France_Massy' # Hongkong_YMT, SouthKorea_Songdo, France_Massy

tripinfo_file = current_file_path(f"./{METHOD}/{SCENARIO_NAME}/tripinfo.out.xml")
tripinfo_parser = TripInfoAnalysis(tripinfo_file)

# 所有车辆一起分析
stats = tripinfo_parser.calculate_multiple_stats(metrics=['duration', 'waitingTime', 'fuel_abs'])
TripInfoAnalysis.print_stats_as_table(stats)

# 按照车辆类型分析
print('-'*10)
vehicle_stats = tripinfo_parser.statistics_by_vehicle_type(metrics=['duration', 'waitingTime'])
print("==> Travel Time: -----------")
print(vehicle_stats['duration'])
print("==> Waiting Time: -----------")
print(vehicle_stats['waitingTime'])