'''
Author: Maonan Wang
Date: 2025-04-24 10:05:52
LastEditTime: 2025-04-24 13:03:18
LastEditors: Maonan Wang
Description: 刷新显示传感器图像
FilePath: /VLM-CloseLoop-TSC/vla_tsc/image_display.py
'''
import cv2
import time

from tshub.utils.get_abs_path import get_abs_path
path_convert = get_abs_path(__file__)

JUNCTION_NAME = "J1" # sumo net 对应的路口 ID
PHASE_NUMBER = 4 # 绿灯相位数量


# 图片文件路径列表
image_paths = [f"{JUNCTION_NAME}_{phase_index}.jpg" for phase_index in range(PHASE_NUMBER)]
N = 5 # 刷新间隔时间（秒）

while True:
    images = []
    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        # 在图像上添加文本标注
        cv2.putText(img, f'Phase {i + 1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        images.append(img)

    # 创建一个空白画布
    combined_image = cv2.vconcat([
        cv2.hconcat([images[0], images[1]]),
        cv2.hconcat([images[2], images[3]])
    ])

    # 显示组合后的图像
    cv2.imshow('Combined Images', combined_image)

    # 等待 N 秒
    time.sleep(N)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 关闭所有窗口
cv2.destroyAllWindows()