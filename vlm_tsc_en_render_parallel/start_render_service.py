'''
Author: WANG Maonan
Date: 2025-07-30 14:57:42
LastEditors: WANG Maonan
Description: 启动渲染脚本开始渲染
LastEditTime: 2025-07-30 15:33:26
'''
import os
import time
import subprocess
import argparse

def path_convert(path):
    """转换路径为当前系统格式
    """
    return os.path.abspath(os.path.normpath(path))

def main():
    parser = argparse.ArgumentParser(description='启动渲染服务')
    parser.add_argument('--scenario', type=str, default="Hongkong_YMT", help='场景名称')
    args = parser.parse_args()

    # 构建Blender命令
    blender_file = path_convert(f"../sim_envs/{args.scenario}/env.blend")
    render_script = path_convert("./render_service.py")
    monitor_path = path_convert(f"./{args.scenario}/")
    
    command = [
        'blender',
        blender_file, # blender 文件
        '--background',
        '--python',
        render_script, # 等待运行的脚本
        '--',
        '--monitor_path', monitor_path,
        '--interval', '2' # 检查的间隔事件
    ]
    
    try:
        print(f"启动Blender渲染服务，场景: {args.scenario}")
        print(f"blender 脚本: {render_script}")
        print(f"监控路径: {monitor_path}")
        time.sleep(5)
        # 启动长期运行的Blender进程
        process = subprocess.Popen(command)
        process.wait()  # 等待进程结束
        
        print("🎉 Blender渲染服务已退出")
    except Exception as e:
        print(f"🔥 启动渲染服务失败: {str(e)}")

if __name__ == '__main__':
    main()