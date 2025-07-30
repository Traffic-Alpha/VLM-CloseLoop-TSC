'''
Author: WANG Maonan
Date: 2025-07-29 17:53:18
LastEditors: WANG Maonan
Description: 渲染场景, 启动命令为
blender /home/tshub/Code_Project/2_Traffic/TrafficAlpha/VLMLight/sim_envs/Hongkong_YMT/env.blend --background --python render_service.py -- --monitor_path /home/tshub/Code_Project/2_Traffic/TrafficAlpha/VLMLight/vlm_tsc_en_render_parallel/Hongkong_YMT 
LastEditTime: 2025-07-30 15:52:50
'''
import os
import sys
import bpy
import time
import json
import argparse

tshub_path = "/home/tshub/Code_Project/2_Traffic/TransSimHub/"
sys.path.insert(0, tshub_path + "tshub/tshub_env3d/")

from vis3d_blender_render import TimestepRenderer, VehicleManager
MODELS_BASE_PATH = f"{tshub_path}/tshub/tshub_env3d/_assets_3d/vehicles_high_poly/"

def parse_args():
    parser = argparse.ArgumentParser(description='持续渲染服务')
    parser.add_argument('--monitor_path', type=str, default="./", help='监控路径')
    parser.add_argument('--interval', type=float, default=2, help='检查间隔(秒)')
    return parser.parse_args()

class RenderService:
    def __init__(self, monitor_path):
        self.monitor_path = os.path.abspath(monitor_path)
        
        # 一次性初始化资源
        print("🚀 初始化渲染服务...")
        self.vehicle_mgr = VehicleManager(MODELS_BASE_PATH)
        self.renderer = TimestepRenderer(
            resolution=480,
            render_mask=False, 
            render_depth=False
        )
        print("✅ 渲染服务初始化完成")
        
        # 性能统计
        self.total_timesteps = 0
        self.total_render_time = 0
        self.start_time = time.time()
    
    def find_tasks(self):
        """查找需要渲染的任务
        """
        for root, dirs, files in os.walk(self.monitor_path):
            for file in files:
                if file == '.ready':
                    ready_file = os.path.join(root, file)
                    return ready_file
    
    def process_task(self, ready_file):
        """处理单个渲染任务
        """
        try:
            # 读取任务信息
            with open(ready_file, 'r') as f:
                task_info = json.load(f)
            
            timestep_path = task_info['timestep_path'] # 获得时间步
            json_path = os.path.join(self.monitor_path, timestep_path, "data.json") # 通过事件步组合成 json 文件
            
            # 检查数据完整性
            if not os.path.exists(json_path):
                print(f"⚠️ JSON文件不存在: {json_path}")
                return False
            
            # 开始渲染
            print(f"\n🕒 开始渲染: {timestep_path}")
            timestep_start = time.time()
            
            # 加载车辆
            vehicles = self.vehicle_mgr.load_vehicles(json_path)
            if not vehicles:
                print(f"⚠️ 未加载车辆: {timestep_path}")
            
            # 渲染时间步
            self.renderer.render_timestep(timestep_path)
            
            # 计算并记录性能
            timestep_elapsed = time.time() - timestep_start
            self.total_timesteps += 1
            self.total_render_time += timestep_elapsed
            
            print(f"✅ 渲染完成: {timestep_path} (耗时: {timestep_elapsed:.2f}秒)")
            
            # 标记任务完成
            os.remove(ready_file) # 删除就绪标记
            done_file = os.path.join(self.monitor_path, '.done') # 新建完成任务的标记
            with open(done_file, 'w') as f:
                f.write('done')
            
            return True
        except Exception as e:
            print(f"🔥 渲染错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self, interval=0.5):
        """运行渲染服务主循环
        """
        print(f"👀 启动渲染监控，路径: {self.monitor_path}")
        print("等待渲染任务... (Ctrl+C 退出)")

        try:
            while True:
                ready_file = self.find_tasks()
                if ready_file:
                    self.process_task(ready_file)
                else:
                    # 没有任务时休眠
                    time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n🛑 收到停止信号，结束渲染服务...")
        finally:
            # 打印性能报告
            total_time = time.time() - self.start_time
            avg_render_time = self.total_render_time / self.total_timesteps if self.total_timesteps > 0 else 0
            
            print("\n📊 渲染性能报告:")
            print(f"  总渲染时间步: {self.total_timesteps}")
            print(f"  总耗时: {total_time:.2f}秒")
            print(f"  平均每步渲染时间: {avg_render_time:.2f}秒")
            print(f"  渲染效率: {self.total_timesteps / total_time:.2f} 步/秒")
            print("🎉 渲染服务已停止")


def main():
    # 处理Blender参数
    if '--' in sys.argv:
        args = sys.argv[sys.argv.index('--') + 1:]
        sys.argv = [sys.argv[0]] + args
    
    args = parse_args()
    
    # 创建并启动渲染服务
    service = RenderService(
        monitor_path=args.monitor_path
    )
    service.run(interval=args.interval)

if __name__ == '__main__':
    main()