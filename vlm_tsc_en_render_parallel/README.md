<!--
 * @Author: WANG Maonan
 * @Date: 2025-07-30 15:51:26
 * @LastEditors: WANG Maonan
 * @Description: 决策+渲染
 * @LastEditTime: 2025-07-30 16:09:02
-->
# VLMLight 决策与高精度渲染

## 文件说明

1. `vlmlight_decision.py` - 交通决策仿真主程序
2. `render_service.py` - Blender渲染服务脚本

## 运行步骤

### 1. 清理环境

```bash
# 删除场景文件夹（替换 YOUR_SCENE 为实际场景名，如 Hongkong_YMT）
rm -rf /YOUR_SCENE

# 终止所有Blender进程
pkill -9 blender
```

### 2. 启动仿真和渲染（需同时运行）

```bash
# 终端1：启动决策仿真
python vlmlight_decision.py

# 终端2：启动Blender渲染服务（替换 YOUR_SCENE 和路径）
blender /YOUR_SCENE/env.blend \
  --background \
  --python render_service.py -- \
  --monitor_path /YOUR_SCENE
```

### 3. 验证进程
```bash
# 检查Blender进程是否运行
ps -ef | grep blender

# 检查Python进程
ps -ef | grep python
```

## 自动化脚本

创建 `run_simulation.sh` 并添加以下内容：
```bash
#!/bin/bash

SCENE="Hongkong_YMT"  # 修改为你的场景名称
RENDER_PATH="/VLMLight/vlm_tsc_en_render_parallel/$SCENE"
BLEND_FILE="/VLMLight/sim_envs/$SCENE/env.blend"

# 清理环境
rm -rf $RENDER_PATH
pkill -9 blender

# 启动决策系统
python vlmlight_decision.py &

# 启动渲染服务
blender $BLEND_FILE --background --python render_service.py -- --monitor_path $RENDER_PATH
```

运行脚本：
```bash
chmod +x run_simulation.sh
./run_simulation.sh
```

## 工作原理

1. **文件监控机制**：`render_service.py` 监控 `monitor_path` 目录的文件变化
2. **进程协作**：
   - 决策进程生成指令文件
   - 渲染进程检测到新文件后执行渲染
   - 渲染完成后生成结果文件
   - 决策进程读取结果后继续下一步