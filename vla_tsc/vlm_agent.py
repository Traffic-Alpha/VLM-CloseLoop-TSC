'''
Author: Maonan Wang
Date: 2025-04-23 18:14:36
LastEditTime: 2025-04-29 13:03:50
LastEditors: Maonan Wang
Description: VLM Agent, Agents 介绍
+ Scene Understanding Agent, 对每一个路口进行描述
+ Analysis Agent, 将 N 个方向的信息转换为 Traffic Phase 的信息, 并进行 summay
+ Group Decision Agents, 不同场景决策 Agents
    + RL Agent, 常规场景下使用 RL 来做出决策
    + Concern Case Agent, 特殊场景进行决策 (决策 + 验证)
FilePath: /VLM-CloseLoop-TSC/vla_tsc/vlm_agent.py
'''
import copy
import json
from qwen_agent import Agent
from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message, ContentItem

from typing import Dict, Iterator, List

from utils.llm_config import (
    vlm_cfg, 
    llm_cfg,
    llm_cfg_json
)

# 输出模板
example_response = json.dumps(
    {
        "decision": "Phase-2",
        "explanation": "图片显示的是一个简单的道路交叉口场景，没有明显的特殊车辆标志。因此，根据现有信息，Phase-2 最拥堵，因此将 Phase-2 设置为绿灯。",
    },
    ensure_ascii=False
)

# 场景理解
scene_understanding_agent = Assistant(
    name='Scene Understanding Agent',
    description='你扮演一个在路口指挥交通的警察负。你接收十字路口某个方向的摄像头数据，需要描述路口信息，包括拥堵程度、特殊事件（如救护车、警车）等场景要素。' + \
        '如果车辆顶部存在红蓝灯光，则为警车。如果没有明显的标志，则为普通车辆。',
    llm=vlm_cfg,
    system_message="你扮演一个在路口指挥交通的警察负。你接收十字路口某个方向的摄像头数据，需要描述路口信息，包括拥堵程度、特殊事件（如救护车、警车）等场景要素。"
)

# 场景分析&总结 (将多个摄像头结果和 Traffic Phase 联合起来)
scene_analysis_agent = Assistant(
    llm=llm_cfg,
    system_message="现在你扮演一个路口指挥交通的警察，你会收到多个路口的描述，请你首先将路口描述和交通相位关联，接着总结每一个交通相位的情况。例如每个相位的拥堵情况和是否存在特殊车辆。"
)

# RL Agent, 常规场景下, 返回强化学习的决策
class RLAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rl_traffic_phase = "Phase-1" # 初始相位

    def _run(self, messages, **kwargs):
        rl_response = json.dumps(
            {
                "decision": f"Phase-{self.rl_traffic_phase[0]}",
                "explanation": "常规场景下使用强化学习推荐的决策。",
            },
            ensure_ascii=False
        )
        yield [Message(role='assistant', content=rl_response, name=self.name)]

    def update_rl_traffic_phase(self, new_phase):
        """更新 RL 推荐的动作
        """
        self.rl_traffic_phase = new_phase

rl_agent = RLAgent(
    name='normal case decision agent',
    description='针对常规交通场景，直接调用预训练的强化学习模型。',
    system_message="在常规场景下，直接给出强化学习推荐的决策。"
)

# Concer Case Agent (包含 3 个 agent, 以此作决策)
class ConcernCaseAgent(Agent):
    """常规场景下决策的 Agents
    """
    def __init__(
            self, phase_num:int, llm_cfg:Dict, 
            *args, **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.avaliable_traffic_phase = [f"Phase-{i}" for i in range(phase_num)]

        # 根据 Traffic Phase 的结果给出决策
        self.decision_agent = Assistant(
            llm=llm_cfg,
            system_message='你扮演一个在路口指挥交通的警察，你有当前每一个相位的交通情况，请你根据当前的场景给出决策。' + \
                '你的目标是优先让有特殊车辆（救护车、警车、消防车）通过；' + \
                '如果没有特殊车辆，则目标是路口整体的排队长度最短。'
        )

        # 检查给出的动作是否符合要求
        self.check_agent = Assistant(
            llm=llm_cfg_json,
            system_message=f'现在你需要判断给出的决策是否符合要求。合规的动作只能是 {self.avaliable_traffic_phase}' + \
                '如果合规，需要输出为 JSON 格式，两个 key 分别是 `decision` 和 `explanation`，不能使用其他的 key。' + \
                    '其中 decision 返回选择的 Phase ID，explanation 给出对决策的解释。示例如下：\n{example_response}。",'
        )

    def _run(self, messages: List[Message], lang: str = 'zh', **kwargs) -> Iterator[List[Message]]:
        """Define the workflow
        """
        # Decision
        new_messages = copy.deepcopy(messages) # 上下文记忆
        new_messages.append(
            Message(
                'user',
                [ContentItem(text='请你根据各个 Traffic Phase 的状况给出决策，规则如下：如果存在特殊车辆，例如警车或是救护车，则优先这个 Traffic Phase；如果不存在特殊车辆，则哪里车辆多就给绿灯。')]
            )
        ) # 添加新的问题
        response = []
        for rsp in self.decision_agent.run(new_messages):
            yield response + rsp
        response.extend(rsp)
        new_messages.extend(rsp)

        # Check
        new_messages.append(
            Message(
                'user', 
                [ContentItem(text=f'请你根据当前的场景来做出决策，返回的数据格式为 JSON，其中 key 分别是 `decision` 和 `explanation`，`decision` 需要包含 Traffic Phase ID。')]
            ))
        for rsp in self.check_agent.run(new_messages, lang=lang, **kwargs):
            yield response + rsp