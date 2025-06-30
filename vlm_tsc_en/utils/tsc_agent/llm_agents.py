'''
Author: Maonan Wang
Date: 2025-04-23 18:14:36
LastEditTime: 2025-06-30 15:23:49
LastEditors: WANG Maonan
Description: VLM Agent (EN), Agents 介绍
+ Scene Understanding Agent, 对每一个路口进行描述
+ Analysis Agent, 将 N 个方向的信息转换为 Traffic Phase 的信息, 并进行 summay
+ Group Decision Agents, 不同场景决策 Agents
    + RL Agent, 常规场景下使用 RL 来做出决策
    + Concern Case Agent, 特殊场景进行决策 (决策 + 验证)
FilePath: /VLM-CloseLoop-TSC/vla_tsc_en/vlm_agent.py
'''
import copy
import json
from qwen_agent import Agent
from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message, ContentItem

from typing import Dict, Iterator, List

from utils.tsc_agent.llm_config import (
    vlm_cfg, 
    llm_cfg,
    llm_cfg_json
)

# 输出模板
example_response = json.dumps(
    {
        "decision": "Phase-2",
        "explanation": "The image depicts a basic road intersection scenario with no visible special vehicle markings; based on the available information, Phase-2 shows the heaviest congestion and should therefore be assigned the green light to optimize traffic flow.",
    },
    ensure_ascii=False
)

# 场景理解
scene_understanding_agent = Assistant(
    name='Scene Understanding Agent',
    description="Act as a traffic police officer managing a cross intersection; you'll receive camera feed data from one direction of the intersection and need to describe the traffic conditions including congestion levels and special situations (such as ambulances or police cars) while focusing on clearly identifiable elements that require immediate attention for effective traffic management.",
    llm=vlm_cfg,
    system_message=(
        "You are now roleplaying as a traffic police officer responsible for monitoring a cross intersection; "
        "using real-time camera feed data from one direction of the intersection, you must accurately describe the traffic conditions including congestion levels and identify any special vehicles (such as ambulances, police trucks, fire truck). "
        "All vehicles without identification should be classified as regular vehicles, and only absolutely verifiable emergency vehicles should be reported to ensure reliable traffic management decisions."
    )
)

# 场景分析&总结 (将多个摄像头结果和 Traffic Phase 联合起来)
scene_analysis_agent = Assistant(
    llm=llm_cfg,
    system_message=(
        "You are now roleplaying as a traffic police officer managing multiple intersections; "
        "you will receive descriptions of these intersections and must first correlate each description with its corresponding traffic phase, "
        "then summarize the status of every traffic phase including congestion levels and confirmed presence of any special vehicles (only those with clearly identifiable markings), "
        "providing a concise overview of each phase's critical traffic conditions for effective decision-making."
    )
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
                "explanation": "Under routine traffic conditions, implement the decisions recommended by the reinforcement learning system to optimize traffic flow.",
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
    description='Under normal traffic conditions, implement the decisions recommended by the reinforcement learning mode.',
    system_message="Under routine traffic conditions, implement the decisions recommended by the reinforcement learning mode to optimize traffic flow while maintaining standard safety protocols."
)

# Concer Case Agent (包含 3 个 agent, 以此作决策)
class ConcernCaseAgent(Agent):
    """特殊场景下决策的 Agents
    """
    def __init__(
            self, phase_num:int, llm_cfg:Dict, 
            *args, **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.available_traffic_phase = [f"Phase-{i}" for i in range(phase_num)]

        # 根据 Traffic Phase 的结果给出决策
        self.decision_agent = Assistant(
            llm=llm_cfg,
            system_message=(
                "You are now roleplaying as a traffic police officer managing an intersection with real-time visibility into each traffic phase; "
                "make operational decisions based on current conditions by prioritizing passage for confirmed emergency vehicles (ambulances, police cars, or fire trucks with clear identifiers), "
                "and when no special vehicles are present, optimize traffic light timing to minimize intersection congestion through the most efficient flow distribution possible."
            )
        )

        # 检查给出的动作是否符合要求
        self.check_agent = Assistant(
            llm=llm_cfg_json,
            system_message=(
                "You must evaluate whether the given traffic decision complies with the specified requirements, "
                f"where valid actions can only be from {self.available_traffic_phase}; "
                "if compliant, output the result in JSON format strictly containing two keys—decision (indicating the selected Phase ID) and explanation (providing the rationale for the decision)—without any additional keys, "
                f"following the exact structure shown in this example: {example_response}."
            )
        )

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        """Define the workflow
        """
        # Decision
        new_messages = copy.deepcopy(messages) # 上下文记忆
        new_messages.append(
            Message(
                'user',
                [ContentItem(text="Make traffic signal decisions based on the status of each Traffic Phase following these rules: if any confirmed emergency vehicles (such as police cars, ambulances, or fire trucks with clear identifiers) are present in a phase, prioritize that phase; otherwise, autonomously analyze and determine the optimal phase sequencing to maintain smooth traffic flow.")]
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
                [ContentItem(
                    text=(
                        "Evaluate whether the current traffic decision is compliant, where valid actions must be from {self.available_traffic_phase}; "
                        "if compliant, return a JSON response containing exactly two keys: decision (specifying the Traffic Phase ID) and explanation (providing the rationale for the decision), with no additional keys permitted."
                    )
                )]
            ))
        for rsp in self.check_agent.run(new_messages, lang=lang, **kwargs):
            yield response + rsp