'''
Author: Maonan Wang
Date: 2025-04-23 18:14:36
LastEditTime: 2025-04-30 17:53:33
LastEditors: Maonan Wang
Description: VLM Agent, Agents 介绍
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

from utils.llm_config import (
    vlm_cfg, 
    llm_cfg,
    llm_cfg_json
)

# 输出模板
example_response = json.dumps(
    {
        "decision": "Phase-2",
        "explanation": "The picture shows a simple road intersection scene, and there are no obvious signs of special vehicles. Therefore, based on the existing information, Phase-2 is the most congested, so set Phase-2 to green. ",
    },
    ensure_ascii=False
)

# 场景理解
scene_understanding_agent = Assistant(
    name='Scene Understanding Agent',
    description='You play the role of a police officer directing traffic at an intersection. You receive the camera data from a certain direction of the crossroads and need to describe the information of the intersection, including elements of the scene such as the degree of congestion and special events (such as ambulances, police cars). ',
    llm=vlm_cfg,
    system_message="You play the role of a police officer directing traffic at an intersection. You receive the camera data from a certain direction of the crossroads and need to describe the information of the intersection, including elements of the scene such as the degree of congestion and special events (such as ambulances, police cars). " + \
        "If there are no obvious signs, they are ordinary vehicles. Only point out the vehicles as special ones when you are absolutely certain. "
)

# 场景分析&总结 (将多个摄像头结果和 Traffic Phase 联合起来)
scene_analysis_agent = Assistant(
    llm=llm_cfg,
    system_message="Now you are playing the role of a police officer directing traffic at an intersection. You will receive descriptions of multiple intersections. First, please associate the intersection descriptions with the traffic phases, and then summarize the situation of each traffic phase. For example, the congestion situation of each phase and whether there are special vehicles. "
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
                "explanation": "In the conventional scenario, use the decisions recommended by reinforcement learning.",
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
    description='For conventional traffic scenarios, directly call the pre-trained reinforcement learning model.',
    system_message="In the conventional scenario, directly provide the decision recommended by reinforcement learning."
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
            system_message='You play the role of a police officer directing traffic at an intersection. You have the traffic situation of each current phase. Please make a decision based on the current scene.' + \
                'Your goal is to give priority to the passage of special vehicles (ambulances, police cars, fire engines). ' + \
                'If there are no special vehicles, the goal is to minimize the overall queue length at the intersection.'
        )

        # 检查给出的动作是否符合要求
        self.check_agent = Assistant(
            llm=llm_cfg_json,
            system_message=f'Now you need to determine whether the given decision meets the requirements. The compliant actions can only be {self.avaliable_traffic_phase}. ' + \
                'If it is compliant, the output should be in JSON format with two keys, namely decision and explanation, and no other keys should be used. ' + \
                    'Among them, decision returns the selected Phase ID, and explanation provides an explanation for the decision. An example is as follows: \n{example_response}.",'
        )

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        """Define the workflow
        """
        # Decision
        new_messages = copy.deepcopy(messages) # 上下文记忆
        new_messages.append(
            Message(
                'user',
                [ContentItem(text='Please make a decision based on the status of each Traffic Phase. The rules are as follows: If there are special vehicles, such as police cars, ambulances or fire engines, give priority to this Traffic Phase; if there are no special vehicles, turn the traffic light green for the direction with more vehicles.')]
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
                [ContentItem(text=f'Please make a decision based on the current scenario and return the data in JSON format. The keys in the JSON are `decision` and `explanation` respectively. The `decision` should include the Traffic Phase ID.')]
            ))
        for rsp in self.check_agent.run(new_messages, lang=lang, **kwargs):
            yield response + rsp