'''
Author: Maonan Wang
Date: 2025-04-24 10:02:21
LastEditTime: 2025-06-30 15:36:48
LastEditors: WANG Maonan
Description: LLM 配置
FilePath: /VLM-CloseLoop-TSC/vla_tsc/utils/llm_config.py
'''
llm_cfg = {
    'model': 'Qwen/Qwen2.5-72B-Instruct-AWQ',
    'model_type': 'oai',
    'model_server': 'http://localhost:5070/v1',
    'api_key': 'token-abc123',

    'generate_cfg': {
        'top_p': 0.8,
    }
} # Language Model

llm_cfg_json = {
    'model': 'Qwen/Qwen2.5-72B-Instruct-AWQ',
    'model_type': 'oai',
    'model_server': 'http://localhost:5070/v1',
    'api_key': 'token-abc123',

    'generate_cfg': {
        'top_p': 0.8,
        'response_format': {"type": "json_object"},
    }
} # Language Model

vlm_cfg = {
    'model': 'Qwen/Qwen2.5-VL-32B-Instruct-AWQ',
    'model_type': 'qwenvl_oai',
    'model_server': 'http://localhost:5030/v1',
    'api_key': 'token-abc123',

    'generate_cfg': {
        'top_p': 0.8,
    }
} # Vision Language Model