'''
Author: Maonan Wang
Date: 2025-04-24 10:02:21
Description: LLM Config for AI Agent (Ollama)
LastEditors: WANG Maonan
LastEditTime: 2025-07-30 13:39:00
'''
llm_cfg = {
    'model': 'qwen3:4b',
    'model_type': 'oai',
    'model_server': 'http://localhost:11435/v1',
    'api_key': 'ollama',
}

llm_cfg_json = {
    'model': 'qwen3:4b',
    'model_type': 'oai',
    'model_server': 'http://localhost:11435/v1',
    'api_key': 'ollama',

    'generate_cfg': {
        'top_p': 0.8,
        'response_format': {"type": "json_object"},
    }
} # Language Model

vlm_cfg = {
    'model': 'qwen2.5vl:3b',
    'model_type': 'qwenvl_oai',
    'model_server': 'http://localhost:11435/v1',
    'api_key': 'ollama',

    'generate_cfg': {
        'top_p': 0.8,
    }
}