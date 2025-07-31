'''
Author: Maonan Wang
Date: 2025-05-09 11:41:49
LastEditTime: 2025-06-30 15:24:32
LastEditors: WANG Maonan
Description: Useful Tools for VLMLight
'''
import os
import json

def save_to_json(data, filename):
    """保存每一个时刻的 3D 场景数据
    """
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"数据已成功保存到 {filename}")
    except Exception as e:
        print(f"保存文件时出现错误: {e}")


def append_response_to_file(file_path: str, content: str) -> bool:
    """
    将回复内容添加到文件中（文件不存在则自动创建）
    
    参数:
        file_path (str): 文件路径
        content (str): 要写入的字符串内容 (agent 的回复)
    
    返回:
        bool: 是否成功写入 (True/False)
    """
    content += '\n-----\n\n\n\n\n'
    try:
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error writing to file: {e}")
        return False


def create_folder(folder_path):
    """
    创建新文件夹的函数
    
    参数:
        folder_path (str): 要创建的文件夹路径
    
    返回:
        str: 操作结果信息 ("Folder created successfully", "Folder already exists", 或错误信息)
    """
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return "Folder created successfully"
        else:
            return "Folder already exists"
    except Exception as e:
        return f"Error creating folder: {str(e)}"