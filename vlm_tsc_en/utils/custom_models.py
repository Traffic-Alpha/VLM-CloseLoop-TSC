'''
@Author: WANG Maonan
@Date: 2023-09-08 18:34:24
@Description: Custom Model
LastEditTime: 2025-06-30 15:25:03
'''
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomModel(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 16):
        """特征提取网络（Transformer版本）
        """
        super().__init__(observation_space, features_dim)
        net_shape = observation_space.shape[-1]  # 12

        # 输入嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(net_shape, 32),  # 12 -> 32
            nn.ReLU(),
        )  # 5*12 -> 5*32

        # Transformer部分
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,  # 每个时刻的特征维度
            nhead=4,  # 多头注意力机制，头数
            dim_feedforward=64,  # 前馈网络的维度
            dropout=0.1  # dropout率
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=2  # 使用2层TransformerEncoder
        )

        # 输出层
        self.output = nn.Sequential(
            nn.Linear(32, 64),  # Transformer的输出维度
            nn.ReLU(),
            nn.Linear(64, features_dim)  # 输出期望的特征维度
        )

    def forward(self, observations):
        # 通过嵌入层映射特征
        embedding = self.embedding(observations)  # batchsize*5*32

        # 调整维度为Transformer期望的输入格式 (seq_len, batch, input_size)
        embedding = embedding.permute(1, 0, 2)  # batchsize*5*32 -> 5*batchsize*32

        # 使用Transformer进行时序特征提取
        transformer_output = self.transformer_encoder(embedding)

        # 获取Transformer输出的最后一个时间步的隐状态
        output = transformer_output[-1, :, :]  # 获取最后一个时间步 (32维)

        # 通过输出层进行处理
        output = self.output(output)

        return output