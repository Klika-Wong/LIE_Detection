import torch
import torch.nn as nn
import torchvision.models as models

class VideoModel(nn.Module):
    def __init__(self,
                 pretrained=True,
                 feature_dim=128,
                 lstm_hidden=64,
                 lstm_layers=1,
                 bidirectional=False,
                 freeze_resnet=True,
                 dropout=0.3):
        """
        视频模型，输入为视频帧序列，输出为特征embedding
        Args:
            pretrained: 是否使用预训练ResNet
            feature_dim: 特征维度
            lstm_hidden: LSTM隐藏维度
            lstm_layers: LSTM层数
            bidirectional: 是否使用双向LSTM
            freeze_resnet: 是否冻结CNN参数
            dropout: dropout概率
        """
        super(VideoModel, self).__init__()
        self.output_dim = feature_dim
        self.feature_dim = feature_dim
        self.bidirectional = bidirectional

        # 使用 ResNet18 提取视觉特征
        resnet = models.resnet18(pretrained=pretrained)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # 去除最后fc层 (512-dim)

        if freeze_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # 映射到统一的特征维度
        self.fc = nn.Linear(512, feature_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # LSTM 进行时序建模
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_output_dim = lstm_hidden * (2 if bidirectional else 1)

        # 输出映射回 feature_dim
        self.output_fc = nn.Linear(lstm_output_dim, feature_dim)
        self.layernorm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, C, H, W)
        输出: (batch, feature_dim)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # 合并batch和时间维度

        features = self.resnet(x)  # (B*T, 512, 1, 1)
        features = features.view(B * T, -1)  # (B*T, 512)
        features = self.fc(features)  # (B*T, feature_dim)
        features = self.relu(features)
        features = self.dropout(features)

        # reshape回时间序列结构
        features = features.view(B, T, self.feature_dim)  # (B, T, feature_dim)

        lstm_out, _ = self.lstm(features)  # (B, T, H)
        final_feat = lstm_out[:, -1, :]  # 取最后一帧的输出

        out = self.output_fc(final_feat)  # (B, feature_dim)
        out = self.relu(out)
        out = self.layernorm(out)
        out = self.dropout(out)

        return out
