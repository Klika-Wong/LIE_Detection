import torch
import torch.nn as nn
import torchvision.models as models


def convert_bn_to_gn(module):
    """
    Recursively replace all BatchNorm2d with GroupNorm.
    GroupNorm is more stable for small-batch training.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            setattr(module, name, nn.GroupNorm(num_groups=8, num_channels=num_channels))
        else:
            convert_bn_to_gn(child)


class AudioModel(nn.Module):
    def __init__(self, input_dim=40, pretrained=False, use_groupnorm=True,
                 embed_dim=128, dropout=0.3, use_lstm=True):
        """
        input_dim: Number of mel bins (e.g., 40)
        embed_dim: Output feature dimension
        use_lstm: Whether to apply bidirectional LSTM
        """
        super().__init__()
        self.output_dim = embed_dim
        self.use_lstm = use_lstm

        # Load ResNet18 backbone
        self.cnn = models.resnet18(pretrained=pretrained)

        # Replace first conv layer to accept 1 channel (e.g., log-mel input)
        self.cnn.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )

        # Optionally replace BN with GN
        if use_groupnorm:
            convert_bn_to_gn(self.cnn)

        # Remove original classifier
        self.cnn.fc = nn.Identity()
        self.cnn.avgpool = nn.Identity()

        # Adaptive pooling to (1,1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Project to lower dimension
        self.embedding = nn.Linear(512, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Optional Bi-LSTM
        if self.use_lstm:
            self.lstm = nn.LSTM(embed_dim, embed_dim // 2,
                                batch_first=True, bidirectional=True)

    def forward(self, x):
        """
        x: (B, 1, freq_bins, time_steps)
        """
        B, C, F, T = x.shape

        # Reshape input for CNN: treat time as batch (for temporal modeling)
        x = x.permute(0, 3, 1, 2)  # (B, T, 1, F)
        x = x.reshape(B * T, 1, F, 1)  # 每一帧当作一张图像 (B*T, 1, F, 1)

        # CNN encoding
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)

        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        x = self.cnn.layer4(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # (B*T, 512)

        # Project and normalize
        x = self.embedding(x)
        x = self.norm(x)
        x = self.dropout(x)

        # Reshape back to (B, T, embed_dim)
        x = x.view(B, T, -1)

        if self.use_lstm:
            x, _ = self.lstm(x)

        # Temporal aggregation: average
        x = x.mean(dim=1)  # (B, embed_dim)
        return x


if __name__ == "__main__":
    batch_size = 8
    freq_bins = 40
    time_steps = 100

    # Simulated log-mel spectrogram input
    dummy_audio = torch.randn(batch_size, 1, freq_bins, time_steps)

    model = AudioModel(
        input_dim=freq_bins,
        pretrained=False,
        use_groupnorm=True,
        embed_dim=128,
        use_lstm=True
    )

    output = model(dummy_audio)
    print(f"Input shape: {dummy_audio.shape}")
    print(f"Output shape: {output.shape}")  # (8, 128)
