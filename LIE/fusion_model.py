import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        assert dim % heads == 0, f"embed_dim ({dim}) must be divisible by num_heads ({heads})"
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 4, dim)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (batch, seq_len, dim), seq_len here=3 (video, audio, text)
        # self-attention跨模态融合
        attn_output, _ = self.attn(x, x, x)  # (batch, seq_len, dim)
        x = self.norm(x + self.dropout(attn_output))
        ff_out = self.ff(x)
        x = self.norm(x + self.dropout(ff_out))
        return x

class FusionModelSmallData(nn.Module):
    """
    小样本多模态融合模型，冻结子模型，只训练融合层和分类头。
    """

    def __init__(self, video_model, audio_model, text_model,
                 fusion_dim=256, num_heads=4, num_classes=2, dropout=0.5, freeze_submodels=True):
        super().__init__()

        # 子模型
        self.video_model = video_model
        self.audio_model = audio_model
        self.text_model = text_model

        if freeze_submodels:
            for model in [video_model, audio_model, text_model]:
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False

        # 要求子模型有 output_dim 属性
        try:
            video_dim = video_model.output_dim
            audio_dim = audio_model.output_dim
            text_dim = text_model.output_dim
        except AttributeError:
            raise ValueError("所有子模型必须定义 output_dim 属性")

        # 融合维度必须能被 num_heads 整除
        assert fusion_dim % num_heads == 0, \
            f"fusion_dim ({fusion_dim}) must be divisible by num_heads ({num_heads})"

        # 各模态先投影到 fusion_dim
        self.video_proj = nn.Linear(video_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)

        # 融合多头注意力层
        self.fusion_attn = CrossModalAttention(dim=fusion_dim, heads=num_heads)

        # 融合后分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )

    def forward(self, video_x, audio_x, input_ids, attention_mask=None, token_type_ids=None):
        # 子模型特征提取
        video_feat = self.video_model(video_x)
        audio_feat = self.audio_model(audio_x)

        text_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            text_inputs["token_type_ids"] = token_type_ids
        text_output = self.text_model(**text_inputs)

        if isinstance(text_output, dict):
            if "pooler_output" in text_output:
                text_feat = text_output["pooler_output"]
            else:
                text_feat = text_output["last_hidden_state"][:, 0]
        else:
            text_feat = text_output

        # 投影到统一维度
        v_proj = self.video_proj(video_feat)
        a_proj = self.audio_proj(audio_feat)
        t_proj = self.text_proj(text_feat)

        # 拼接为 seq_len=3 的序列，用于 attention 融合 (batch, seq_len=3, fusion_dim)
        fused_seq = torch.stack([v_proj, a_proj, t_proj], dim=1)

        # 交叉多头注意力融合
        fused = self.fusion_attn(fused_seq)

        # 融合后取平均或取第一个token作为整体表示
        fused_repr = fused.mean(dim=1)  # (batch, fusion_dim)

        # 分类输出
        logits = self.classifier(fused_repr)
        return logits
