import os
import json
import random
import threading
import numpy as np
from tqdm import tqdm

import traceback

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchaudio

from transformers import BertModel, BertTokenizer
from torch.utils.tensorboard import SummaryWriter
import wandb
import optuna
from sklearn.model_selection import train_test_split


# ---------------------- Utilities ----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------- Dataset ----------------------
from PIL import Image
import torchvision.transforms as T
import torch

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as TAudio
import torchvision.transforms as T
from torch.utils import data
from PIL import Image
import json
import random

import os
import json
import torch
import random
import traceback
import threading
import torchaudio
import torch.nn as nn
from torch.utils import data
from PIL import Image
from torchvision import transforms as T
import torchaudio.transforms as TAudio


import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # 🔵 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # 🟢 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy', color='blue')
    plt.plot(epochs, val_accs, label='Val Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



class MultimodalDataset(data.Dataset):
    def __init__(self, samples, tokenizer, device,
                 max_text_len=128, max_audio_len=500,
                 video_transform=None, target_audio_channels=4,
                 sample_rate=16000, max_video_len=16,
                 augment=False, skip_error=True):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len
        self.video_transform = video_transform
        self.target_audio_channels = target_audio_channels
        self.sample_rate = sample_rate
        self.device = device
        self.max_video_len = max_video_len
        self.augment = augment
        self.skip_error = skip_error

        # 音频处理
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=40,
            n_fft=400,
            hop_length=160
        )

        self.audio_augment = nn.Sequential(
            TAudio.FrequencyMasking(freq_mask_param=15),
            TAudio.TimeMasking(time_mask_param=35)
        ) if augment else None

        # 视频处理
        if self.video_transform is None:
            if self.augment:
                self.video_transform = T.Compose([
                    T.Resize((224, 224)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    T.ToTensor()
                ])
            else:
                self.video_transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor()
                ])

    def __len__(self):
        return len(self.samples)

    def _load_image_with_timeout(self, path, timeout=3):
        result = {}

        def target():
            try:
                result['img'] = Image.open(path).convert('RGB')
            except Exception as e:
                print(f"[ERROR] Loading image failed: {e}")
                result['img'] = None

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            print(f"[WARNING] Timeout loading image: {path}")
            return None
        return result.get('img', None)

    def _augment_text(self, text):
        """简单文本增强：语气词替换、重复词插入等"""
        if random.random() < 0.3:
            text = text.replace(".", "...").replace(",", ", uh")
        if random.random() < 0.2:
            words = text.split()
            if len(words) > 5:
                insert_pos = random.randint(1, len(words) - 2)
                words.insert(insert_pos, words[insert_pos])  # 重复词
                text = " ".join(words)
        return text

    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]
            print(f"[DEBUG] Loading sample {idx}")

            # === 文本处理 ===
            with open(sample['text_json_path'], 'r', encoding='utf-8') as f:
                text_segments = json.load(f)
            full_text = " ".join([seg['text'] for seg in text_segments])

            if self.augment:
                full_text = self._augment_text(full_text)

            enc = self.tokenizer(
                full_text,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_len,
                return_tensors='pt'
            )
            input_ids = enc['input_ids'].squeeze(0)
            attn_mask = enc['attention_mask'].squeeze(0)

            # === 音频处理 ===
            waveform, sr = torchaudio.load(sample['audio_path'])
            waveform = waveform.cpu().detach().clone().contiguous()

            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            c, l = waveform.shape
            if c < self.target_audio_channels:
                padding = torch.zeros(self.target_audio_channels - c, l, dtype=waveform.dtype)
                waveform = torch.cat([waveform, padding], dim=0)
            elif c > self.target_audio_channels:
                waveform = waveform[:self.target_audio_channels]

            mel = self.melspec(waveform)  # [C, F, T]
            mel = mel.mean(dim=0, keepdim=True)  # [1, F, T]

            if self.audio_augment:
                mel = self.audio_augment(mel)

            _, n_mels, t_len = mel.shape
            if t_len > self.max_audio_len:
                mel = mel[:, :, :self.max_audio_len]
            elif t_len < self.max_audio_len:
                pad_len = self.max_audio_len - t_len
                pad_tensor = torch.zeros((1, n_mels, pad_len), dtype=mel.dtype)
                mel = torch.cat([mel, pad_tensor], dim=2)

            # === 视频处理 ===
            frame_paths = sample.get('frame_images', [])
            if len(frame_paths) > self.max_video_len:
                frame_paths = frame_paths[:self.max_video_len]

            frames = []
            for p in frame_paths:
                img = self._load_image_with_timeout(p)
                if img is not None:
                    frames.append(self.video_transform(img))

            if len(frames) < self.max_video_len:
                if frames:
                    pad_frame = torch.zeros_like(frames[0])
                else:
                    pad_frame = torch.zeros(3, 224, 224)
                frames.extend([pad_frame] * (self.max_video_len - len(frames)))

            video_tensor = torch.stack(frames)  # (T, C, H, W)
            label = torch.tensor(sample['label'], dtype=torch.long)

            return (
                    video_tensor,
                    mel,
                    input_ids,
                    attn_mask,
                    label
                )


        except Exception as e:
            traceback.print_exc()
            print(f"[ERROR] Failed to load sample {idx}: {e}")
            if self.skip_error:
                return self.__getitem__((idx + 1) % len(self.samples))  # 尝试加载下一个样本
            else:
                raise e

# ---------------------- Models ----------------------
class VideoTransformer(nn.Module):
    def __init__(self, feature_dim=128, num_frames=16, pretrained=True, freeze_cnn=True):
        super().__init__()
        self.cnn = models.resnet18(pretrained=pretrained)
        self.cnn.fc = nn.Identity()
        self.cnn.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, 512))
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x):
        # x: (B, T, C, H, W)
        if x is None:
            B = 1
            feature = torch.zeros(B, self.fc.out_features).to(self.fc.weight.device)
            return feature
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.cnn(x).view(B, T, -1)  # (B, T, 512)
        feat = feat + self.pos_embed[:, :T, :]
        feat = self.transformer(feat)
        feat = feat.mean(dim=1)
        return self.fc(feat)


class AudioModel(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(40, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        # x: (B, 1, freq, time) -> squeeze freq dim to (B, freq, time)
        x = x.squeeze(1)
        return self.net(x)


class TextModel(nn.Module):
    def __init__(self, feature_dim=128, bert_name='bert-base-uncased', freeze_bert=True):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.fc = nn.Linear(self.bert.config.hidden_size, feature_dim)

    def forward(self, input_ids, attn_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        cls = out.pooler_output
        return self.fc(cls)


def get_num_heads(embed_dim, max_heads=16):
    for h in range(max_heads, 0, -1):
        if embed_dim % h == 0:
            return h
    return 1


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, max_heads=16):
        super().__init__()
        num_heads = get_num_heads(embed_dim, max_heads)
        print(f"Using num_heads={num_heads} for embed_dim={embed_dim}")
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value)
        return attn_output


class FusionModel(nn.Module):
    def __init__(self, video_model, audio_model, text_model, fusion_dim=256, num_classes=2, dropout=0.5):
        super().__init__()
        self.video_model = video_model
        self.audio_model = audio_model
        self.text_model = text_model

        self.vid_proj = nn.Linear(self.video_model.fc.out_features, fusion_dim)
        self.aud_proj = nn.Linear(audio_model.net[-1].out_features, fusion_dim)
        self.txt_proj = nn.Linear(self.text_model.fc.out_features, fusion_dim)

        self.vid2aud = CrossModalAttention(fusion_dim)
        self.aud2vid = CrossModalAttention(fusion_dim)
        self.vid2txt = CrossModalAttention(fusion_dim)

        total_dim = fusion_dim * 3
        self.head = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Dropout(dropout),
            nn.Linear(total_dim, num_classes)
        )

    def forward(self, vid, aud, input_ids, attn_mask):
        vid_feat = self.video_model(vid)
        aud_feat = self.audio_model(aud)
        txt_feat = self.text_model(input_ids, attn_mask)

        v = self.vid_proj(vid_feat).unsqueeze(1)  # (B, 1, D)
        a = self.aud_proj(aud_feat).unsqueeze(1)  # (B, 1, D)
        t = self.txt_proj(txt_feat).unsqueeze(1)  # (B, 1, D)

        va = self.vid2aud(v, a, a)  # (B,1,D)
        av = self.aud2vid(a, v, v)  # (B,1,D)
        vt = self.vid2txt(v, t, t)  # (B,1,D)

        fused = torch.cat([va.squeeze(1), av.squeeze(1), vt.squeeze(1)], dim=1)  # (B, 3*D)
        out = self.head(fused)
        return out


# ---------------------- Training and Validation ----------------------
def train_epoch(model, dataloader, criterion, optimizer, device, epoch, trial=None, tb_writer=None, wandb_run=None):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for i, (vid, aud, input_ids, attn_mask, labels) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()

        aud = aud.to(device)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)
        vid = vid.to(device) if vid is not None else None

        # 🔍 输入检查
        if torch.isnan(aud).any() or torch.isnan(input_ids).any():
            print(f"[WARNING] Skipping batch {i}: NaN in input tensors")
            continue

        try:
            outputs = model(vid, aud, input_ids, attn_mask)
            # 🔍 输出检查
            if torch.isnan(outputs).any() or outputs.mean().item() == 0:
                print(f"[WARNING] Skipping batch {i}: outputs are NaN or zero")
                continue

            loss = criterion(outputs, labels)

            # 🔍 loss 检查
            if torch.isnan(loss) or loss.item() == 0:
                print(f"[WARNING] Skipping batch {i}: invalid loss: {loss.item()}")
                continue

            loss.backward()
            # 🔍 Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        except Exception as e:
            print(f"[ERROR] Exception at batch {i}: {e}")
            continue

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        # 🔍 Optuna 中间报告
        if trial is not None and i % 10 == 0 and total_samples > 0:
            intermediate_value = total_loss / total_samples
            trial.report(intermediate_value, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # 🔍 Debug logging
        if i % 10 == 0:
            print(f"[DEBUG] Epoch {epoch} | Batch {i}")
            print(f"[DEBUG] Loss: {loss.item():.4f} | Output mean: {outputs.mean().item():.4f}")

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    if tb_writer is not None:
        tb_writer.add_scalar("Train/Loss", avg_loss, epoch)
        tb_writer.add_scalar("Train/Accuracy", accuracy, epoch)

    if wandb_run is not None:
        wandb_run.log({"Train Loss": avg_loss, "Train Accuracy": accuracy, "epoch": epoch})

    return avg_loss, accuracy



def validate_epoch(model, dataloader, criterion, device, epoch, tb_writer=None, wandb_run=None):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for vid, aud, input_ids, attn_mask, labels in dataloader:
            aud = aud.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            vid = vid.to(device) if vid is not None else None

            outputs = model(vid, aud, input_ids, attn_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    if tb_writer is not None:
        tb_writer.add_scalar("Val/Loss", avg_loss, epoch)
        tb_writer.add_scalar("Val/Accuracy", accuracy, epoch)

    if wandb_run is not None:
        wandb_run.log({"Val Loss": avg_loss, "Val Accuracy": accuracy, "epoch": epoch})

    return avg_loss, accuracy


# ---------------------- Main Run / Optuna ----------------------
def main(config, trial=None):
    set_seed(config['seed'])
    device = torch.device(config['device'])

    # 加载样本列表，字典 'audio_path', 'text_json_path', 'label'
    with open(config['data_list_path'], 'r') as f:
        samples = json.load(f)
    
    train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=config['seed'])

    # 初始化 tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['bert_model'])

    # 构建数据集，训练集开启数据增强
    train_dataset = MultimodalDataset(train_samples, tokenizer, device, augment=True)
    val_dataset = MultimodalDataset(val_samples, tokenizer, device, augment=False)

    # 构建数据加载器
    train_loader = data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    video_model = VideoTransformer(feature_dim=config['fusion_dim']).to(device)
    audio_model = AudioModel(embed_dim=config['fusion_dim']).to(device)
    text_model = TextModel(feature_dim=config['fusion_dim'], bert_name=config['bert_model']).to(device)

    model = FusionModel(video_model, audio_model, text_model, fusion_dim=config['fusion_dim'], num_classes=config['num_classes'])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    tb_writer = None
    wandb_run = None
    if config.get('use_tensorboard', False):
        tb_writer = SummaryWriter(log_dir=config['tensorboard_logdir'])
    if config.get('use_wandb', False):
        wandb.init(project=config['wandb_project'], config=config)
        wandb_run = wandb.run

    best_val_acc = 0
    best_epoch = 0
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, trial, tb_writer, wandb_run)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, epoch, tb_writer, wandb_run)
        
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch}: Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}")

        # 保存最好模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

        # 早停判断
        if epoch - best_epoch >= config.get('early_stopping_patience', 5):
            print("Early stopping triggered")
            break

    if tb_writer is not None:
        tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    
    plot_path = os.path.join(save_dir, "training_curves.png")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, plot_path)

    
    return best_val_acc


def objective(trial):
    config = {
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data_list_path': 'samples.json',  #######
        'bert_model': 'bert-base-uncased',
        'batch_size': trial.suggest_categorical('batch_size', [16, 32]),
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-3),
        'epochs': 20,
        'fusion_dim': trial.suggest_categorical('fusion_dim', [128, 256]),
        'num_classes': 2,
        'use_tensorboard': False,
        'use_wandb': False,
        'save_dir': './checkpoints',
        'early_stopping_patience': 5,
        'tensorboard_logdir': './runs'
    }

    best_val_acc = main(config, trial)
    return best_val_acc


if __name__ == "__main__":
    # Optuna 调参入口
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")
