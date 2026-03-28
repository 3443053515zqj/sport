import os
import json
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel
from tqdm import tqdm


# 将 JSON 中的字符串标签转为模型可识别的整数 ID
LABEL_TO_ID = {
    "": 0,               # 普通解说 / 无特定动作
    "soccer-ball": 1,    # 进球
    "y-card": 2,         # 黄牌
    "r-card": 3,         # 红牌
    "corner": 4,         # 角球
    "substitution": 5,   # 换人
    "injury": 6,         # 受伤
    "whistle": 7,        # 吹哨
    "time": 8,           # 补时等时间信息
    "funfact": 9         # 有趣的数据
}
NUM_CLASSES = len(LABEL_TO_ID)


def parse_soccernet_data(text_data_dir, video_data_dir):
    """解析目录结构，提取所有带时间戳的事件并与视频路径对齐"""
    samples = []
    # 假设目前只处理 2014-2015 赛季的数据
    season = "2014-2015"
    text_season_dir = os.path.join(text_data_dir, season)
    
    if not os.path.exists(text_season_dir):
        print(f"警告: 找不到文本数据目录 {text_season_dir}")
        return samples

    for match_folder in os.listdir(text_season_dir):
        json_path = os.path.join(text_season_dir, match_folder, "Labels-caption.json")
        if not os.path.exists(json_path):
            continue
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for ann in data.get("annotations", []):
            try:
                # 获取是上半场(1)还是下半场(2)
                half = int(ann["gameTime"].split(" - ")[0])
                # 根据半场信息拼凑视频文件路径
                video_file = os.path.join(video_data_dir, season, match_folder, f"{half}_224p.mkv")
                
                if os.path.exists(video_file):
                    label_str = ann.get("label", "")
                    label_id = LABEL_TO_ID.get(label_str, 0) # 未知标签默认归为0
                    
                    samples.append({
                        "video_path": video_file,
                        "timestamp_ms": int(ann["position"]),
                        "text": ann["description"],
                        "label": label_id
                    })
            except Exception as e:
                # 忽略格式不符合预期的异常数据
                continue
                
    return samples

class SoccerNetDataset(Dataset):
    """适配长视频的局部窗口抽帧数据集"""
    def __init__(self, samples, num_frames=8, window_sec=10):
        self.samples = samples
        self.num_frames = num_frames
        self.window_sec = window_sec
        
        # 加载 HuggingFace 的预训练处理器
        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    def __len__(self):
        return len(self.samples)

    def _extract_window_frames(self, video_path, timestamp_ms):
        """根据事件发生的时间戳，截取前后几秒的画面帧"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        #if fps == 0 or np.isnan(fps): 
            #fps = 25 # 容错处理，防止视频损坏
            
        center_frame_idx = int((timestamp_ms / 1000.0) * fps)
        half_window_frames = int((self.window_sec / 2) * fps)
        
        start_frame = max(0, center_frame_idx - half_window_frames)
        end_frame = center_frame_idx + half_window_frames
        
        # 均匀采样 num_frames 帧
        frame_indices = np.linspace(start_frame, end_frame, self.num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()
        
        # 长度补齐
        while len(frames) < self.num_frames:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
        return frames

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 处理视觉流
        frames = self._extract_window_frames(sample["video_path"], sample["timestamp_ms"])
        pixel_values = self.image_processor(images=frames, return_tensors="pt").pixel_values
        
        # 处理文本流
        text_inputs = self.tokenizer(sample["text"], padding='max_length', truncation=True, max_length=64, return_tensors="pt")
        
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs.input_ids.squeeze(0),
            "attention_mask": text_inputs.attention_mask.squeeze(0),
            "label": torch.tensor(sample["label"], dtype=torch.long)
        }


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合层：让文本指导视频特征的提取"""
    def __init__(self, visual_dim, text_dim, hidden_dim, num_heads=8):
        super().__init__()
        self.v_proj = nn.Linear(visual_dim, hidden_dim)
        self.t_proj = nn.Linear(text_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, visual_features, text_features):
        v_emb = self.v_proj(visual_features)
        t_emb = self.t_proj(text_features)
        attn_output, _ = self.cross_attn(query=t_emb, key=v_emb, value=v_emb)
        fused_features = self.layer_norm(t_emb + self.dropout(attn_output))
        return fused_features

class SportsEventUnderstandingModel(nn.Module):
    """端到端多模态体育事件理解大模型"""
    def __init__(self, num_classes, hidden_dim=512):
        super().__init__()
        
        self.vision_encoder = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.text_encoder = AutoModel.from_pretrained("bert-base-chinese")
        
        self.fusion_layer = CrossAttentionFusion(
            visual_dim=self.vision_encoder.config.hidden_size,
            text_dim=self.text_encoder.config.hidden_size,
            hidden_dim=hidden_dim
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        batch_size, num_frames, c, h, w = pixel_values.shape
        
        # 提取视觉特征
        pixel_values = pixel_values.view(-1, c, h, w)
        vision_outputs = self.vision_encoder(pixel_values)
        v_features = vision_outputs.last_hidden_state[:, 0, :].view(batch_size, num_frames, -1)
        
        # 提取文本特征
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        t_features = text_outputs.last_hidden_state
        
        # 跨模态融合与分类
        fused_features = self.fusion_layer(v_features, t_features)
        global_feature = fused_features.mean(dim=1) 
        logits = self.classifier(global_feature)
        
        return logits


def train_model(model, dataloader, criterion, optimizer, device, num_epochs=5):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values, input_ids, attention_mask)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
            
        print(f"Epoch [{epoch+1}/{num_epochs}] 结束 | 平均 Loss: {running_loss/len(dataloader):.4f} | 准确率: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    TEXT_DIR = "./text_data/caption-2023/england_epl" 
    VIDEO_DIR = "./video_data/england_epl"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用计算设备: {device}")

    print("1. 正在扫描并解析数据集...")
    parsed_samples = parse_soccernet_data(TEXT_DIR, VIDEO_DIR)
    print(f"解析完成！共找到 {len(parsed_samples)} 个有效的事件样本。")

    if len(parsed_samples) > 0:
        print("2. 正在构建数据加载器 (DataLoader)...")
        # 截取事件前后共 10 秒的视频画面，抽取 8 帧
        dataset = SoccerNetDataset(parsed_samples, num_frames=8, window_sec=10)
        # batch_size 设置为 4 比较保守，可以调到 8 或 16
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2) 
        
        print("3. 正在初始化多模态模型...")
        model = SportsEventUnderstandingModel(num_classes=NUM_CLASSES)
        
        # 定义损失函数与分层学习率优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW([
            {'params': model.vision_encoder.parameters(), 'lr': 1e-5},
            {'params': model.text_encoder.parameters(), 'lr': 1e-5},
            {'params': model.fusion_layer.parameters(), 'lr': 1e-4},
            {'params': model.classifier.parameters(), 'lr': 1e-4}
        ])
        
        print("4. 开始训练网络...")
        train_model(model, dataloader, criterion, optimizer, device, num_epochs=10)
        
        # 训练结束后保存模型权重
        torch.save(model.state_dict(), "sports_multimodal_model.pth")
        print("训练完成！模型已保存为 sports_multimodal_model.pth")
    else:
        print("未找到有效数据，请检查 TEXT_DIR 和 VIDEO_DIR 的路径是否正确。")
