import cv2
import json
import torch
import numpy as np
from collections import deque
from transformers import AutoTokenizer, AutoImageProcessor
from tqdm import tqdm


try:
    from run_sport import (
        SportsEventUnderstandingModel, 
        NUM_CLASSES,
        LABEL_TO_ID
    )
except ImportError:
    print("❌ 错误：找不到 run_sport.py。请确保该文件在当前目录下。")
    exit()


DISPLAY_LABELS = {
    0: "Normal Play", 1: "GOAL !!!", 2: "Yellow Card", 3: "Red Card",
    4: "Corner Kick", 5: "Substitution", 6: "Injury", 7: "Whistle",
    8: "Time Update", 9: "Fun Fact"
}

def load_match_timeline(json_path, half=1):
    """读取 JSON 文件，提取指定半场的时间线和解说词"""
    timeline = []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for ann in data.get("annotations", []):
        try:
            event_half = int(ann["gameTime"].split(" - ")[0])
            if event_half == half:
                timeline.append({
                    "timestamp_ms": int(ann["position"]),
                    "text": ann["description"]
                })
        except:
            continue
            
    timeline.sort(key=lambda x: x["timestamp_ms"])
    return timeline

def test_full_half_match(video_path, json_path, model_path, output_path, process_minutes=5):
    """测试整段半场视频，并将标准小号字幕渲染成视频"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"1. 正在使用 {device} 加载模型...")

    model = SportsEventUnderstandingModel(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    print("2. 正在加载比赛时间线 (JSON)...")
    timeline = load_match_timeline(json_path, half=1)
    
    print(f"3. 打开视频流: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 25
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    max_frames = int(process_minutes * 60 * fps) if process_minutes else total_frames
    max_frames = min(max_frames, total_frames)

    # 视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # 滑动窗口配置
    window_sec = 10
    num_frames_for_model = 8
    max_buffer_size = int(window_sec * fps)
    frame_buffer = deque(maxlen=max_buffer_size)
    
    # 状态变量
    current_prediction = "Waiting for match..."
    current_text = "Match kicks off..."
    timeline_idx = 0
    inference_interval = int(fps * 2.0)
    
    print(f"4. 开始渲染标准字幕视频，预计处理 {max_frames} 帧...")
    progress_bar = tqdm(total=max_frames, desc="视频渲染进度")

    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time_ms = (frame_count / fps) * 1000
        
        # --- 更新真实解说字幕 ---
        if timeline_idx < len(timeline) and current_time_ms >= timeline[timeline_idx]["timestamp_ms"]:
            current_text = timeline[timeline_idx]["text"]
            timeline_idx += 1

        # 加入缓冲区
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(rgb_frame)

        # --- 触发模型推理 ---
        if len(frame_buffer) == max_buffer_size and frame_count % inference_interval == 0:
            indices = np.linspace(0, max_buffer_size - 1, num_frames_for_model, dtype=int)
            sampled_frames = [frame_buffer[i] for i in indices]
            
            # 视觉处理
            pixel_values = image_processor(images=sampled_frames, return_tensors="pt").pixel_values
            pixel_values = pixel_values.unsqueeze(0).to(device)
            
            # 文本处理
            text_inputs = tokenizer(current_text, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
            input_ids = text_inputs.input_ids.to(device)
            attention_mask = text_inputs.attention_mask.to(device)
            
            # 模型预测
            with torch.no_grad():
                outputs = model(pixel_values, input_ids, attention_mask)
                _, predicted_id = torch.max(outputs, 1)
                current_prediction = DISPLAY_LABELS.get(predicted_id.item(), "Unknown")

        # --- 画面渲染绘制 (字幕缩小版) ---
        display_frame = frame.copy()
        
        # 底部黑色半透明字幕条 (把高度调到 70 像素)
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, frame_height - 70), (frame_width, frame_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

        # 字幕文本自动截断 (90 个字符)
        display_text = current_text[:90] + "..." if len(current_text) > 90 else current_text
        
        # --- 绘制 Live Commentary ---
        # fontScale 从 0.6 调小为 0.45，thickness 从 2 调小为 1
        cv2.putText(display_frame, f"Live Commentary: {display_text}", 
                    (20, frame_height - 42), # 稍微往下移一点，更贴合黑色背景
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.45, 
                    (220, 220, 220), # 使用浅灰色，更柔和，像字幕
                    1)
        
        # --- 绘制 AI Prediction ---
        # fontScale 从 0.8 调小为 0.55，thickness 从 2 调小为 1
        color = (0, 255, 0) if current_prediction == "Normal Play" else (0, 0, 255)
        cv2.putText(display_frame, f"AI Multimodal Predict: {current_prediction}", 
                    (20, frame_height - 15), # 移动到离底部更近
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.55, 
                    color, 
                    1) # 粗细也改为 1

        # 写入视频
        out_video.write(display_frame)
        
        frame_count += 1
        progress_bar.update(1)

    # 资源释放
    progress_bar.close()
    cap.release()
    out_video.release()
    print(f"\n✅ 测试完毕！已生成标准字幕版文件: {output_path}")


if __name__ == "__main__":
    BASE_DIR = "video_data/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"
    JSON_DIR = "text_data/caption-2023/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"
    
    VIDEO_FILE = f"{BASE_DIR}/1_224p.mkv"
    JSON_FILE = f"{JSON_DIR}/Labels-caption.json"
    MODEL_WEIGHTS = "sports_multimodal_model.pth"
    OUTPUT_FILE = "chelsea_burnley_half1_small_text.mp4"
    
    try:
        test_full_half_match(
            video_path=VIDEO_FILE,
            json_path=JSON_FILE,
            model_path=MODEL_WEIGHTS,
            output_path=OUTPUT_FILE,
            process_minutes=15 # 设置视频时间
        )
    except Exception as e:
        print(f"运行出错: {e}")
