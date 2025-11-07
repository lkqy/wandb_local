#!/usr/bin/env python3
"""
高级功能示例 - 展示所有高级特性（修复版本）
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import wandb_local as wandb


class AdvancedModel(nn.Module):
    """带有多个层的复杂模型"""
    
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))
        
        # 隐藏层
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
        
        # 输出层
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def create_sample_images(n_images=5):
    """创建示例图像数据"""
    images = []
    for i in range(n_images):
        # 创建不同模式的图像
        if i % 3 == 0:
            # 随机噪声
            img = np.random.rand(64, 64, 3) * 255
        elif i % 3 == 1:
            # 渐变图像
            x = np.linspace(0, 255, 64)
            y = np.linspace(0, 255, 64)
            xx, yy = np.meshgrid(x, y)
            img = np.stack([xx, yy, xx+yy], axis=-1) % 255
        else:
            # 圆形图案
            center = (32, 32)
            radius = 20
            img = np.zeros((64, 64, 3))
            for y in range(64):
                for x in range(64):
                    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                    if dist <= radius:
                        img[y, x] = [255, 100, 100]
        
        images.append(img.astype(np.uint8))
    return images


def create_sample_audio(duration=2, sample_rate=44100):
    """创建示例音频数据"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 创建复合音调
    freq1 = 440  # A4
    freq2 = 554  # C#5
    freq3 = 659  # E5
    
    audio = (np.sin(2 * np.pi * freq1 * t) * 0.3 + 
             np.sin(2 * np.pi * freq2 * t) * 0.2 + 
             np.sin(2 * np.pi * freq3 * t) * 0.1)
    
    # 添加衰减
    audio *= np.exp(-t * 2)
    
    return audio.astype(np.float32)


def create_sample_video(n_frames=16, height=64, width=64):
    """创建示例视频数据"""
    video = []
    
    for frame_idx in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 移动的圆形
        center_x = int(width // 2 + 20 * np.sin(frame_idx * 2 * np.pi / n_frames))
        center_y = height // 2
        radius = 10
        
        for y in range(height):
            for x in range(width):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist <= radius:
                    frame[y, x] = [255, 255, 100]
        
        video.append(frame)
    
    return np.array(video)


def main():
    """主函数"""
    print("🚀 WandB Local 高级功能示例（修复版本）")
    print("=" * 60)
    
    # 实验配置
    config = {
        "learning_rate": 0.001,
        "epochs": 3,  # 减少epoch数以加快测试
        "batch_size": 64,
        "input_size": 100,
        "hidden_sizes": [128, 64, 32],
        "output_size": 10,
        "dropout_rate": 0.3,
        "n_samples": 500,  # 减少样本数以加快测试
        "experiment_type": "advanced_features_demo"
    }
    
    # 初始化实验
    print("📊 初始化高级实验...")
    run = wandb.init(
        project="advanced-example-fixed",
        name="feature-demonstration-fixed",
        config=config,
        tags=["demo", "advanced", "multimedia", "artifacts"],
        notes="展示所有高级功能的综合示例（修复版本）"
    )
    
    print(f"✅ 实验已启动: {run.run_id}")
    print(f"📁 数据存储路径: {run.dir}")
    
    # 创建模型
    print("🧠 创建复杂模型...")
    model = AdvancedModel(
        config["input_size"],
        config["hidden_sizes"],
        config["output_size"],
        config["dropout_rate"]
    )
    
    # 监控模型
    print("👀 监控模型（梯度+参数）...")
    wandb.watch(model, log="all", log_freq=25)
    
    # 设置训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # 生成数据
    print("📊 生成训练数据...")
    X_train = torch.randn(config["n_samples"], config["input_size"])
    y_train = torch.randint(0, config["output_size"], (config["n_samples"],))
    
    X_val = torch.randn(config["n_samples"] // 4, config["input_size"])
    y_val = torch.randint(0, config["output_size"], (config["n_samples"] // 4,))
    
    # 创建结果表格
    results_table = wandb.Table(columns=[
        "epoch", "batch_idx", "train_loss", "val_loss", 
        "accuracy", "learning_rate", "grad_norm"
    ])
    
    # 训练循环
    print("🏃 开始训练...")
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        n_batches = len(X_train) // config["batch_size"]
        
        for batch_idx in range(n_batches):
            # 获取批次数据
            start_idx = batch_idx * config["batch_size"]
            end_idx = start_idx + config["batch_size"]
            
            batch_X = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            
            # 计算梯度范数
            grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            optimizer.step()
            total_loss += loss.item()
            
            # 记录批次指标
            if batch_idx % 10 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_grad_norm": grad_norm,
                    "batch_idx": batch_idx,
                    "epoch": epoch
                })
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            
            _, predicted = torch.max(val_outputs, 1)
            accuracy = (predicted == y_val).float().mean().item()
        
        avg_train_loss = total_loss / n_batches
        
        # 记录epoch指标
        print(f"Epoch {epoch+1}/{config['epochs']} - "
              f"Loss: {avg_train_loss:.4f} - "
              f"Val Loss: {val_loss.item():.4f} - "
              f"Accuracy: {accuracy:.4f} - "
              f"Grad Norm: {grad_norm:.4f}")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss.item(),
            "accuracy": accuracy,
            "grad_norm": grad_norm,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })
        
        # 添加到结果表格
        results_table.add_data(
            epoch, batch_idx, avg_train_loss, val_loss.item(),
            accuracy, optimizer.param_groups[0]["lr"], grad_norm
        )
        
        # 每2个epoch记录多媒体数据
        if epoch % 2 == 0:
            print(f"🎨 记录多媒体数据 (epoch {epoch})...")
            
            # 记录图像
            images = create_sample_images(3)
            image_captions = [f"Epoch {epoch} - Image {i}" for i in range(len(images))]
            wandb.log({
                "sample_images": [wandb.Image(img, caption=caption) 
                                for img, caption in zip(images, image_captions)]
            })
            
            # 记录音频
            audio = create_sample_audio(duration=1)
            wandb.log({
                "sample_audio": wandb.Audio(audio, 44100, 
                                          caption=f"训练音频 - Epoch {epoch}")
            })
            
            # 记录视频
            video = create_sample_video(n_frames=8)
            wandb.log({
                "sample_video": wandb.Video(video, fps=2, 
                                          caption=f"训练视频 - Epoch {epoch}")
            })
    
    # 记录最终表格
    print("📋 记录结果表格...")
    wandb.log({"training_results": results_table})
    
    # 创建和保存模型Artifact
    print("📦 创建模型Artifact...")
    model_artifact = wandb.Artifact(
        "trained-model", 
        type="model",
        description=f"在{config['epochs']}个epoch后训练的模型",
        metadata={
            "final_accuracy": accuracy,
            "total_epochs": config["epochs"],
            "model_architecture": "AdvancedModel",
            "hidden_sizes": config["hidden_sizes"]
        }
    )
    
    # 保存模型
    model_path = "final_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_accuracy': accuracy,
        'epochs_trained': config["epochs"]
    }, model_path)
    
    model_artifact.add_file(model_path)
    wandb.log({"final_model": model_artifact})
    
    # 创建数据Artifact
    print("📊 创建数据Artifact...")
    data_artifact = wandb.Artifact(
        "training-data",
        type="dataset",
        description="用于训练的虚拟数据集"
    )
    
    # 保存训练数据（示例）
    train_data_path = "train_data_sample.npy"
    np.save(train_data_path, X_train[:100].numpy())  # 只保存一小部分作为示例
    data_artifact.add_file(train_data_path)
    
    wandb.log({"training_data": data_artifact})
    
    # 创建配置Artifact
    print("⚙️ 创建配置Artifact...")
    config_artifact = wandb.Artifact(
        "experiment-config",
        type="config",
        description="实验配置文件"
    )
    
    import json
    config_path = "experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    config_artifact.add_file(config_path)
    wandb.log({"experiment_config": config_artifact})
    
    # 发送告警
    print("🔔 发送完成告警...")
    wandb.alert(
        title="高级实验完成",
        text=f"所有高级功能演示完成！\n"
              f"最终准确率: {accuracy:.3f}\n"
              f"总训练轮数: {config['epochs']}\n"
              f"模型已保存到Artifact",
        level="SUCCESS",
        wait_duration=5
    )
    
    # 结束实验
    print("🏁 结束实验...")
    wandb.finish()
    
    print("\n✅ 高级实验完成！")
    print(f"📊 查看结果: {run.dir}")
    print(f"🔗 运行ID: {run.run_id}")
    
    # 显示实验摘要（直接从run对象获取）
    print(f"\n📈 实验摘要:")
    for key, value in run.summary.items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # 显示保存的文件
    print(f"\n💾 保存的文件:")
    if os.path.exists(run.dir):
        for root, dirs, files in os.walk(run.dir):
            level = root.replace(run.dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:10]:  # 只显示前10个文件
                print(f"{subindent}{file}")
            if len(files) > 10:
                print(f"{subindent}... 还有 {len(files) - 10} 个文件")


if __name__ == "__main__":
    main()