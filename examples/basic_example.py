#!/usr/bin/env python3
"""
基础使用示例 - 简单的实验跟踪
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# 添加父目录到路径，以便导入wandb_local
sys.path.insert(0, str(Path(__file__).parent.parent))

import wandb_local as wandb


class SimpleModel(nn.Module):
    """简单的全连接神经网络"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def generate_dummy_data(n_samples, input_size, output_size):
    """生成虚拟训练数据"""
    X = np.random.randn(n_samples, input_size).astype(np.float32)
    y = np.random.randint(0, output_size, size=(n_samples,))
    return torch.tensor(X), torch.tensor(y, dtype=torch.long)


def main():
    """主函数"""
    print("🚀 WandB Local 基础示例")
    print("=" * 50)
    
    # 实验配置
    config = {
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32,
        "input_size": 100,
        "hidden_size": 50,
        "output_size": 10,
        "n_samples": 1000
    }
    
    # 初始化实验
    print("📊 初始化实验...")
    run = wandb.init(
        project="basic-example",
        name="simple-neural-network",
        config=config,
        tags=["demo", "basic", "pytorch"],
        notes="这是一个基础的神经网络训练示例"
    )
    
    print(f"✅ 实验已启动: {run.run_id}")
    print(f"📁 数据存储路径: {run.dir}")
    
    # 创建模型
    print("🧠 创建模型...")
    model = SimpleModel(
        config["input_size"], 
        config["hidden_size"], 
        config["output_size"]
    )
    
    # 监控模型
    print("👀 开始监控模型...")
    wandb.watch(model, log="all", log_freq=50)
    
    # 设置训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # 生成数据
    print("📊 生成训练数据...")
    X_train, y_train = generate_dummy_data(
        config["n_samples"], 
        config["input_size"], 
        config["output_size"]
    )
    
    X_val, y_val = generate_dummy_data(
        config["n_samples"] // 4, 
        config["input_size"], 
        config["output_size"]
    )
    
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
            optimizer.step()
            
            total_loss += loss.item()
            
            # 记录批次指标
            if batch_idx % 10 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_idx": batch_idx,
                    "epoch": epoch
                })
        
        # 计算平均训练损失
        avg_train_loss = total_loss / n_batches
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            
            # 计算准确率
            _, predicted = torch.max(val_outputs, 1)
            accuracy = (predicted == y_val).float().mean().item()
        
        # 记录epoch指标
        print(f"Epoch {epoch+1}/{config['epochs']} - "
              f"Loss: {avg_train_loss:.4f} - "
              f"Val Loss: {val_loss.item():.4f} - "
              f"Accuracy: {accuracy:.4f}")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss.item(),
            "accuracy": accuracy
        })
        
        # 每3个epoch保存一次模型
        if epoch % 3 == 0:
            model_path = f"model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)
            print(f"💾 保存模型: {model_path}")
    
    # 创建示例图像
    print("🎨 创建示例图像...")
    sample_image = np.random.rand(64, 64, 3) * 255
    wandb.log({
        "sample_image": wandb.Image(
            sample_image.astype(np.uint8), 
            caption="随机生成的示例图像"
        )
    })
    
    # 创建结果表格
    print("📋 创建结果表格...")
    results_table = wandb.Table(columns=["epoch", "train_loss", "val_loss", "accuracy"])
    
    # 重新获取历史数据创建表格
    history = wandb.get_history(run.run_id)
    for entry in history:
        if "epoch" in entry.get("data", {}):
            results_table.add_data(
                entry["data"]["epoch"],
                entry["data"].get("train_loss", 0),
                entry["data"].get("val_loss", 0),
                entry["data"].get("accuracy", 0)
            )
    
    wandb.log({"results_table": results_table})
    
    # 发送告警
    print("🔔 发送完成告警...")
    final_accuracy = accuracy  # 使用最后一次的准确率
    wandb.alert(
        title="训练完成",
        text=f"模型训练完成！最终准确率: {final_accuracy:.3f}",
        level="SUCCESS"
    )
    
    # 结束实验
    print("🏁 结束实验...")
    wandb.finish()
    
    print("\n✅ 实验完成！")
    print(f"📊 查看结果: {run.dir}")
    print(f"🔗 运行ID: {run.run_id}")
    
    # 显示实验总结
    summary = wandb.get_summary(run.run_id)
    print(f"\n📈 实验摘要:")
    for key, value in summary.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    main()