#!/usr/bin/env python3
"""
超参数搜索示例 - 展示sweep功能
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import time

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import wandb_local as wandb


class SimpleNet(nn.Module):
    """简单的神经网络"""
    
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_function(config=None):
    """
    训练函数 - 用于超参数搜索
    
    Args:
        config: 超参数配置字典
        
    Returns:
        dict: 包含最终指标的字典
    """
    # 使用wandb.config获取配置
    if config is None:
        config = wandb.config
    
    print(f"🚀 开始训练 - 配置: {config}")
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建模型
    model = SimpleNet(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        output_size=config["output_size"],
        dropout_rate=config["dropout_rate"]
    )
    
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.001)
    )
    
    # 生成虚拟数据
    n_samples = 1000
    X_train = torch.randn(n_samples, config["input_size"])
    y_train = torch.randint(0, config["output_size"], (n_samples,))
    
    X_val = torch.randn(n_samples // 4, config["input_size"])
    y_val = torch.randint(0, config["output_size"], (n_samples // 4,))
    
    # 训练循环
    best_val_loss = float('inf')
    best_accuracy = 0.0
    
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        n_batches = len(X_train) // config["batch_size"]
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * config["batch_size"]
            end_idx = start_idx + config["batch_size"]
            
            batch_X = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            
            _, predicted = torch.max(val_outputs, 1)
            accuracy = (predicted == y_val).float().mean().item()
        
        # 记录指标
        wandb.log({
            "epoch": epoch,
            "train_loss": total_loss / n_batches,
            "val_loss": val_loss.item(),
            "accuracy": accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })
        
        # 更新最佳指标
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
    
    # 返回最终结果
    return {
        "final_loss": best_val_loss,
        "final_accuracy": best_accuracy,
        "epochs_trained": config["epochs"]
    }


def main():
    """主函数"""
    print("🎯 WandB Local 超参数搜索示例")
    print("=" * 60)
    
    # 定义搜索配置
    sweep_configs = [
        {
            "name": "grid_search_example",
            "method": "grid",
            "metric": {
                "name": "final_accuracy",
                "goal": "maximize"
            },
            "parameters": {
                "learning_rate": {"values": [0.001, 0.01, 0.1]},
                "hidden_size": {"values": [32, 64, 128]},
                "batch_size": {"values": [16, 32, 64]},
                "dropout_rate": {"values": [0.2, 0.5]},
                "epochs": {"value": 5},
                "input_size": {"value": 100},
                "output_size": {"value": 10}
            }
        },
        {
            "name": "random_search_example", 
            "method": "random",
            "metric": {
                "name": "final_loss",
                "goal": "minimize"
            },
            "parameters": {
                "learning_rate": {
                    "distribution": "log_uniform",
                    "min": 0.0001,
                    "max": 0.1
                },
                "hidden_size": {"values": [32, 64, 128, 256]},
                "batch_size": {"values": [16, 32, 64, 128]},
                "dropout_rate": {
                    "distribution": "uniform",
                    "min": 0.1,
                    "max": 0.8
                },
                "weight_decay": {
                    "distribution": "log_uniform", 
                    "min": 0.00001,
                    "max": 0.001
                },
                "epochs": {"value": 8},
                "input_size": {"value": 100},
                "output_size": {"value": 10}
            }
        }
    ]
    
    # 运行不同的搜索策略
    for i, sweep_config in enumerate(sweep_configs):
        print(f"\n🎲 执行搜索 #{i+1}: {sweep_config['name']}")
        print(f"方法: {sweep_config['method']}")
        print(f"目标: {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")
        
        # 初始化实验
        run = wandb.init(
            project="sweep-example",
            name=f"{sweep_config['name']}-{int(time.time())}",
            config=sweep_config,
            tags=["sweep", sweep_config["method"], "hyperparameter-search"]
        )
        
        print(f"✅ 搜索实验已启动: {run.run_id}")
        
        # 执行搜索
        try:
            results = wandb.sweep(
                sweep_config=sweep_config,
                function=train_function,
                project="sweep-example",
                num_workers=1  # 本地版本，使用顺序执行
            )
            
            # 显示搜索结果
            print(f"\n📊 搜索结果摘要:")
            print(f"   总配置数: {results['configs']}")
            print(f"   成功运行: {results['completed']}")
            print(f"   失败运行: {results['failed']}")
            print(f"   最佳运行: {results['best_run']}")
            print(f"   最佳指标: {results['best_metric']:.4f}")
            
            # 保存搜索结果
            results_file = f"sweep_results_{sweep_config['name']}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            wandb.save(results_file)
            print(f"💾 搜索结果已保存: {results_file}")
            
        except Exception as e:
            print(f"❌ 搜索执行失败: {e}")
            wandb.alert(
                title="搜索执行失败",
                text=f"搜索 {sweep_config['name']} 执行失败: {str(e)}",
                level="ERROR"
            )
        
        # 结束当前搜索实验
        wandb.finish()
        print(f"✅ 搜索 #{i+1} 完成")
    
    # 创建对比实验
    print(f"\n🔄 创建对比实验...")
    
    # 手动运行几个有代表性的配置进行对比
    baseline_configs = [
        {"learning_rate": 0.001, "hidden_size": 64, "batch_size": 32, "dropout_rate": 0.5},
        {"learning_rate": 0.01, "hidden_size": 128, "batch_size": 64, "dropout_rate": 0.3},
        {"learning_rate": 0.1, "hidden_size": 32, "batch_size": 16, "dropout_rate": 0.2}
    ]
    
    # 添加固定参数
    for config in baseline_configs:
        config.update({
            "epochs": 10,
            "input_size": 100,
            "output_size": 10,
            "weight_decay": 0.001
        })
    
    comparison_results = []
    
    for i, baseline_config in enumerate(baseline_configs):
        print(f"\n🧪 运行对比实验 #{i+1}")
        
        run = wandb.init(
            project="sweep-example",
            name=f"baseline-comparison-{i}",
            config=baseline_config,
            tags=["baseline", "comparison", "manual"]
        )
        
        result = train_function(baseline_config)
        comparison_results.append({
            "config": baseline_config,
            "result": result,
            "run_id": run.run_id
        })
        
        wandb.finish()
    
    # 创建比较表格
    print(f"📊 创建比较结果表格...")
    
    # 重新初始化一个实验来记录比较结果
    run = wandb.init(
        project="sweep-example",
        name="comparison-summary",
        tags=["summary", "comparison"]
    )
    
    comparison_table = wandb.Table(columns=[
        "experiment_type", "learning_rate", "hidden_size", 
        "batch_size", "dropout_rate", "final_loss", "final_accuracy"
    ])
    
    # 添加搜索结果
    for sweep_config in sweep_configs:
        try:
            # 这里简化处理，实际应该解析详细的搜索结果
            comparison_table.add_data(
                f"sweep_{sweep_config['method']}",
                "various", "various", "various", "various",
                "search_result", "search_result"
            )
        except:
            pass
    
    # 添加基准结果
    for result in comparison_results:
        comparison_table.add_data(
            "baseline_manual",
            result["config"]["learning_rate"],
            result["config"]["hidden_size"],
            result["config"]["batch_size"],
            result["config"]["dropout_rate"],
            result["result"]["final_loss"],
            result["result"]["final_accuracy"]
        )
    
    wandb.log({"comparison_results": comparison_table})
    
    # 发送总结告警
    wandb.alert(
        title="超参数搜索完成",
        text=f"所有搜索实验完成！\n"
              f"运行了 {len(sweep_configs)} 种搜索策略\n"
              f"对比了 {len(baseline_configs)} 个基准配置\n"
              f"详细结果已保存到实验目录",
        level="SUCCESS"
    )
    
    wandb.finish()
    
    print("\n✅ 所有超参数搜索实验完成！")
    print(f"📊 查看详细结果在各实验目录中")
    
    # 显示最佳配置建议
    if comparison_results:
        best_result = min(comparison_results, key=lambda x: x["result"]["final_loss"])
        print(f"\n🏆 最佳手动配置:")
        print(f"   配置: {best_result['config']}")
        print(f"   结果: {best_result['result']}")


if __name__ == "__main__":
    main()