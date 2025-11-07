"""
Utility functions for wandb_local
"""

import os
import json
import time
from pathlib import Path
from typing import Optional


# Global configuration
_global_config = {
    "base_dir": "experiments",
    "current_project": None,
    "current_run": None
}

# 获取当前脚本目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def set_dir(dir: str):
    """Set the base directory for experiments"""
    _global_config["base_dir"] = dir
    os.makedirs(dir, exist_ok=True)


def get_dir() -> str:
    """Get the current base directory"""
    return _global_config["base_dir"]


def get_experiment_dir(project: str, run_name: str) -> str:
    """Get the directory path for an experiment"""
    base_dir = _global_config["base_dir"]
    # 使用绝对路径，相对于当前脚本目录
    return os.path.join(SCRIPT_DIR, "..", base_dir, project, run_name)


def ensure_dir(path: str):
    """Ensure a directory exists"""
    os.makedirs(path, exist_ok=True)


def atomic_write(path: str, content: str):
    """Write content to a file"""
    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 直接写入文件（简化版本，适用于所有环境）
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        # 如果还是失败，打印警告但继续运行
        print(f"Warning: Could not write to {path}: {e}")


def reset():
    """Reset the global state"""
    global _global_config
    _global_config = {
        "base_dir": "experiments",
        "current_project": None,
        "current_run": None
    }
    
    # Reset core state
    from .core import _state
    _state.reset()


def load_run(run_id: str, base_dir: Optional[str] = None):
    """Load a run from disk"""
    if base_dir is None:
        base_dir = _global_config["base_dir"]
        
    # 使用绝对路径
    full_base_dir = os.path.join(SCRIPT_DIR, "..", base_dir)
        
    # Find run directory
    for project_dir in os.listdir(full_base_dir):
        project_path = os.path.join(full_base_dir, project_dir)
        if os.path.isdir(project_path):
            for run_dir in os.listdir(project_path):
                run_path = os.path.join(project_path, run_dir)
                metadata_path = os.path.join(run_path, "wandb-metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    if metadata.get("run_id") == run_id:
                        return run_path
                        
    raise FileNotFoundError(f"Run {run_id} not found")


def list_runs(project: str = None, base_dir: Optional[str] = None) -> list:
    """List all runs"""
    if base_dir is None:
        base_dir = _global_config["base_dir"]
    
    # 使用绝对路径，相对于当前脚本目录
    full_base_dir = os.path.join(SCRIPT_DIR, "..", base_dir)
        
    runs = []
    
    if project:
        projects = [project]
    else:
        try:
            projects = [d for d in os.listdir(full_base_dir) if os.path.isdir(os.path.join(full_base_dir, d))]
        except FileNotFoundError:
            projects = []
        
    for proj in projects:
        project_path = os.path.join(full_base_dir, proj)
        if os.path.exists(project_path):
            try:
                run_dirs = os.listdir(project_path)
            except (FileNotFoundError, PermissionError) as e:
                run_dirs = []
                
            for run_dir in run_dirs:
                run_path = os.path.join(project_path, run_dir)
                metadata_path = os.path.join(run_path, "wandb-metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        runs.append({
                            "run_id": metadata.get("run_id"),
                            "project": metadata.get("project"),
                            "name": metadata.get("name"),
                            "status": metadata.get("status"),
                            "start_time": metadata.get("start_time"),
                            "path": run_path
                        })
                    except (json.JSONDecodeError, IOError):
                        # 如果metadata文件损坏或无法读取，跳过
                        continue
                    
    return runs


def get_summary(run_id: str, base_dir: Optional[str] = None) -> dict:
    """Get summary for a run"""
    try:
        run_path = load_run(run_id, base_dir)
        summary_path = os.path.join(run_path, "summary.json")
        
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                return json.load(f)
        return {}
    except FileNotFoundError:
        return {}


def get_history(run_id: str, base_dir: Optional[str] = None) -> list:
    """Get full history for a run"""
    try:
        run_path = load_run(run_id, base_dir)
        history_path = os.path.join(run_path, "history.jsonl")
        
        history = []
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            history.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                            
        return history
    except FileNotFoundError:
        # 如果运行ID不存在，返回空列表
        return []