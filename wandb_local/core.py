"""
Core WandB compatible APIs
"""

import os
import json
import time
import pickle
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
from .utils import get_experiment_dir, ensure_dir, atomic_write


class _ExperimentState:
    """Global experiment state management"""
    
    def __init__(self):
        self.current_run = None
        self.runs = {}
        self.lock = threading.RLock()
        
    def set_run(self, run):
        with self.lock:
            self.current_run = run
            if run and run.run_id:
                self.runs[run.run_id] = run
                
    def get_run(self, run_id=None):
        with self.lock:
            if run_id:
                return self.runs.get(run_id)
            return self.current_run
            
    def reset(self):
        with self.lock:
            self.current_run = None


# Global state
_state = _ExperimentState()


class Run:
    """Represents a single experiment run"""
    
    def __init__(self, project: str = None, name: str = None, config: Dict = None, 
                 dir: str = None, tags: List[str] = None, notes: str = None,
                 group: str = None, job_type: str = None, **kwargs):
        self.project = project or "uncategorized"
        self.name = name or f"run_{int(time.time())}"
        self.run_id = f"{self.project}_{self.name}_{int(time.time())}"
        self.config = config or {}
        self.tags = tags or []
        self.notes = notes or ""
        self.group = group
        self.job_type = job_type
        self.start_time = datetime.now()
        self.dir = dir or get_experiment_dir(self.project, self.name)
        self.kwargs = kwargs
        
        # Internal state
        self.history = []
        self.summary = {}
        self.media_files = []
        self.artifacts = []
        self.model_hooks = []
        self.alerts = []
        self.is_finished = False
        
        # Setup directories
        ensure_dir(self.dir)
        ensure_dir(os.path.join(self.dir, "media"))
        ensure_dir(os.path.join(self.dir, "artifacts"))
        
        # Save metadata (with error handling)
        try:
            self._save_metadata()
        except Exception as e:
            print(f"Warning: Could not save metadata during initialization: {e}")
            # 继续运行而不是失败
        
    def _save_metadata(self):
        """Save run metadata"""
        metadata = {
            "run_id": self.run_id,
            "project": self.project,
            "name": self.name,
            "config": self.config,
            "tags": self.tags,
            "notes": self.notes,
            "group": self.group,
            "job_type": self.job_type,
            "start_time": self.start_time.isoformat(),
            "dir": self.dir,
            "status": "running"
        }
        
        try:
            metadata_path = os.path.join(self.dir, "wandb-metadata.json")
            atomic_write(metadata_path, json.dumps(metadata, indent=2))
            
            # Save config separately
            config_path = os.path.join(self.dir, "config.json")
            atomic_write(config_path, json.dumps(self.config, indent=2))
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
            # 在无法保存元数据时，继续运行而不是失败
            
    def log(self, data: Dict[str, Any], step: int = None, commit: bool = True):
        """Log metrics and data"""
        if self.is_finished:
            raise RuntimeError("Cannot log to finished run")
            
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step if step is not None else len(self.history),
            "data": {}
        }
        
        # Process data (handle special types)
        for key, value in data.items():
            if hasattr(value, 'to_json'):
                # Handle wandb objects (Table, Image, etc.)
                log_entry["data"][key] = value.to_json()
                if hasattr(value, 'save'):
                    value.save(self.dir)
            elif isinstance(value, (np.ndarray, np.generic)):
                log_entry["data"][key] = float(value)
            elif isinstance(value, (list, tuple)):
                log_entry["data"][key] = [float(v) if isinstance(v, (np.ndarray, np.generic)) else v for v in value]
            else:
                log_entry["data"][key] = value
                
        self.history.append(log_entry)
        
        # Update summary (keep latest values)
        for key, value in log_entry["data"].items():
            if not isinstance(value, dict) or not value.get('_type'):  # Don't summarize media objects
                self.summary[key] = value
                
        if commit:
            self._save_history()
            
    def _save_history(self):
        """Save history to file"""
        try:
            history_path = os.path.join(self.dir, "history.jsonl")
            with open(history_path, 'a') as f:
                for entry in self.history[-len(self.history):]:  # Only save new entries
                    f.write(json.dumps(entry) + '\n')
                    
            # Save summary
            summary_path = os.path.join(self.dir, "summary.json")
            atomic_write(summary_path, json.dumps(self.summary, indent=2))
        except Exception as e:
            print(f"Warning: Could not save history: {e}")
            # 在无法保存历史记录时，继续运行而不是失败
            
    def save(self, path: str, policy: str = "live", base_path: str = None):
        """Save a file or directory"""
        if base_path:
            full_path = os.path.join(base_path, path)
        else:
            full_path = path
            
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Path does not exist: {full_path}")
            
        # Copy to run directory
        dest_path = os.path.join(self.dir, "files", os.path.basename(path))
        ensure_dir(os.path.dirname(dest_path))
        
        if os.path.isdir(full_path):
            import shutil
            shutil.copytree(full_path, dest_path, dirs_exist_ok=True)
        else:
            import shutil
            shutil.copy2(full_path, dest_path)
            
        # Record in run metadata
        if "saved_files" not in self.config:
            self.config["saved_files"] = []
        self.config["saved_files"].append({
            "path": path,
            "policy": policy,
            "timestamp": datetime.now().isoformat()
        })
        
        self._save_metadata()
        return dest_path
        
    def watch(self, models, criterion=None, log: str = "gradients", 
              log_freq: int = 1000, idx: int = None):
        """Watch models for gradient/parameter logging"""
        import torch
        
        if not isinstance(models, (list, tuple)):
            models = [models]
            
        for i, model in enumerate(models):
            if isinstance(model, torch.nn.Module):
                hook_data = {
                    "model_idx": idx if idx is not None else i,
                    "log_freq": log_freq,
                    "log_gradients": "gradients" in log,
                    "log_parameters": "parameters" in log,
                    "hooks": []
                }
                
                def make_hook(name, is_param):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            output = output[0]
                        if hasattr(output, 'grad_fn'):
                            step = len(self.history)
                            if step % log_freq == 0:
                                # Log gradient norms
                                if is_param and output.grad is not None:
                                    grad_norm = output.grad.norm().item()
                                    self.log({f"{name}/grad_norm": grad_norm}, step=step)
                                # Log parameter stats
                                param_norm = output.norm().item()
                                self.log({f"{name}/param_norm": param_norm}, step=step)
                    return hook
                
                # Register hooks
                for name, module in model.named_modules():
                    if len(list(module.children())) == 0:  # Leaf modules
                        hook = module.register_forward_hook(make_hook(name, False))
                        hook_data["hooks"].append(hook)
                        
                self.model_hooks.append(hook_data)
                
    def alert(self, title: str, text: str, level: str = "INFO", 
              wait_duration: int = 0):
        """Send an alert"""
        alert_data = {
            "title": title,
            "text": text,
            "level": level,
            "wait_duration": wait_duration,
            "timestamp": datetime.now().isoformat()
        }
        self.alerts.append(alert_data)
        
        # Save alerts
        try:
            alerts_path = os.path.join(self.dir, "alerts.jsonl")
            with open(alerts_path, 'a') as f:
                f.write(json.dumps(alert_data) + '\n')
        except Exception as e:
            print(f"Warning: Could not save alerts: {e}")
            
        print(f"[ALERT {level}] {title}: {text}")
        
    def finish(self, exit_code: int = 0):
        """Finish the run"""
        if self.is_finished:
            return
            
        self.is_finished = True
        self.end_time = datetime.now()
        
        # Clean up model hooks
        for hook_data in self.model_hooks:
            for hook in hook_data["hooks"]:
                hook.remove()
                
        # Update metadata
        try:
            metadata_path = os.path.join(self.dir, "wandb-metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                metadata["status"] = "finished"
                metadata["end_time"] = self.end_time.isoformat()
                metadata["exit_code"] = exit_code
                atomic_write(metadata_path, json.dumps(metadata, indent=2))
        except Exception as e:
            print(f"Warning: Could not update metadata on finish: {e}")
            
        print(f"Run {self.run_id} finished successfully")
        _state.set_run(None)


# Core API functions
def init(project: str = None, name: str = None, config: Dict = None,
         dir: str = None, tags: List[str] = None, notes: str = None,
         group: str = None, job_type: str = None, reinit: bool = None,
         id: str = None, resume: Union[bool, str] = None,
         **kwargs) -> Run:
    """Initialize a new run"""
    
    if reinit or (resume is None and _state.current_run is None):
        # Create new run
        run = Run(project=project, name=name, config=config, dir=dir,
                 tags=tags, notes=notes, group=group, job_type=job_type, **kwargs)
        _state.set_run(run)
        print(f"Initialized run: {run.run_id}")
        return run
    elif resume and _state.current_run:
        # Resume existing run
        return _state.current_run
    else:
        # Return current run
        return _state.current_run


def log(data: Dict[str, Any], step: int = None, commit: bool = True, **kwargs):
    """Log data to current run"""
    run = _state.get_run()
    if run is None:
        raise RuntimeError("No active run. Call wandb.init() first.")
    run.log(data, step=step, commit=commit, **kwargs)


def finish(exit_code: int = 0, **kwargs):
    """Finish current run"""
    run = _state.get_run()
    if run:
        run.finish(exit_code=exit_code, **kwargs)


def save(path: str, policy: str = "live", base_path: str = None, **kwargs):
    """Save file in current run"""
    run = _state.get_run()
    if run is None:
        raise RuntimeError("No active run. Call wandb.init() first.")
    return run.save(path, policy=policy, base_path=base_path, **kwargs)


def watch(models, criterion=None, log: str = "gradients", 
          log_freq: int = 1000, idx: int = None, **kwargs):
    """Watch models"""
    run = _state.get_run()
    if run is None:
        raise RuntimeError("No active run. Call wandb.init() first.")
    run.watch(models, criterion=criterion, log=log, 
              log_freq=log_freq, idx=idx, **kwargs)


def alert(title: str, text: str, level: str = "INFO", 
          wait_duration: int = 0, **kwargs):
    """Send alert"""
    run = _state.get_run()
    if run is None:
        # Allow alerts without active run for debugging
        print(f"[ALERT {level}] {title}: {text}")
        return
    run.alert(title, text, level=level, wait_duration=wait_duration, **kwargs)


# Config property (compatible with wandb.config)
class _ConfigProxy:
    """Proxy for accessing current run config"""
    
    def __getitem__(self, key):
        run = _state.get_run()
        if run is None:
            raise RuntimeError("No active run. Call wandb.init() first.")
        return run.config[key]
        
    def __setitem__(self, key, value):
        run = _state.get_run()
        if run is None:
            raise RuntimeError("No active run. Call wandb.init() first.")
        run.config[key] = value
        run._save_metadata()
        
    def get(self, key, default=None):
        run = _state.get_run()
        if run is None:
            raise RuntimeError("No active run. Call wandb.init() first.")
        return run.config.get(key, default)
        
    def update(self, *args, **kwargs):
        run = _state.get_run()
        if run is None:
            raise RuntimeError("No active run. Call wandb.init() first.")
        run.config.update(*args, **kwargs)
        run._save_metadata()
        
    def __contains__(self, key):
        run = _state.get_run()
        if run is None:
            raise RuntimeError("No active run. Call wandb.init() first.")
        return key in run.config
        
    def keys(self):
        run = _state.get_run()
        if run is None:
            raise RuntimeError("No active run. Call wandb.init() first.")
        return run.config.keys()
        
    def items(self):
        run = _state.get_run()
        if run is None:
            raise RuntimeError("No active run. Call wandb.init() first.")
        return run.config.items()


# Global config object
config = _ConfigProxy()