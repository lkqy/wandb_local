"""
Sweep functionality for hyperparameter search
"""

import os
import json
import time
import itertools
from typing import Dict, Any, List, Callable, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import numpy as np

from .core import init, finish, _state


class SweepConfig:
    """Configuration for hyperparameter sweep"""
    
    def __init__(self, method: str = "grid", metric: Dict[str, str] = None,
                 parameters: Dict[str, Dict] = None, **kwargs):
        self.method = method  # "grid", "random", "bayes"
        self.metric = metric or {"name": "loss", "goal": "minimize"}
        self.parameters = parameters or {}
        self.kwargs = kwargs
        
        # Sweep state
        self.sweep_id = f"sweep_{int(time.time())}"
        self.runs = []
        self.best_run = None
        self.best_metric = None
        
    def generate_configs(self, num_samples: int = None) -> List[Dict[str, Any]]:
        """Generate parameter configurations for sweep"""
        configs = []
        
        if self.method == "grid":
            # Cartesian product of all parameter values
            param_names = list(self.parameters.keys())
            param_values = []
            
            for param_name, param_config in self.parameters.items():
                if "values" in param_config:
                    param_values.append(param_config["values"])
                elif "min" in param_config and "max" in param_config:
                    # Generate discrete values for grid search
                    if param_config.get("distribution") == "uniform":
                        step = param_config.get("q", 1.0)
                        values = list(np.arange(param_config["min"], 
                                              param_config["max"] + step, step))
                    elif param_config.get("distribution") == "log_uniform":
                        step = param_config.get("q", 1.0)
                        log_min = np.log(param_config["min"])
                        log_max = np.log(param_config["max"])
                        log_values = np.arange(log_min, log_max + np.log(step), np.log(step))
                        values = [np.exp(v) for v in log_values]
                    else:
                        # Linear grid
                        num_points = param_config.get("num_points", 10)
                        values = np.linspace(param_config["min"], param_config["max"], num_points)
                        values = values.tolist()
                    param_values.append(values)
                else:
                    raise ValueError(f"Invalid parameter config for {param_name}")
                    
            # Generate all combinations
            for combination in itertools.product(*param_values):
                config = dict(zip(param_names, combination))
                configs.append(config)
                
        elif self.method == "random":
            # Random sampling
            if num_samples is None:
                num_samples = 100
                
            for _ in range(num_samples):
                config = {}
                for param_name, param_config in self.parameters.items():
                    if "values" in param_config:
                        config[param_name] = random.choice(param_config["values"])
                    elif "min" in param_config and "max" in param_config:
                        if param_config.get("distribution") == "uniform":
                            config[param_name] = random.uniform(param_config["min"], param_config["max"])
                        elif param_config.get("distribution") == "log_uniform":
                            log_min = np.log(param_config["min"])
                            log_max = np.log(param_config["max"])
                            config[param_name] = np.exp(random.uniform(log_min, log_max))
                        elif param_config.get("distribution") == "normal":
                            mean = param_config.get("mean", (param_config["min"] + param_config["max"]) / 2)
                            std = param_config.get("std", (param_config["max"] - param_config["min"]) / 6)
                            config[param_name] = random.gauss(mean, std)
                            # Clip to bounds
                            config[param_name] = max(param_config["min"], min(param_config["max"], config[param_name]))
                        else:
                            config[param_name] = random.uniform(param_config["min"], param_config["max"])
                    else:
                        raise ValueError(f"Invalid parameter config for {param_name}")
                        
                configs.append(config)
                
        elif self.method == "bayes":
            # Simplified Bayesian optimization (random for now)
            if num_samples is None:
                num_samples = 50
                
            # For a real implementation, you'd use a library like scikit-optimize
            for _ in range(num_samples):
                config = {}
                for param_name, param_config in self.parameters.items():
                    if "values" in param_config:
                        config[param_name] = random.choice(param_config["values"])
                    elif "min" in param_config and "max" in param_config:
                        config[param_name] = random.uniform(param_config["min"], param_config["max"])
                configs.append(config)
                
        return configs
        
    def save_state(self, base_dir: str):
        """Save sweep state"""
        sweep_dir = os.path.join(base_dir, "sweeps", self.sweep_id)
        os.makedirs(sweep_dir, exist_ok=True)
        
        state = {
            "sweep_id": self.sweep_id,
            "method": self.method,
            "metric": self.metric,
            "parameters": self.parameters,
            "runs": self.runs,
            "best_run": self.best_run,
            "best_metric": self.best_metric
        }
        
        state_path = os.path.join(sweep_dir, "sweep_state.json")
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)


class _SweepAgent:
    """Agent that runs a single sweep configuration"""
    
    def __init__(self, sweep_id: str, config: Dict[str, Any], function: Callable,
                 project: str = None, count: int = None):
        self.sweep_id = sweep_id
        self.config = config
        self.function = function
        self.project = project
        self.count = count
        
    def run(self) -> Dict[str, Any]:
        """Run the configuration and return results"""
        # Initialize run with sweep configuration
        run = init(
            project=self.project,
            name=f"{self.sweep_id}_run_{int(time.time())}",
            config=self.config
        )
        
        try:
            # Run the training function
            result = self.function(config=self.config)
            
            # Extract metric from result
            metric_name = run.config.get("metric", {}).get("name", "loss")
            metric_value = result.get(metric_name) if isinstance(result, dict) else result
            
            # Log results
            if isinstance(result, dict):
                from .core import log
                log(result)
            else:
                from .core import log
                log({metric_name: result})
                
            return {
                "run_id": run.run_id,
                "config": self.config,
                "result": result,
                "metric_value": metric_value,
                "success": True
            }
            
        except Exception as e:
            return {
                "run_id": getattr(run, 'run_id', 'unknown'),
                "config": self.config,
                "error": str(e),
                "success": False
            }
            
        finally:
            finish()


def sweep(sweep_config: Union[Dict[str, Any], SweepConfig], function: Callable,
          project: str = None, entity: str = None, **kwargs):
    """
    Run hyperparameter sweep
    
    Args:
        sweep_config: Sweep configuration dict or SweepConfig object
        function: Training function that takes config dict and returns metric
        project: Project name
        entity: Entity name (ignored in local version)
        **kwargs: Additional arguments
    """
    
    # Convert to SweepConfig if needed
    if isinstance(sweep_config, dict):
        sweep_config = SweepConfig(**sweep_config)
        
    # Generate configurations
    configs = sweep_config.generate_configs()
    print(f"Generated {len(configs)} configurations for sweep {sweep_config.sweep_id}")
    
    # Get current run directory for saving
    current_run = _state.get_run()
    base_dir = current_run.dir if current_run else "experiments"
    
    # Run sweep configurations
    results = []
    
    # Support parallel execution
    num_workers = kwargs.get('num_workers', 1)
    
    if num_workers > 1:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            for i, config in enumerate(configs):
                agent = _SweepAgent(
                    sweep_id=sweep_config.sweep_id,
                    config=config,
                    function=function,
                    project=project
                )
                future = executor.submit(agent.run)
                futures.append(future)
                
            # Collect results
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                # Update best run
                if result["success"]:
                    metric_value = result["metric_value"]
                    if sweep_config.best_metric is None:
                        sweep_config.best_metric = metric_value
                        sweep_config.best_run = result["run_id"]
                    else:
                        if sweep_config.metric.get("goal") == "maximize":
                            if metric_value > sweep_config.best_metric:
                                sweep_config.best_metric = metric_value
                                sweep_config.best_run = result["run_id"]
                        else:  # minimize
                            if metric_value < sweep_config.best_metric:
                                sweep_config.best_metric = metric_value
                                sweep_config.best_run = result["run_id"]
                
                print(f"Completed run {result['run_id']}: {result.get('metric_value', 'N/A')}")
                
    else:
        # Sequential execution
        for i, config in enumerate(configs):
            print(f"Running configuration {i+1}/{len(configs)}")
            
            agent = _SweepAgent(
                sweep_id=sweep_config.sweep_id,
                config=config,
                function=function,
                project=project
            )
            
            result = agent.run()
            results.append(result)
            
            # Update best run
            if result["success"]:
                metric_value = result["metric_value"]
                if sweep_config.best_metric is None:
                    sweep_config.best_metric = metric_value
                    sweep_config.best_run = result["run_id"]
                else:
                    if sweep_config.metric.get("goal") == "maximize":
                        if metric_value > sweep_config.best_metric:
                            sweep_config.best_metric = metric_value
                            sweep_config.best_run = result["run_id"]
                    else:  # minimize
                        if metric_value < sweep_config.best_metric:
                            sweep_config.best_metric = metric_value
                            sweep_config.best_run = result["run_id"]
                            
            print(f"Completed run {result['run_id']}: {result.get('metric_value', 'N/A')}")
            
    # Update sweep state
    sweep_config.runs = results
    sweep_config.save_state(base_dir)
    
    # Print summary
    print(f"\nSweep {sweep_config.sweep_id} completed!")
    print(f"Best run: {sweep_config.best_run}")
    print(f"Best {sweep_config.metric['name']}: {sweep_config.best_metric}")
    
    # Return sweep results
    return {
        "sweep_id": sweep_config.sweep_id,
        "configs": len(configs),
        "completed": len([r for r in results if r["success"]]),
        "failed": len([r for r in results if not r["success"]]),
        "best_run": sweep_config.best_run,
        "best_metric": sweep_config.best_metric,
        "results": results
    }