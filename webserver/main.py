"""
WandB Local Web Server - FastAPI Backend
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import aiofiles

# 添加父目录到路径，以便导入wandb_local
sys.path.insert(0, str(Path(__file__).parent.parent))

from wandb_local.utils import list_runs, get_summary, get_history, load_run


@dataclass
class ExperimentData:
    """实验数据结构"""
    run_id: str
    project: str
    name: str
    status: str
    start_time: str
    end_time: Optional[str] = None
    config: Dict[str, Any] = None
    summary: Dict[str, Any] = None
    history: List[Dict[str, Any]] = None
    tags: List[str] = None
    notes: str = ""
    path: str = ""


class ExperimentManager:
    """实验管理器 - 自动发现和加载实验"""
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = base_dir
        print(self.base_dir)
        self.experiments: Dict[str, ExperimentData] = {}
        self.websocket_connections: List[WebSocket] = []
        self._running = True
        
    async def start_auto_discovery(self):
        """启动自动发现机制"""
        while self._running:
            await self._discover_experiments()
            await asyncio.sleep(5)  # 每5秒检查一次
            
    async def stop_auto_discovery(self):
        """停止自动发现"""
        self._running = False
        
    async def _discover_experiments(self):
        """发现新的实验"""
        try:
            runs = list_runs(base_dir=self.base_dir)
            
            for run_info in runs:
                run_id = run_info["run_id"]
                
                # 如果是新实验或状态有变化
                if (run_id not in self.experiments or 
                    self.experiments[run_id].status != run_info.get("status", "unknown")):
                    
                    # 加载完整实验数据
                    experiment = await self._load_experiment(run_id)
                    if experiment:
                        self.experiments[run_id] = experiment
                        
                        # 通知WebSocket客户端
                        await self._notify_new_experiment(experiment)
                        
        except Exception as e:
            print(f"Error discovering experiments: {e}")
            
    async def _load_experiment(self, run_id: str) -> Optional[ExperimentData]:
        """加载单个实验的完整数据"""
        try:
            # 获取实验基本信息
            runs = list_runs(base_dir=self.base_dir)
            run_info = next((r for r in runs if r["run_id"] == run_id), None)
            
            if not run_info:
                return None
                
            # 加载配置和摘要
            config = {}
            summary = {}
            metadata_path = os.path.join(run_info["path"], "wandb-metadata.json")
            
            if os.path.exists(metadata_path):
                async with aiofiles.open(metadata_path, 'r') as f:
                    content = await f.read()
                    metadata = json.loads(content)
                    config = metadata.get("config", {})
                    tags = metadata.get("tags", [])
                    notes = metadata.get("notes", "")
                    
            # 加载摘要
            try:
                summary = get_summary(run_id, base_dir=self.base_dir)
            except:
                summary = {}
                
            # 加载历史数据
            history = []
            try:
                history = get_history(run_id, base_dir=self.base_dir)
            except:
                pass
                
            return ExperimentData(
                run_id=run_id,
                project=run_info.get("project", "unknown"),
                name=run_info.get("name", "unknown"),
                status=run_info.get("status", "unknown"),
                start_time=run_info.get("start_time", ""),
                end_time=metadata.get("end_time") if "end_time" in metadata else None,
                config=config,
                summary=summary,
                history=history,
                tags=tags,
                notes=notes,
                path=run_info["path"]
            )
            
        except Exception as e:
            print(f"Error loading experiment {run_id}: {e}")
            return None
            
    async def _notify_new_experiment(self, experiment: ExperimentData):
        """通知WebSocket客户端新实验"""
        if self.websocket_connections:
            message = {
                "type": "new_experiment",
                "data": experiment.__dict__
            }
            
            disconnected = []
            for websocket in self.websocket_connections:
                try:
                    await websocket.send_json(message)
                except:
                    disconnected.append(websocket)
                    
            # 移除断开的连接
            for ws in disconnected:
                self.websocket_connections.remove(ws)
                
    def get_experiment(self, run_id: str) -> Optional[ExperimentData]:
        """获取单个实验"""
        return self.experiments.get(run_id)
        
    def get_all_experiments(self) -> List[ExperimentData]:
        """获取所有实验"""
        return list(self.experiments.values())
        
    def get_projects(self) -> List[str]:
        """获取所有项目"""
        return list(set(exp.project for exp in self.experiments.values()))
        
    async def refresh_experiment(self, run_id: str):
        """刷新单个实验数据"""
        if run_id in self.experiments:
            updated = await self._load_experiment(run_id)
            if updated:
                self.experiments[run_id] = updated


# 全局实验管理器实例
experiment_manager = ExperimentManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    discovery_task = asyncio.create_task(experiment_manager.start_auto_discovery())
    yield
    # 关闭时
    await experiment_manager.stop_auto_discovery()
    discovery_task.cancel()


# 创建FastAPI应用
app = FastAPI(
    title="WandB Local Web Server",
    description="Web interface for WandB Local experiment tracking",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """主仪表板页面"""
    return FileResponse("templates/index.html")


@app.get("/api/experiments")
async def get_experiments(
    project: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """获取实验列表"""
    experiments = experiment_manager.get_all_experiments()
    
    # 筛选
    if project:
        experiments = [exp for exp in experiments if exp.project == project]
    if status:
        experiments = [exp for exp in experiments if exp.status == status]
        
    # 限制数量
    experiments = experiments[:limit]
    
    return {
        "experiments": [exp.__dict__ for exp in experiments],
        "total": len(experiments)
    }


@app.get("/api/experiments/{run_id}")
async def get_experiment(run_id: str):
    """获取单个实验详情"""
    experiment = experiment_manager.get_experiment(run_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    return experiment.__dict__


@app.get("/api/projects")
async def get_projects():
    """获取所有项目"""
    projects = experiment_manager.get_projects()
    return {"projects": projects}


@app.get("/api/experiments/{run_id}/history")
async def get_experiment_history(run_id: str):
    """获取实验历史数据"""
    experiment = experiment_manager.get_experiment(run_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    return {"history": experiment.history}


@app.get("/api/experiments/{run_id}/summary")
async def get_experiment_summary(run_id: str):
    """获取实验摘要"""
    experiment = experiment_manager.get_experiment(run_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    return {"summary": experiment.summary}


@app.get("/api/experiments/{run_id}/media")
async def get_experiment_media(run_id: str):
    """获取实验媒体文件"""
    experiment = experiment_manager.get_experiment(run_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    media_dir = os.path.join(experiment.path, "media")
    media_files = []
    
    if os.path.exists(media_dir):
        for root, dirs, files in os.walk(media_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, media_dir)
                media_files.append({
                    "name": file,
                    "path": rel_path,
                    "type": _get_file_type(file),
                    "size": os.path.getsize(file_path),
                    "modified": os.path.getmtime(file_path)
                })
                
    return {"media": media_files}


@app.get("/api/experiments/{run_id}/artifacts")
async def get_experiment_artifacts(run_id: str):
    """获取实验Artifact"""
    experiment = experiment_manager.get_experiment(run_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    artifacts_dir = os.path.join(experiment.path, "artifacts")
    artifacts = []
    
    if os.path.exists(artifacts_dir):
        for artifact_name in os.listdir(artifacts_dir):
            artifact_path = os.path.join(artifacts_dir, artifact_name)
            if os.path.isdir(artifact_path):
                metadata_path = os.path.join(artifact_path, "artifact_metadata.json")
                metadata = {}
                if os.path.exists(metadata_path):
                    async with aiofiles.open(metadata_path, 'r') as f:
                        content = await f.read()
                        metadata = json.loads(content)
                        
                artifacts.append({
                    "name": artifact_name,
                    "path": artifact_path,
                    "metadata": metadata
                })
                
    return {"artifacts": artifacts}


@app.post("/api/experiments/{run_id}/refresh")
async def refresh_experiment(run_id: str):
    """刷新实验数据"""
    await experiment_manager.refresh_experiment(run_id)
    return {"status": "success"}


@app.get("/api/experiments/{run_id}/files/{file_path:path}")
async def get_experiment_file(run_id: str, file_path: str):
    """获取实验文件"""
    experiment = experiment_manager.get_experiment(run_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    full_path = os.path.join(experiment.path, file_path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(full_path)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket连接端点"""
    await websocket.accept()
    experiment_manager.websocket_connections.append(websocket)
    
    try:
        while True:
            # 保持连接活跃
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        experiment_manager.websocket_connections.remove(websocket)


def _get_file_type(filename: str) -> str:
    """获取文件类型"""
    ext = os.path.splitext(filename)[1].lower()
    
    image_exts = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
    audio_exts = ['.wav', '.mp3', '.ogg', '.flac', '.aac']
    video_exts = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm']
    
    if ext in image_exts:
        return "image"
    elif ext in audio_exts:
        return "audio"
    elif ext in video_exts:
        return "video"
    elif ext == '.json':
        return "json"
    else:
        return "other"


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
