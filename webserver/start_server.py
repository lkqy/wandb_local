#!/usr/bin/env python3
"""
WandB Local Web Server 启动脚本
"""

import os
import sys
import uvicorn
from pathlib import Path

# 设置正确的路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 确保在webserver目录下运行
os.chdir(Path(__file__).parent)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )