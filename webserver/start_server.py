#!/usr/bin/env python3
"""
WandB Local Web Server 启动脚本
"""

import os
import sys
import subprocess
import webbrowser
import time
import signal
import platform
from pathlib import Path

def check_dependencies():
    """检查必要的依赖是否已安装"""
    try:
        import fastapi
        import uvicorn
        import aiofiles
        print("✅ 依赖检查通过")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请先安装依赖: pip install -r requirements.txt")
        return False

def setup_environment():
    """设置环境变量"""
    # 确保实验数据目录存在
    experiments_dir = Path("../experiments")
    experiments_dir.mkdir(exist_ok=True)
    
    # 设置环境变量
    os.environ["EXPERIMENTS_DIR"] = str(experiments_dir.absolute())
    
    print(f"✅ 实验数据目录: {experiments_dir.absolute()}")

def start_server(host="0.0.0.0", port=8000, debug=True):
    """启动Web服务器"""
    print("🚀 启动 WandB Local Web Server...")
    print(f"📡 服务器地址: http://{host}:{port}")
    print(f"🔧 调试模式: {debug}")
    
    # 构建启动命令
    cmd = [
        sys.executable, "-m", "uvicorn",
        "main:app",
        f"--host={host}",
        f"--port={port}",
        "--reload" if debug else "",
        "--log-level=info"
    ]
    
    # 移除空字符串
    cmd = [arg for arg in cmd if arg]
    
    try:
        # 启动服务器进程
        process = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        print("⏳ 等待服务器启动...")
        time.sleep(3)
        
        # 检查进程是否正常运行
        if process.poll() is None:
            print("✅ 服务器启动成功！")
            print(f"🌐 访问地址: http://localhost:{port}")
            
            # 自动打开浏览器（可选）
            if platform.system() == "Darwin":  # macOS
                webbrowser.open(f"http://localhost:{port}")
            elif platform.system() == "Windows":  # Windows
                webbrowser.open(f"http://localhost:{port}")
            else:  # Linux
                try:
                    webbrowser.open(f"http://localhost:{port}")
                except:
                    pass
            
            return process
        else:
            print("❌ 服务器启动失败")
            stdout, stderr = process.communicate()
            if stdout:
                print("STDOUT:", stdout)
            if stderr:
                print("STDERR:", stderr)
            return None
            
    except Exception as e:
        print(f"❌ 启动服务器时出错: {e}")
        return None

def signal_handler(sig, frame):
    """信号处理函数"""
    print("\n🛑 收到停止信号，正在关闭服务器...")
    if server_process:
        server_process.terminate()
        server_process.wait()
    print("✅ 服务器已关闭")
    sys.exit(0)

def main():
    """主函数"""
    print("🎛️  WandB Local Web Server 启动器")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return 1
    
    # 设置环境
    setup_environment()
    
    # 配置参数
    host = "0.0.0.0"
    port = 8000
    debug = True
    
    print("\n📋 启动配置:")
    print(f"   主机: {host}")
    print(f"   端口: {port}")
    print(f"   调试模式: {debug}")
    
    # 设置信号处理
    global server_process
    server_process = None
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动服务器
    server_process = start_server(host, port, debug)
    
    if server_process:
        print("\n🎉 服务器已成功启动！")
        print("💡 提示:")
        print("   - 按 Ctrl+C 停止服务器")
        print("   - 查看控制台输出获取更多信息")
        print("   - 在浏览器中访问 http://localhost:8000")
        
        try:
            # 等待服务器进程
            server_process.wait()
        except KeyboardInterrupt:
            signal_handler(None, None)
    else:
        print("\n❌ 服务器启动失败")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())