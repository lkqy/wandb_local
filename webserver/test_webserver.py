#!/usr/bin/env python3
"""
Web服务器测试脚本
"""

import requests
import json
import time
from pathlib import Path

# 测试配置
BASE_URL = "http://localhost:8000"
TIMEOUT = 10

def test_endpoint(name, method, url, **kwargs):
    """测试单个API端点"""
    print(f"🧪 测试 {name}...")
    
    try:
        response = requests.request(method, url, timeout=TIMEOUT, **kwargs)
        
        if response.status_code == 200:
            print(f"   ✅ {name} - 成功 (状态码: {response.status_code})")
            return True
        else:
            print(f"   ❌ {name} - 失败 (状态码: {response.status_code})")
            print(f"      响应: {response.text[:200]}...")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ {name} - 错误: {e}")
        return False

def test_web_interface():
    """测试Web界面"""
    print("🌐 测试Web界面...")
    
    tests = [
        ("主页面", "GET", f"{BASE_URL}/"),
        ("静态资源", "GET", f"{BASE_URL}/static/js/dashboard.js"),
        ("API文档", "GET", f"{BASE_URL}/docs"),
    ]
    
    results = []
    for name, method, url in tests:
        success = test_endpoint(name, method, url)
        results.append(success)
        time.sleep(0.5)  # 避免请求过快
    
    return all(results)

def test_api_endpoints():
    """测试API端点"""
    print("🔌 测试API端点...")
    
    tests = [
        ("获取实验列表", "GET", f"{BASE_URL}/api/experiments"),
        ("获取项目列表", "GET", f"{BASE_URL}/api/projects"),
    ]
    
    results = []
    for name, method, url in tests:
        success = test_endpoint(name, method, url)
        results.append(success)
        time.sleep(0.5)
    
    return all(results)

def test_experiment_data():
    """测试实验数据API"""
    print("📊 测试实验数据API...")
    
    # 首先获取实验列表
    try:
        response = requests.get(f"{BASE_URL}/api/experiments", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            experiments = data.get('experiments', [])
            
            if experiments:
                # 测试第一个实验的详情
                first_experiment = experiments[0]
                run_id = first_experiment['run_id']
                
                tests = [
                    ("获取实验详情", "GET", f"{BASE_URL}/api/experiments/{run_id}"),
                    ("获取实验历史", "GET", f"{BASE_URL}/api/experiments/{run_id}/history"),
                    ("获取实验摘要", "GET", f"{BASE_URL}/api/experiments/{run_id}/summary"),
                    ("获取媒体文件", "GET", f"{BASE_URL}/api/experiments/{run_id}/media"),
                    ("获取Artifact", "GET", f"{BASE_URL}/api/experiments/{run_id}/artifacts"),
                ]
                
                results = []
                for name, method, url in tests:
                    success = test_endpoint(name, method, url)
                    results.append(success)
                    time.sleep(0.5)
                
                return all(results)
            else:
                print("   ⚠️  没有找到实验数据，跳过实验数据测试")
                return True
        else:
            print(f"   ❌ 获取实验列表失败 (状态码: {response.status_code})")
            return False
            
    except Exception as e:
        print(f"   ❌ 测试实验数据时出错: {e}")
        return False

def test_websocket():
    """测试WebSocket连接"""
    print("🔌 测试WebSocket连接...")
    
    try:
        import websocket
        
        ws_url = "ws://localhost:8000/ws"
        ws = websocket.create_connection(ws_url, timeout=TIMEOUT)
        
        # 发送测试消息
        ws.send("ping")
        
        # 接收响应
        response = ws.recv()
        
        ws.close()
        
        print(f"   ✅ WebSocket连接成功")
        return True
        
    except ImportError:
        print("   ⚠️  未安装websocket-client，跳过WebSocket测试")
        return True
    except Exception as e:
        print(f"   ❌ WebSocket连接失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 WandB Local Web Server 测试")
    print("=" * 50)
    
    # 检查服务器是否运行
    try:
        response = requests.get(BASE_URL, timeout=5)
        if response.status_code != 200:
            print("❌ 服务器未运行或无法访问")
            print("请先启动服务器: python start_server.py")
            return False
    except requests.exceptions.RequestException:
        print("❌ 无法连接到服务器")
        print("请先启动服务器: python start_server.py")
        return False
    
    # 运行测试
    print("✅ 服务器已运行，开始测试...")
    print()
    
    test_results = []
    
    # 测试Web界面
    print("📱 测试Web界面...")
    web_results = test_web_interface()
    test_results.append(web_results)
    print()
    
    # 测试API端点
    print("🔌 测试API端点...")
    api_results = test_api_endpoints()
    test_results.append(api_results)
    print()
    
    # 测试实验数据
    print("📊 测试实验数据...")
    data_results = test_experiment_data()
    test_results.append(data_results)
    print()
    
    # 测试WebSocket
    print("🔌 测试WebSocket...")
    ws_results = test_websocket()
    test_results.append(ws_results)
    print()
    
    # 总结结果
    print("📋 测试结果总结:")
    print("=" * 30)
    
    all_passed = all(test_results)
    
    test_names = [
        "Web界面测试",
        "API端点测试", 
        "实验数据测试",
        "WebSocket测试"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {i+1}. {name}: {status}")
    
    print()
    if all_passed:
        print("🎉 所有测试通过！Web服务器运行正常。")
        print("🌐 访问地址: http://localhost:8000")
    else:
        print("⚠️  部分测试失败，请检查服务器配置和日志。")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)