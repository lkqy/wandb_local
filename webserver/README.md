# WandB Local Web Server

功能强大的Web界面，用于可视化和分析WandB Local实验数据。

## ✨ 特性

### 🎯 核心功能
- **实时实验监控** - 自动发现和显示新实验
- **交互式仪表板** - 现代化的用户界面
- **实验详情查看** - 深入分析单个实验数据
- **多媒体展示** - 支持图像、音频、视频预览
- **数据可视化** - 使用Plotly.js的动态图表
- **实验对比** - 多实验并排比较功能

### 🔧 高级功能
- **智能筛选** - 按项目、状态、标签、时间筛选
- **实时搜索** - 全文搜索实验数据
- **WebSocket支持** - 实时数据更新
- **自动刷新** - 后台自动加载新数据
- **响应式设计** - 适配各种屏幕尺寸

## 🚀 快速开始

### 安装依赖
```bash
cd webserver
pip install -r requirements.txt
```

### 启动服务器
```bash
python main.py
```

服务器将在 http://localhost:8000 启动

### 生产环境部署
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📖 使用指南

### 访问Web界面
1. 打开浏览器访问 `http://localhost:8000`
2. 查看实时实验数据
3. 使用筛选器查找特定实验
4. 点击实验卡片查看详细信息

### 主要功能

#### 1. 实验列表
- 显示所有实验的概览信息
- 实时状态指示器
- 关键性能指标展示
- 标签和元数据显示

#### 2. 实验详情
- 完整的实验配置信息
- 性能指标可视化图表
- 训练历史数据展示
- 媒体文件预览
- 实验摘要统计

#### 3. 实验对比
- 选择多个实验进行对比
- 并排显示关键指标
- 统一的性能分析

#### 4. 筛选和搜索
- 按项目名称筛选
- 按实验状态筛选
- 按标签筛选
- 按时间范围筛选
- 全文搜索功能

### API端点

#### 获取实验列表
```http
GET /api/experiments
GET /api/experiments?project=my-project&status=running
```

#### 获取单个实验
```http
GET /api/experiments/{run_id}
```

#### 获取实验历史
```http
GET /api/experiments/{run_id}/history
```

#### 获取媒体文件
```http
GET /api/experiments/{run_id}/media
```

#### 获取Artifact
```http
GET /api/experiments/{run_id}/artifacts
```

#### 刷新实验数据
```http
POST /api/experiments/{run_id}/refresh
```

## 📁 项目结构

```
webserver/
├── main.py              # FastAPI后端主程序
├── requirements.txt     # Python依赖
├── templates/           # HTML模板
│   └── index.html      # 主仪表板页面
├── static/              # 静态资源
│   ├── css/            # 样式文件
│   ├── js/             # JavaScript文件
│   └── images/         # 图片资源
└── README.md           # 本文档
```

## 🔧 配置选项

### 环境变量
```bash
# 实验数据目录
export EXPERIMENTS_DIR=/path/to/experiments

# 服务器配置
export HOST=0.0.0.0
export PORT=8000
export DEBUG=false
```

### 自定义配置
在 `main.py` 中可以修改以下配置：

```python
# 自动发现间隔（秒）
AUTO_DISCOVERY_INTERVAL = 5

# 最大实验数量
MAX_EXPERIMENTS = 1000

# WebSocket配置
WEBSOCKET_HEARTBEAT_INTERVAL = 30
```

## 🎨 界面设计

### 设计理念
- **现代化** - 使用Tailwind CSS构建
- **用户友好** - 直观的交互设计
- **响应式** - 适配各种设备
- **高性能** - 优化的加载和渲染

### 颜色方案
- **主色调** - 蓝色系 (#3b82f6)
- **背景色** - 浅灰系 (#f8fafc)
- **文本色** - 深灰系 (#1e293b)
- **状态色** - 绿色(运行中)、蓝色(完成)、红色(失败)

## 📊 数据可视化

### 支持的图表类型
- **折线图** - 训练损失、准确率趋势
- **散点图** - 参数关系分析
- **柱状图** - 实验对比
- **热力图** - 参数网格搜索

### 图表交互
- **缩放** - 鼠标滚轮缩放
- **平移** - 拖拽移动视图
- **悬停** - 显示详细数据
- **导出** - 保存为PNG/SVG

## 🔒 安全考虑

### 访问控制
- CORS配置保护
- 静态文件安全检查
- API访问限制

### 数据安全
- 文件路径验证
- 内容类型检查
- 大小限制

## ⚡ 性能优化

### 前端优化
- **懒加载** - 按需加载实验数据
- **虚拟滚动** - 处理大量实验列表
- **缓存策略** - 智能数据缓存
- **压缩传输** - GZIP压缩

### 后端优化
- **异步处理** - 非阻塞IO操作
- **连接池** - 数据库连接复用
- **缓存层** - Redis缓存支持
- **负载均衡** - 多工作进程

## 🐛 故障排除

### 常见问题

1. **无法连接到服务器**
   - 检查端口是否被占用
   - 确认防火墙设置
   - 查看服务器日志

2. **实验数据不显示**
   - 检查实验目录路径
   - 确认数据文件格式
   - 查看文件权限

3. **图表加载失败**
   - 检查Plotly.js加载
   - 确认数据格式正确
   - 查看浏览器控制台

### 调试技巧

1. **查看日志**
```bash
tail -f webserver.log
```

2. **浏览器调试**
- F12打开开发者工具
- 查看Console错误信息
- 检查Network请求

3. **API测试**
```bash
curl http://localhost:8000/api/experiments
```

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

### 开发环境设置
```bash
# 克隆项目
git clone <repository-url>
cd webserver

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 启动开发服务器
python main.py
```

### 代码规范
- 使用Black格式化代码
- 遵循PEP 8规范
- 添加类型注解
- 编写单元测试

## 📄 许可证

本项目采用 MIT 许可证。

## 🔮 未来规划

### 即将推出的功能
- [ ] 用户认证和权限管理
- [ ] 实验报告生成
- [ ] 更多图表类型支持
- [ ] 数据导出功能
- [ ] 移动端应用

### 技术路线图
- **v1.1** - 用户系统和权限管理
- **v1.2** - 高级分析和报告功能
- **v1.3** - 移动端支持和PWA
- **v2.0** - 分布式部署和集群支持

---

**文档版本**: 1.0.0  
**最后更新**: 2024年11月7日  
**维护者**: WandB Local Team