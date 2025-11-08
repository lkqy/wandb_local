// 实验详情页面 JavaScript
class ExperimentDetailManager {
    constructor() {
        this.experiment = null;
        this.runId = null;
        this.charts = {};
        
        this.init();
    }
    
    init() {
        this.extractRunId();
        this.setupEventListeners();
        this.loadExperiment();
    }
    
    extractRunId() {
        // 从URL中提取运行ID
        const pathParts = window.location.pathname.split('/');
        this.runId = pathParts[pathParts.length - 1];
    }
    
    setupEventListeners() {
        // 标签页切换
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', (e) => {
                const tab = e.target.dataset.tab;
                this.switchTab(tab);
            });
        });
        
        // 刷新按钮
        document.getElementById('refresh-btn').addEventListener('click', () => {
            this.loadExperiment();
        });
        
        // 加入对比按钮
        document.getElementById('compare-btn').addEventListener('click', () => {
            this.addToComparison();
        });
    }
    
    async loadExperiment() {
        this.showLoading(true);
        
        try {
            const response = await fetch(`/api/experiments/${this.runId}`);
            if (!response.ok) {
                throw new Error('Experiment not found');
            }
            
            this.experiment = await response.json();
            this.renderExperiment();
            
        } catch (error) {
            console.error('Error loading experiment:', error);
            this.showError('无法加载实验数据');
        } finally {
            this.showLoading(false);
        }
    }
    
    renderExperiment() {
        if (!this.experiment) return;
        
        // 渲染头部信息
        this.renderHeader();
        
        // 渲染统计信息
        this.renderStats();
        
        // 渲染默认标签页（概览）
        this.switchTab('overview');
    }
    
    renderHeader() {
        const exp = this.experiment;
        
        document.getElementById('experiment-name').textContent = exp.name;
        document.getElementById('experiment-project').textContent = `项目: ${exp.project}`;
        document.getElementById('experiment-status').innerHTML = `
            <span class="status-indicator status-${exp.status}"></span>
            <span class="capitalize">${exp.status}</span>
        `;
        document.getElementById('experiment-time').textContent = 
            `开始于 ${this.formatDate(exp.start_time)}`;
        
        // 渲染标签
        const tagsContainer = document.getElementById('experiment-tags');
        if (exp.tags && exp.tags.length > 0) {
            tagsContainer.innerHTML = exp.tags.map(tag => 
                `<span class="tag">${tag}</span>`
            ).join('');
        }
        
        // 渲染描述
        const descriptionContainer = document.getElementById('experiment-description');
        if (exp.notes) {
            descriptionContainer.textContent = exp.notes;
        } else {
            descriptionContainer.textContent = '暂无实验描述';
        }
    }
    
    renderStats() {
        const exp = this.experiment;
        const history = exp.history || [];
        
        // 计算统计信息
        const epochs = Math.max(...history.map(h => h.epoch || 0), 0);
        const duration = this.calculateDuration(exp.start_time, exp.end_time);
        
        // 寻找最佳指标
        const losses = history.map(h => h.train_loss).filter(v => v > 0);
        const accuracies = history.map(h => h.accuracy).filter(v => v > 0);
        
        const bestLoss = losses.length > 0 ? Math.min(...losses) : 0;
        const bestAccuracy = accuracies.length > 0 ? Math.max(...accuracies) : 0;
        
        // 更新显示
        document.getElementById('metric-epochs').textContent = epochs;
        document.getElementById('metric-duration').textContent = duration;
        document.getElementById('metric-best-loss').textContent = bestLoss.toFixed(4);
        document.getElementById('metric-best-accuracy').textContent = `${(bestAccuracy * 100).toFixed(1)}%`;
    }
    
    switchTab(tabName) {
        // 更新标签按钮状态
        document.querySelectorAll('.tab-button').forEach(button => {
            button.classList.toggle('active', button.dataset.tab === tabName);
        });
        
        // 显示对应内容
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('hidden', content.id !== `${tabName}-tab`);
        });
        
        // 加载标签页内容
        this.loadTabContent(tabName);
    }
    
    async loadTabContent(tabName) {
        switch (tabName) {
            case 'overview':
                this.renderOverviewTab();
                break;
            case 'metrics':
                this.renderMetricsTab();
                break;
            case 'config':
                this.renderConfigTab();
                break;
            case 'media':
                this.renderMediaTab();
                break;
            case 'artifacts':
                this.renderArtifactsTab();
                break;
            case 'logs':
                this.renderLogsTab();
                break;
        }
    }
    
    renderOverviewTab() {
        this.renderTrainingChart();
        this.renderMetricsTable();
    }
    
    renderTrainingChart() {
        const history = this.experiment.history || [];
        
        if (history.length === 0) {
            document.getElementById('training-chart').innerHTML = 
                '<div class="text-center text-gray-400 py-8">暂无训练数据</div>';
            return;
        }
        
        // 准备数据
        const epochs = history.map(h => h.epoch);
        const trainLoss = history.map(h => h.train_loss);
        const valLoss = history.map(h => h.val_loss);
        const accuracy = history.map(h => h.accuracy);
        
        // 创建训练损失图表
        const trainingData = [{
            x: epochs,
            y: trainLoss,
            type: 'scatter',
            mode: 'lines+markers',
            name: '训练损失',
            line: { color: '#3b82f6' }
        }];
        
        if (valLoss.some(v => v > 0)) {
            trainingData.push({
                x: epochs,
                y: valLoss,
                type: 'scatter',
                mode: 'lines+markers',
                name: '验证损失',
                line: { color: '#ef4444' }
            });
        }
        
        const trainingLayout = {
            title: '损失曲线',
            xaxis: { title: 'Epoch' },
            yaxis: { title: 'Loss' },
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#f1f5f9' },
            margin: { t: 50, r: 50, b: 50, l: 50 }
        };
        
        Plotly.newPlot('training-chart', trainingData, trainingLayout, {responsive: true});
        
        // 创建准确率图表
        if (accuracy.some(v => v > 0)) {
            const accuracyData = [{
                x: epochs,
                y: accuracy.map(a => a * 100),
                type: 'scatter',
                mode: 'lines+markers',
                name: '准确率',
                line: { color: '#10b981' }
            }];
            
            const accuracyLayout = {
                title: '准确率曲线',
                xaxis: { title: 'Epoch' },
                yaxis: { title: 'Accuracy (%)' },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#f1f5f9' },
                margin: { t: 50, r: 50, b: 50, l: 50 }
            };
            
            Plotly.newPlot('validation-chart', accuracyData, accuracyLayout, {responsive: true});
        }
    }
    
    renderMetricsTable() {
        const history = this.experiments.history || [];
        
        if (history.length === 0) {
            document.getElementById('metrics-table').innerHTML = 
                '<div class="text-center text-gray-400 py-8">暂无指标数据</div>';
            return;
        }
        
        // 创建表格
        const table = document.createElement('table');
        table.className = 'w-full text-sm';
        
        // 表头
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        headerRow.className = 'border-b border-gray-600';
        
        ['Epoch', '训练损失', '验证损失', '准确率'].forEach(header => {
            const th = document.createElement('th');
            th.className = 'text-left py-2 px-4 text-gray-300';
            th.textContent = header;
            headerRow.appendChild(th);
        });
        
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        // 表体
        const tbody = document.createElement('tbody');
        
        history.forEach((row, index) => {
            const tr = document.createElement('tr');
            tr.className = index % 2 === 0 ? 'bg-slate-700' : '';
            
            [
                row.epoch,
                row.train_loss?.toFixed(4) || '-',
                row.val_loss?.toFixed(4) || '-',
                row.accuracy ? `${(row.accuracy * 100).toFixed(2)}%` : '-'
            ].forEach(value => {
                const td = document.createElement('td');
                td.className = 'py-2 px-4 text-gray-300';
                td.textContent = value;
                tr.appendChild(td);
            });
            
            tbody.appendChild(tr);
        });
        
        table.appendChild(tbody);
        
        document.getElementById('metrics-table').innerHTML = '';
        document.getElementById('metrics-table').appendChild(table);
    }
    
    renderMetricsTab() {
        // 使用ECharts创建更复杂的图表
        this.renderLossChart();
        this.renderAccuracyChart();
    }
    
    renderLossChart() {
        const history = this.experiment.history || [];
        
        if (history.length === 0) return;
        
        const chart = echarts.init(document.getElementById('loss-chart'));
        
        const option = {
            title: {
                text: '损失曲线',
                textStyle: { color: '#f1f5f9' }
            },
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(30, 41, 59, 0.9)',
                borderColor: '#475569',
                textStyle: { color: '#f1f5f9' }
            },
            legend: {
                data: ['训练损失', '验证损失'],
                textStyle: { color: '#cbd5e1' }
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '3%',
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: history.map(h => `Epoch ${h.epoch}`),
                axisLine: { lineStyle: { color: '#475569' } },
                axisLabel: { color: '#94a3b8' }
            },
            yAxis: {
                type: 'value',
                axisLine: { lineStyle: { color: '#475569' } },
                axisLabel: { color: '#94a3b8' },
                splitLine: { lineStyle: { color: '#334155' } }
            },
            series: [
                {
                    name: '训练损失',
                    type: 'line',
                    data: history.map(h => h.train_loss),
                    lineStyle: { color: '#3b82f6' },
                    itemStyle: { color: '#3b82f6' }
                },
                {
                    name: '验证损失',
                    type: 'line',
                    data: history.map(h => h.val_loss),
                    lineStyle: { color: '#ef4444' },
                    itemStyle: { color: '#ef4444' }
                }
            ]
        };
        
        chart.setOption(option);
        
        // 响应式
        window.addEventListener('resize', () => chart.resize());
    }
    
    renderAccuracyChart() {
        const history = this.experiment.history || [];
        
        if (history.length === 0) return;
        
        const chart = echarts.init(document.getElementById('accuracy-chart'));
        
        const option = {
            title: {
                text: '准确率曲线',
                textStyle: { color: '#f1f5f9' }
            },
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(30, 41, 59, 0.9)',
                borderColor: '#475569',
                textStyle: { color: '#f1f5f9' },
                formatter: function(params) {
                    return params[0].name + '<br/>' +
                           params[0].seriesName + ': ' + 
                           (params[0].value * 100).toFixed(2) + '%';
                }
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '3%',
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: history.map(h => `Epoch ${h.epoch}`),
                axisLine: { lineStyle: { color: '#475569' } },
                axisLabel: { color: '#94a3b8' }
            },
            yAxis: {
                type: 'value',
                axisLine: { lineStyle: { color: '#475569' } },
                axisLabel: { 
                    color: '#94a3b8',
                    formatter: function(value) {
                        return (value * 100).toFixed(0) + '%';
                    }
                },
                splitLine: { lineStyle: { color: '#334155' } }
            },
            series: [{
                name: '准确率',
                type: 'line',
                data: history.map(h => h.accuracy),
                lineStyle: { color: '#10b981' },
                itemStyle: { color: '#10b981' },
                smooth: true
            }]
        };
        
        chart.setOption(option);
        
        // 响应式
        window.addEventListener('resize', () => chart.resize());
    }
    
    renderConfigTab() {
        const config = this.experiment.config || {};
        
        const container = document.getElementById('config-content');
        
        if (Object.keys(config).length === 0) {
            container.innerHTML = '<div class="text-gray-400">暂无配置信息</div>';
            return;
        }
        
        container.innerHTML = `
            <pre class="text-sm text-gray-300 overflow-x-auto">${JSON.stringify(config, null, 2)}</pre>
        `;
    }
    
    async renderMediaTab() {
        try {
            const response = await fetch(`/api/experiments/${this.runId}/media`);
            const data = await response.json();
            
            const container = document.getElementById('media-content');
            
            if (!data.media || data.media.length === 0) {
                container.innerHTML = '<div class="text-gray-400">暂无媒体文件</div>';
                return;
            }
            
            container.innerHTML = data.media.map(file => {
                const fileUrl = `/api/experiments/${this.runId}/files/media/${file.path}`;
                
                if (file.type === 'image') {
                    return `
                        <div class="bg-slate-700 p-4 rounded-lg">
                            <img src="${fileUrl}" alt="${file.name}" class="w-full h-48 object-cover rounded">
                            <div class="mt-2 text-sm text-gray-300">${file.name}</div>
                        </div>
                    `;
                } else {
                    return `
                        <div class="bg-slate-700 p-4 rounded-lg flex items-center">
                            <i class="fas fa-file text-2xl text-blue-400 mr-4"></i>
                            <div>
                                <div class="text-white">${file.name}</div>
                                <div class="text-sm text-gray-400">${this.formatFileSize(file.size)}</div>
                            </div>
                        </div>
                    `;
                }
            }).join('');
            
        } catch (error) {
            console.error('Error loading media:', error);
            document.getElementById('media-content').innerHTML = 
                '<div class="text-red-400">加载媒体文件失败</div>';
        }
    }
    
    async renderArtifactsTab() {
        try {
            const response = await fetch(`/api/experiments/${this.runId}/artifacts`);
            const data = await response.json();
            
            const container = document.getElementById('artifacts-content');
            
            if (!data.artifacts || data.artifacts.length === 0) {
                container.innerHTML = '<div class="text-gray-400">暂无Artifacts</div>';
                return;
            }
            
            container.innerHTML = data.artifacts.map(artifact => `
                <div class="bg-slate-700 p-4 rounded-lg mb-4">
                    <div class="flex items-center mb-2">
                        <i class="fas fa-archive text-blue-400 mr-3"></i>
                        <span class="text-white font-semibold">${artifact.name}</span>
                    </div>
                    ${artifact.metadata.description ? `
                        <div class="text-sm text-gray-400 mb-2">${artifact.metadata.description}</div>
                    ` : ''}
                    <div class="text-xs text-gray-500">
                        创建时间: ${this.formatDate(artifact.metadata.created_at || new Date().toISOString())}
                    </div>
                </div>
            `).join('');
            
        } catch (error) {
            console.error('Error loading artifacts:', error);
            document.getElementById('artifacts-content').innerHTML = 
                '<div class="text-red-400">加载Artifacts失败</div>';
        }
    }
    
    renderLogsTab() {
        const container = document.getElementById('logs-content');
        
        // 生成模拟日志信息
        const logs = [
            `[${this.experiment.start_time}] 实验 ${this.experiment.name} 开始运行`,
            `[${this.experiment.start_time}] 项目: ${this.experiment.project}`,
            `[${this.experiment.start_time}] 运行ID: ${this.experiment.run_id}`,
            ...Object.entries(this.experiment.config || {}).map(([key, value]) => 
                `[${this.experiment.start_time}] 配置: ${key} = ${value}`
            ),
            `[${this.experiment.end_time || new Date().toISOString()}] 实验状态: ${this.experiment.status}`
        ];
        
        container.innerHTML = logs.map(log => `
            <div class="text-gray-300">${log}</div>
        `).join('');
    }
    
    addToComparison() {
        // 这里可以实现将实验添加到对比列表的功能
        console.log('Adding to comparison:', this.runId);
        // 可以存储到localStorage或通过其他方式传递
    }
    
    showLoading(show) {
        const indicator = document.getElementById('loading-indicator');
        if (show) {
            indicator.classList.remove('hidden');
        } else {
            indicator.classList.add('hidden');
        }
    }
    
    showError(message) {
        const container = document.getElementById('tab-content');
        container.innerHTML = `
            <div class="text-center py-12">
                <i class="fas fa-exclamation-triangle text-4xl text-red-400 mb-4"></i>
                <h3 class="text-xl font-semibold text-white mb-2">加载失败</h3>
                <p class="text-gray-400">${message}</p>
            </div>
        `;
    }
    
    // 工具函数
    calculateDuration(startTime, endTime) {
        const start = new Date(startTime);
        const end = endTime ? new Date(endTime) : new Date();
        const diff = end - start;
        
        const hours = Math.floor(diff / (1000 * 60 * 60));
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
        
        if (hours > 0) {
            return `${hours}h ${minutes}m`;
        } else {
            return `${minutes}m`;
        }
    }
    
    formatDate(dateString) {
        return new Date(dateString).toLocaleString('zh-CN');
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// 初始化实验详情管理器
document.addEventListener('DOMContentLoaded', () => {
    new ExperimentDetailManager();
});