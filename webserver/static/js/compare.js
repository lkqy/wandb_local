// 实验对比页面 JavaScript
class CompareManager {
    constructor() {
        this.experiments = [];
        this.selectedExperiments = [];
        this.availableExperiments = [];
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadProjects();
        this.checkUrlParams();
    }
    
    setupEventListeners() {
        // 加载实验按钮
        document.getElementById('load-experiments').addEventListener('click', () => {
            this.loadExperiments();
        });
        
        // 筛选器变化
        document.getElementById('project-filter').addEventListener('change', () => {
            this.filterExperiments();
        });
        
        document.getElementById('status-filter').addEventListener('change', () => {
            this.filterExperiments();
        });
        
        // 开始对比按钮
        document.getElementById('start-comparison').addEventListener('click', () => {
            this.startComparison();
        });
    }
    
    checkUrlParams() {
        // 检查URL参数中是否有预选的实验
        const urlParams = new URLSearchParams(window.location.search);
        const runIds = urlParams.getAll('runs');
        
        if (runIds.length > 0) {
            this.selectedExperiments = runIds;
            this.loadExperimentsForComparison(runIds);
        }
    }
    
    async loadProjects() {
        try {
            const response = await fetch('/api/projects');
            const data = await response.json();
            
            const select = document.getElementById('project-filter');
            select.innerHTML = '<option value="">所有项目</option>' +
                data.projects.map(project => 
                    `<option value="${project}">${project}</option>`
                ).join('');
                
        } catch (error) {
            console.error('Error loading projects:', error);
        }
    }
    
    async loadExperiments() {
        const projectFilter = document.getElementById('project-filter').value;
        const statusFilter = document.getElementById('status-filter').value;
        
        try {
            let url = '/api/experiments';
            const params = new URLSearchParams();
            
            if (projectFilter) params.append('project', projectFilter);
            if (statusFilter) params.append('status', statusFilter);
            
            if (params.toString()) {
                url += `?${params.toString()}`;
            }
            
            const response = await await fetch(url);
            const data = await response.json();
            
            this.availableExperiments = data.experiments;
            this.renderExperimentList();
            
        } catch (error) {
            console.error('Error loading experiments:', error);
            document.getElementById('experiment-list').innerHTML = 
                '<div class="text-red-400 text-center py-4">加载实验失败</div>';
        }
    }
    
    renderExperimentList() {
        const container = document.getElementById('experiment-list');
        
        if (this.availableExperiments.length === 0) {
            container.innerHTML = '<div class="text-gray-400 text-center py-4">暂无可用实验</div>';
            return;
        }
        
        container.innerHTML = this.availableExperiments.map(exp => `
            <div class="experiment-card ${this.selectedExperiments.includes(exp.run_id) ? 'selected' : ''}" 
                 data-run-id="${exp.run_id}">
                <div class="flex justify-between items-start mb-2">
                    <div class="flex-1">
                        <h4 class="font-semibold text-white">${exp.name}</h4>
                        <p class="text-sm text-gray-400">${exp.project}</p>
                    </div>
                    <input type="checkbox" class="experiment-checkbox" 
                           ${this.selectedExperiments.includes(exp.run_id) ? 'checked' : ''}>
                </div>
                <div class="text-sm text-gray-400 mb-2">
                    <span class="status-indicator status-${exp.status}"></span>
                    <span class="capitalize">${exp.status}</span>
                    <span class="mx-2">•</span>
                    <span>${this.formatDate(exp.start_time)}</span>
                </div>
                ${exp.metrics && Object.keys(exp.metrics).length > 0 ? `
                    <div class="grid grid-cols-2 gap-2 text-xs">
                        ${Object.entries(exp.metrics).slice(0, 2).map(([key, value]) => `
                            <div class="bg-slate-700 p-2 rounded">
                                <div class="text-gray-400">${this.formatMetricName(key)}</div>
                                <div class="text-white font-semibold">${this.formatMetric(value)}</div>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
            </div>
        `).join('');
        
        // 添加事件监听器
        container.querySelectorAll('.experiment-checkbox').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const runId = e.target.closest('.experiment-card').dataset.runId;
                this.toggleExperimentSelection(runId, e.target.checked);
            });
        });
        
        // 点击卡片也可以切换选择
        container.querySelectorAll('.experiment-card').forEach(card => {
            card.addEventListener('click', (e) => {
                if (e.target.type !== 'checkbox') {
                    const checkbox = card.querySelector('.experiment-checkbox');
                    checkbox.checked = !checkbox.checked;
                    checkbox.dispatchEvent(new Event('change'));
                }
            });
        });
    }
    
    toggleExperimentSelection(runId, selected) {
        if (selected) {
            if (!this.selectedExperiments.includes(runId)) {
                this.selectedExperiments.push(runId);
            }
        } else {
            this.selectedExperiments = this.selectedExperiments.filter(id => id !== runId);
        }
        
        this.updateSelectionUI();
        this.updateStartButton();
    }
    
    updateSelectionUI() {
        document.querySelectorAll('.experiment-card').forEach(card => {
            const runId = card.dataset.runId;
            const isSelected = this.selectedExperiments.includes(runId);
            card.classList.toggle('selected', isSelected);
        });
    }
    
    updateStartButton() {
        const button = document.getElementById('start-comparison');
        button.disabled = this.selectedExperiments.length < 2;
    }
    
    async startComparison() {
        if (this.selectedExperiments.length < 2) {
            alert('请至少选择两个实验进行对比');
            return;
        }
        
        try {
            // 加载选中的实验详情
            const experiments = [];
            for (const runId of this.selectedExperiments) {
                const response = await fetch(`/api/experiments/${runId}`);
                const experiment = await response.json();
                experiments.push(experiment);
            }
            
            this.experiments = experiments;
            this.renderComparison();
            
        } catch (error) {
            console.error('Error loading experiments for comparison:', error);
            alert('加载实验数据失败');
        }
    }
    
    async loadExperimentsForComparison(runIds) {
        try {
            const experiments = [];
            for (const runId of runIds) {
                const response = await fetch(`/api/experiments/${runId}`);
                const experiment = await response.json();
                experiments.push(experiment);
            }
            
            this.experiments = experiments;
            this.renderComparison();
            
        } catch (error) {
            console.error('Error loading experiments for comparison:', error);
        }
    }
    
    renderComparison() {
        // 隐藏空状态
        document.getElementById('empty-state').classList.add('hidden');
        
        // 显示对比结果
        document.getElementById('comparison-results').classList.remove('hidden');
        
        // 渲染图表
        this.renderLossComparisonChart();
        this.renderAccuracyComparisonChart();
        
        // 渲染表格
        this.renderComparisonTable();
        
        // 渲染配置对比
        this.renderConfigComparison();
    }
    
    renderLossComparisonChart() {
        const chart = echarts.init(document.getElementById('loss-comparison-chart'));
        
        const series = this.experiments.map((exp, index) => {
            const history = exp.history || [];
            const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'];
            
            return {
                name: exp.name,
                type: 'line',
                data: history.map(h => h.train_loss).filter(v => v > 0),
                lineStyle: { color: colors[index % colors.length] },
                itemStyle: { color: colors[index % colors.length] },
                smooth: true
            };
        }).filter(series => series.data.length > 0);
        
        const option = {
            title: {
                text: '训练损失对比',
                textStyle: { color: '#f1f5f9' }
            },
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(30, 41, 59, 0.9)',
                borderColor: '#475569',
                textStyle: { color: '#f1f5f9' }
            },
            legend: {
                data: this.experiments.map(exp => exp.name),
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
                data: Array.from({length: Math.max(...series.map(s => s.data.length))}, (_, i) => `Epoch ${i}`),
                axisLine: { lineStyle: { color: '#475569' } },
                axisLabel: { color: '#94a3b8' }
            },
            yAxis: {
                type: 'value',
                axisLine: { lineStyle: { color: '#475569' } },
                axisLabel: { color: '#94a3b8' },
                splitLine: { lineStyle: { color: '#334155' } }
            },
            series: series
        };
        
        chart.setOption(option);
        window.addEventListener('resize', () => chart.resize());
    }
    
    renderAccuracyComparisonChart() {
        const chart = echarts.init(document.getElementById('accuracy-comparison-chart'));
        
        const series = this.experiments.map((exp, index) => {
            const history = exp.history || [];
            const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'];
            
            return {
                name: exp.name,
                type: 'line',
                data: history.map(h => h.accuracy * 100).filter(v => v > 0),
                lineStyle: { color: colors[index % colors.length] },
                itemStyle: { color: colors[index % colors.length] },
                smooth: true
            };
        }).filter(series => series.data.length > 0);
        
        const option = {
            title: {
                text: '验证准确率对比',
                textStyle: { color: '#f1f5f9' }
            },
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(30, 41, 59, 0.9)',
                borderColor: '#475569',
                textStyle: { color: '#f1f5f9' },
                formatter: function(params) {
                    return params[0].name + '<br/>' +
                           params.map(p => p.seriesName + ': ' + p.value.toFixed(2) + '%').join('<br/>');
                }
            },
            legend: {
                data: this.experiments.map(exp => exp.name),
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
                data: Array.from({length: Math.max(...series.map(s => s.data.length))}, (_, i) => `Epoch ${i}`),
                axisLine: { lineStyle: { color: '#475569' } },
                axisLabel: { color: '#94a3b8' }
            },
            yAxis: {
                type: 'value',
                axisLine: { lineStyle: { color: '#475569' } },
                axisLabel: { 
                    color: '#94a3b8',
                    formatter: function(value) {
                        return value.toFixed(0) + '%';
                    }
                },
                splitLine: { lineStyle: { color: '#334155' } }
            },
            series: series
        };
        
        chart.setOption(option);
        window.addEventListener('resize', () => chart.resize());
    }
    
    renderComparisonTable() {
        const table = document.getElementById('comparison-table');
        
        // 收集所有指标
        const allMetrics = new Set();
        this.experiments.forEach(exp => {
            if (exp.metrics) {
                Object.keys(exp.metrics).forEach(metric => allMetrics.add(metric));
            }
        });
        
        const metrics = Array.from(allMetrics);
        
        // 创建表头
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        
        ['指标', ...this.experiments.map(exp => exp.name)].forEach((header, index) => {
            const th = document.createElement('th');
            th.textContent = header;
            if (index > 0) {
                th.className = 'text-center';
            }
            headerRow.appendChild(th);
        });
        
        thead.appendChild(headerRow);
        
        // 创建表体
        const tbody = document.createElement('tbody');
        
        // 基本信息行
        const basicInfo = [
            { name: '项目', key: 'project' },
            { name: '状态', key: 'status' },
            { name: '开始时间', key: 'start_time' },
            { name: '运行时长', key: 'duration' }
        ];
        
        basicInfo.forEach(info => {
            const row = document.createElement('tr');
            
            const nameCell = document.createElement('td');
            nameCell.textContent = info.name;
            nameCell.className = 'font-semibold';
            row.appendChild(nameCell);
            
            this.experiments.forEach(exp => {
                const cell = document.createElement('td');
                cell.className = 'text-center';
                
                if (info.key === 'duration') {
                    cell.textContent = this.calculateDuration(exp.start_time, exp.end_time);
                } else if (info.key === 'start_time') {
                    cell.textContent = this.formatDate(exp[info.key]);
                } else {
                    cell.textContent = exp[info.key] || '-';
                }
                
                row.appendChild(cell);
            });
            
            tbody.appendChild(row);
        });
        
        // 指标行
        metrics.forEach(metric => {
            const row = document.createElement('tr');
            
            const nameCell = document.createElement('td');
            nameCell.textContent = this.formatMetricName(metric);
            nameCell.className = 'font-semibold';
            row.appendChild(nameCell);
            
            // 收集该指标的所有值以确定最佳值
            const values = this.experiments.map(exp => 
                exp.metrics && exp.metrics[metric]
            ).filter(v => v !== undefined);
            
            const bestValue = metric.includes('accuracy') || metric.includes('score') 
                ? Math.max(...values) : Math.min(...values);
            
            this.experiments.forEach(exp => {
                const cell = document.createElement('td');
                cell.className = 'text-center metric-cell';
                
                const value = exp.metrics && exp.metrics[metric];
                if (value !== undefined) {
                    cell.textContent = this.formatMetric(value);
                    
                    // 标记最佳和最差值
                    if (value === bestValue) {
                        cell.classList.add('best');
                    } else if (values.length > 1) {
                        const worstValue = metric.includes('accuracy') || metric.includes('score')
                            ? Math.min(...values) : Math.max(...values);
                        if (value === worstValue) {
                            cell.classList.add('worst');
                        }
                    }
                } else {
                    cell.textContent = '-';
                }
                
                row.appendChild(cell);
            });
            
            tbody.appendChild(row);
        });
        
        table.innerHTML = '';
        table.appendChild(thead);
        table.appendChild(tbody);
    }
    
    renderConfigComparison() {
        const container = document.getElementById('config-comparison');
        
        // 收集所有配置参数
        const allConfigs = new Set();
        this.experiments.forEach(exp => {
            if (exp.config) {
                Object.keys(exp.config).forEach(key => allConfigs.add(key));
            }
        });
        
        const configs = Array.from(allConfigs);
        
        if (configs.length === 0) {
            container.innerHTML = '<div class="text-gray-400">暂无配置信息</div>';
            return;
        }
        
        // 创建配置对比表格
        const table = document.createElement('table');
        table.className = 'comparison-table';
        
        // 表头
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        
        ['配置参数', ...this.experiments.map(exp => exp.name)].forEach((header, index) => {
            const th = document.createElement('th');
            th.textContent = header;
            if (index > 0) {
                th.className = 'text-center';
            }
            headerRow.appendChild(th);
        });
        
        thead.appendChild(headerRow);
        
        // 表体
        const tbody = document.createElement('tbody');
        
        configs.forEach(config => {
            const row = document.createElement('tr');
            
            const nameCell = document.createElement('td');
            nameCell.textContent = this.formatMetricName(config);
            nameCell.className = 'font-semibold';
            row.appendChild(nameCell);
            
            // 收集该配置的所有值
            const values = this.experiments.map(exp => 
                exp.config && exp.config[config]
            ).filter(v => v !== undefined);
            
            // 检查是否有不同值
            const hasDifferentValues = new Set(values).size > 1;
            
            this.experiments.forEach(exp => {
                const cell = document.createElement('td');
                cell.className = 'text-center';
                
                const value = exp.config && exp.config[config];
                if (value !== undefined) {
                    cell.textContent = typeof value === 'object' ? JSON.stringify(value) : value;
                    
                    // 如果有不同值，高亮显示
                    if (hasDifferentValues) {
                        cell.style.backgroundColor = 'rgba(245, 158, 11, 0.1)';
                        cell.style.color = '#f59e0b';
                    }
                } else {
                    cell.textContent = '-';
                }
                
                row.appendChild(cell);
            });
            
            tbody.appendChild(row);
        });
        
        table.appendChild(thead);
        table.appendChild(tbody);
        
        container.innerHTML = '';
        container.appendChild(table);
    }
    
    // 工具函数
    formatDate(dateString) {
        return new Date(dateString).toLocaleDateString('zh-CN');
    }
    
    formatMetric(value) {
        if (typeof value === 'number') {
            if (value < 1) {
                return value.toFixed(4);
            } else if (value < 100) {
                return value.toFixed(2);
            } else {
                return value.toFixed(0);
            }
        }
        return value;
    }
    
    formatMetricName(name) {
        return name.replace(/[_-]/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
    
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
}

// 初始化对比管理器
document.addEventListener('DOMContentLoaded', () => {
    new CompareManager();
});