// WandB Local Dashboard JavaScript
class DashboardManager {
    constructor() {
        this.experiments = [];
        this.selectedExperiments = [];
        this.comparisonMode = false;
        this.websocket = null;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setupWebSocket();
        this.loadExperiments();
        this.startAutoRefresh();
    }
    
    setupEventListeners() {
        // 刷新按钮
        document.getElementById('refresh-btn').addEventListener('click', () => {
            this.loadExperiments();
        });
        
        // 对比模式开关
        document.getElementById('comparison-toggle').addEventListener('click', () => {
            this.toggleComparisonMode();
        });
        
        // 搜索功能
        document.getElementById('search-input').addEventListener('input', (e) => {
            this.filterExperiments();
        });
        
        document.getElementById('search-btn').addEventListener('click', () => {
            this.filterExperiments();
        });
        
        // 筛选器
        document.getElementById('project-filter').addEventListener('change', () => {
            this.filterExperiments();
        });
        
        document.getElementById('time-filter').addEventListener('change', () => {
            this.filterExperiments();
        });
        
        // 状态筛选器
        document.querySelectorAll('.status-filter').forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.filterExperiments();
            });
        });
        
        // 模态框关闭
        document.getElementById('close-modal').addEventListener('click', () => {
            this.closeModal();
        });
        
        // 对比面板关闭
        document.getElementById('close-comparison').addEventListener('click', () => {
            this.closeComparisonPanel();
        });
        
        // 开始对比
        document.getElementById('start-comparison').addEventListener('click', () => {
            this.startComparison();
        });
        
        // 点击模态框背景关闭
        document.getElementById('experiment-modal').addEventListener('click', (e) => {
            if (e.target.id === 'experiment-modal') {
                this.closeModal();
            }
        });
    }
    
    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
            this.updateStatus('已连接', 'success');
        };
        
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'new_experiment') {
                this.handleNewExperiment(data.data);
            }
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateStatus('已断开', 'error');
            // 尝试重新连接
            setTimeout(() => this.setupWebSocket(), 5000);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('连接错误', 'error');
        };
    }
    
    updateStatus(text, type) {
        const statusIndicator = document.getElementById('status-indicator');
        const statusText = document.getElementById('status-text');
        
        statusIndicator.className = `status-indicator status-${type}`;
        statusText.textContent = text;
    }
    
    async loadExperiments() {
        this.showLoading(true);
        
        try {
            const response = await fetch('/api/experiments');
            const data = await response.json();
            
            this.experiments = data.experiments;
            this.updateStats();
            this.renderExperiments();
            this.updateProjectFilter();
            this.updateTagFilters();
            
        } catch (error) {
            console.error('Error loading experiments:', error);
            this.showEmptyState();
        } finally {
            this.showLoading(false);
        }
    }
    
    updateStats() {
        const total = this.experiments.length;
        const running = this.experiments.filter(exp => exp.status === 'running').length;
        const finished = this.experiments.filter(exp => exp.status === 'finished').length;
        const projects = new Set(this.experiments.map(exp => exp.project)).size;
        
        document.getElementById('total-experiments').textContent = total;
        document.getElementById('running-experiments').textContent = running;
        document.getElementById('finished-experiments').textContent = finished;
        document.getElementById('total-projects').textContent = projects;
    }
    
    renderExperiments() {
        const container = document.getElementById('experiments-grid');
        const filteredExperiments = this.getFilteredExperiments();
        
        if (filteredExperiments.length === 0) {
            this.showEmptyState();
            return;
        }
        
        container.innerHTML = filteredExperiments.map(exp => this.createExperimentCard(exp)).join('');
        
        // 添加事件监听器
        container.querySelectorAll('.experiment-card').forEach(card => {
            const runId = card.dataset.runId;
            
            // 点击查看详情
            card.addEventListener('click', (e) => {
                if (!e.target.closest('.comparison-checkbox')) {
                    this.showExperimentDetail(runId);
                }
            });
            
            // 对比选择
            const checkbox = card.querySelector('.comparison-checkbox');
            if (checkbox) {
                checkbox.addEventListener('change', (e) => {
                    this.toggleExperimentSelection(runId, e.target.checked);
                });
            }
        });
        
        this.hideEmptyState();
    }
    
    createExperimentCard(experiment) {
        const statusClass = `status-${experiment.status}`;
        const duration = this.calculateDuration(experiment.start_time, experiment.end_time);
        const metrics = experiment.metrics || {};
        
        return `
            <div class="experiment-card card-hover fade-in" data-run-id="${experiment.run_id}">
                <div class="flex justify-between items-start mb-4">
                    <div class="flex-1">
                        <h3 class="text-lg font-semibold text-white mb-1">${experiment.name}</h3>
                        <p class="text-sm text-gray-400 mb-2">${experiment.project}</p>
                        <div class="flex items-center text-sm text-gray-400">
                            <span class="status-indicator ${statusClass}"></span>
                            <span class="capitalize">${experiment.status}</span>
                            <span class="mx-2">•</span>
                            <span>${duration}</span>
                        </div>
                    </div>
                    ${this.comparisonMode ? `
                        <div class="comparison-checkbox-container">
                            <input type="checkbox" class="comparison-checkbox" 
                                   ${this.selectedExperiments.includes(experiment.run_id) ? 'checked' : ''}>
                        </div>
                    ` : ''}
                </div>
                
                ${experiment.tags && experiment.tags.length > 0 ? `
                    <div class="flex flex-wrap gap-1 mb-4">
                        ${experiment.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                    </div>
                ` : ''}
                
                ${Object.keys(metrics).length > 0 ? `
                    <div class="grid grid-cols-2 gap-4 mb-4">
                        ${Object.entries(metrics).slice(0, 4).map(([key, value]) => `
                            <div class="text-center">
                                <div class="text-lg font-bold text-white">${this.formatMetric(value)}</div>
                                <div class="text-xs text-gray-400">${this.formatMetricName(key)}</div>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
                
                <div class="flex justify-between items-center text-sm">
                    <span class="text-gray-400">${this.formatDate(experiment.start_time)}</span>
                    <div class="space-x-2">
                        <button class="text-blue-400 hover:text-blue-300" onclick="window.location.href='/experiment/${experiment.run_id}'">
                            <i class="fas fa-eye mr-1"></i>详情
                        </button>
                        <button class="text-green-400 hover:text-green-300" onclick="dashboard.addToComparison('${experiment.run_id}')">
                            <i class="fas fa-plus mr-1"></i>对比
                        </button>
                    </div>
                </div>
            </div>
        `;
    }
    
    getFilteredExperiments() {
        const projectFilter = document.getElementById('project-filter').value;
        const timeFilter = document.getElementById('time-filter').value;
        const searchTerm = document.getElementById('search-input').value.toLowerCase();
        
        const statusFilters = Array.from(document.querySelectorAll('.status-filter:checked'))
            .map(cb => cb.value);
        
        return this.experiments.filter(exp => {
            // 项目筛选
            if (projectFilter && exp.project !== projectFilter) return false;
            
            // 状态筛选
            if (statusFilters.length > 0 && !statusFilters.includes(exp.status)) return false;
            
            // 时间筛选
            if (timeFilter && !this.isWithinTimeRange(exp.start_time, timeFilter)) return false;
            
            // 搜索筛选
            if (searchTerm) {
                const searchableText = `${exp.name} ${exp.project} ${exp.tags?.join(' ') || ''}`.toLowerCase();
                if (!searchableText.includes(searchTerm)) return false;
            }
            
            return true;
        });
    }
    
    updateProjectFilter() {
        const select = document.getElementById('project-filter');
        const projects = [...new Set(this.experiments.map(exp => exp.project))].sort();
        
        select.innerHTML = '<option value="">所有项目</option>' +
            projects.map(project => `<option value="${project}">${project}</option>`).join('');
    }
    
    updateTagFilters() {
        const container = document.getElementById('tag-filters');
        const allTags = [...new Set(this.experiments.flatMap(exp => exp.tags || []))].sort();
        
        container.innerHTML = allTags.map(tag => `
            <span class="filter-tag" data-tag="${tag}">${tag}</span>
        `).join('');
        
        // 添加标签筛选事件
        container.querySelectorAll('.filter-tag').forEach(tag => {
            tag.addEventListener('click', () => {
                tag.classList.toggle('active');
                this.filterExperiments();
            });
        });
    }
    
    toggleComparisonMode() {
        this.comparisonMode = !this.comparisonMode;
        const button = document.getElementById('comparison-toggle');
        const panel = document.getElementById('comparison-panel');
        
        if (this.comparisonMode) {
            button.classList.add('comparison-mode');
            panel.classList.remove('hidden');
        } else {
            button.classList.remove('comparison-mode');
            panel.classList.add('hidden');
            this.selectedExperiments = [];
        }
        
        this.renderExperiments();
        this.updateComparisonPanel();
    }
    
    toggleExperimentSelection(runId, selected) {
        if (selected) {
            if (!this.selectedExperiments.includes(runId)) {
                this.selectedExperiments.push(runId);
            }
        } else {
            this.selectedExperiments = this.selectedExperiments.filter(id => id !== runId);
        }
        
        this.updateComparisonPanel();
    }
    
    updateComparisonPanel() {
        const content = document.getElementById('comparison-content');
        const button = document.getElementById('start-comparison');
        
        if (this.selectedExperiments.length === 0) {
            content.innerHTML = '<p class="text-gray-400 text-center">请选择要对比的实验</p>';
            button.disabled = true;
        } else {
            const selected = this.experiments.filter(exp => 
                this.selectedExperiments.includes(exp.run_id)
            );
            
            content.innerHTML = selected.map(exp => `
                <div class="comparison-selected p-3 rounded-lg">
                    <div class="font-semibold text-white">${exp.name}</div>
                    <div class="text-sm text-gray-400">${exp.project}</div>
                </div>
            `).join('');
            
            button.disabled = this.selectedExperiments.length < 2;
        }
    }
    
    addToComparison(runId) {
        if (!this.comparisonMode) {
            this.toggleComparisonMode();
        }
        
        if (!this.selectedExperiments.includes(runId)) {
            this.selectedExperiments.push(runId);
            this.updateComparisonPanel();
            this.renderExperiments(); // 重新渲染以更新复选框状态
        }
    }
    
    startComparison() {
        if (this.selectedExperiments.length >= 2) {
            // 跳转到对比页面
            const params = new URLSearchParams();
            this.selectedExperiments.forEach(id => params.append('runs', id));
            window.location.href = `/compare?${params.toString()}`;
        }
    }
    
    showExperimentDetail(runId) {
        const experiment = this.experiments.find(exp => exp.run_id === runId);
        if (!experiment) return;
        
        const modal = document.getElementById('experiment-modal');
        const title = document.getElementById('modal-title');
        const content = document.getElementById('modal-content');
        
        title.textContent = experiment.name;
        content.innerHTML = this.createExperimentDetailContent(experiment);
        
        modal.classList.remove('hidden');
    }
    
    createExperimentDetailContent(experiment) {
        const metrics = experiment.metrics || {};
        const config = experiment.config || {};
        
        return `
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                    <h4 class="text-lg font-semibold text-white mb-4">基本信息</h4>
                    <div class="space-y-3">
                        <div class="flex justify-between">
                            <span class="text-gray-400">运行ID:</span>
                            <span class="text-white">${experiment.run_id}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">项目:</span>
                            <span class="text-white">${experiment.project}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">状态:</span>
                            <span class="text-white capitalize">${experiment.status}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">开始时间:</span>
                            <span class="text-white">${this.formatDate(experiment.start_time)}</span>
                        </div>
                        ${experiment.end_time ? `
                            <div class="flex justify-between">
                                <span class="text-gray-400">结束时间:</span>
                                <span class="text-white">${this.formatDate(experiment.end_time)}</span>
                            </div>
                        ` : ''}
                    </div>
                </div>
                
                <div>
                    <h4 class="text-lg font-semibold text-white mb-4">关键指标</h4>
                    <div class="grid grid-cols-2 gap-4">
                        ${Object.entries(metrics).map(([key, value]) => `
                            <div class="bg-slate-700 p-3 rounded-lg">
                                <div class="text-sm text-gray-400">${this.formatMetricName(key)}</div>
                                <div class="text-lg font-bold text-white">${this.formatMetric(value)}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
            
            ${Object.keys(config).length > 0 ? `
                <div class="mt-6">
                    <h4 class="text-lg font-semibold text-white mb-4">配置参数</h4>
                    <div class="bg-slate-800 p-4 rounded-lg">
                        <pre class="text-sm text-gray-300 overflow-x-auto">${JSON.stringify(config, null, 2)}</pre>
                    </div>
                </div>
            ` : ''}
            
            <div class="mt-6 flex space-x-4">
                <button class="btn-primary" onclick="window.location.href='/experiment/${experiment.run_id}'">
                    <i class="fas fa-eye mr-2"></i>查看完整详情
                </button>
                <button class="btn-secondary" onclick="dashboard.addToComparison('${experiment.run_id}')">
                    <i class="fas fa-plus mr-2"></i>加入对比
                </button>
            </div>
        `;
    }
    
    closeModal() {
        document.getElementById('experiment-modal').classList.add('hidden');
    }
    
    closeComparisonPanel() {
        this.toggleComparisonMode();
    }
    
    filterExperiments() {
        this.renderExperiments();
    }
    
    showLoading(show) {
        const indicator = document.getElementById('loading-indicator');
        if (show) {
            indicator.classList.remove('hidden');
            document.getElementById('experiments-grid').classList.add('hidden');
        } else {
            indicator.classList.add('hidden');
            document.getElementById('experiments-grid').classList.remove('hidden');
        }
    }
    
    showEmptyState() {
        document.getElementById('empty-state').classList.remove('hidden');
        document.getElementById('experiments-grid').classList.add('hidden');
    }
    
    hideEmptyState() {
        document.getElementById('empty-state').classList.add('hidden');
        document.getElementById('experiments-grid').classList.remove('hidden');
    }
    
    startAutoRefresh() {
        // 每30秒自动刷新一次
        setInterval(() => {
            this.loadExperiments();
        }, 30000);
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
    
    isWithinTimeRange(dateString, timeFilter) {
        const date = new Date(dateString);
        const now = new Date();
        const diff = now - date;
        
        switch (timeFilter) {
            case '1h':
                return diff < 60 * 60 * 1000;
            case '24h':
                return diff < 24 * 60 * 60 * 1000;
            case '7d':
                return diff < 7 * 24 * 60 * 60 * 1000;
            case '30d':
                return diff < 30 * 24 * 60 * 60 * 1000;
            default:
                return true;
        }
    }
    
    handleNewExperiment(experiment) {
        // 检查是否已存在
        const existingIndex = this.experiments.findIndex(exp => exp.run_id === experiment.run_id);
        
        if (existingIndex >= 0) {
            this.experiments[existingIndex] = experiment;
        } else {
            this.experiments.unshift(experiment);
        }
        
        this.updateStats();
        this.renderExperiments();
    }
}

// 初始化仪表板
const dashboard = new DashboardManager();

// 全局函数
window.dashboard = dashboard;