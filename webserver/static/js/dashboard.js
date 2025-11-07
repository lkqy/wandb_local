// Vue.js 组件定义
const ExperimentCard = {
    props: ['experiment', 'comparisonMode', 'selected'],
    template: `
        <div :class="['experiment-card card-hover fade-in', { 'selected': selected, 'comparison-mode': comparisonMode }]" 
             @click="handleClick">
            <div class="flex justify-between items-start mb-4">
                <div class="flex-1">
                    <h3 class="text-lg font-semibold text-gray-900 mb-1">{{ experiment.name }}</h3>
                    <p class="text-sm text-gray-600">项目: {{ experiment.project }}</p>
                    <p class="text-xs text-gray-500 mt-1">ID: {{ experiment.run_id }}</p>
                </div>
                <div class="flex items-center space-x-2">
                    <input v-if="comparisonMode" 
                           type="checkbox" 
                           :checked="selected"
                           @click.stop
                           @change="$emit('toggle-selection', experiment.run_id)"
                           class="w-4 h-4 text-blue-600">
                    <div class="flex items-center text-sm text-gray-600">
                        <span :class="['status-indicator', 'status-' + experiment.status]"></span>
                        {{ getStatusText(experiment.status) }}
                    </div>
                </div>
            </div>
            
            <div class="grid grid-cols-2 gap-4 mb-4">
                <div v-for="metric in displayMetrics" :key="metric.key">
                    <div class="text-2xl font-bold text-gray-900">
                        {{ formatMetricValue(experiment.summary?.[metric.key]) }}
                    </div>
                    <div class="text-sm text-gray-600">{{ metric.label }}</div>
                    <div v-if="metric.trend" class="metric-trend">
                        <i :class="['fas', 'fa-arrow-' + metric.trend, 'text-xs', 'trend-' + metric.trend]"></i>
                    </div>
                </div>
            </div>
            
            <div v-if="experiment.tags && experiment.tags.length > 0" class="mb-4">
                <div class="flex flex-wrap gap-2">
                    <span v-for="tag in experiment.tags.slice(0, 3)" :key="tag" class="filter-tag">
                        {{ tag }}
                    </span>
                    <span v-if="experiment.tags.length > 3" class="text-xs text-gray-500">
                        +{{ experiment.tags.length - 3 }}
                    </span>
                </div>
            </div>
            
            <div class="flex justify-between items-center text-sm text-gray-600">
                <span>开始时间: {{ formatDate(experiment.start_time) }}</span>
                <span>{{ formatDuration(experiment.duration) }}</span>
            </div>
            
            <div class="mt-4 pt-4 border-t border-gray-200">
                <div class="flex space-x-2">
                    <button class="btn-primary text-sm px-3 py-1">
                        <i class="fas fa-chart-line mr-1"></i>查看详情
                    </button>
                    <button @click.stop="refreshExperiment(experiment.run_id)" class="btn-secondary text-sm px-3 py-1">
                        <i class="fas fa-sync-alt mr-1"></i>刷新
                    </button>
                </div>
            </div>
        </div>
    `,
    computed: {
        displayMetrics() {
            const metrics = [];
            const summary = this.experiment.summary || {};
            
            // 智能选择要显示的指标
            if (summary.accuracy !== undefined) {
                metrics.push({ key: 'accuracy', label: '准确率' });
            }
            if (summary.loss !== undefined) {
                metrics.push({ key: 'loss', label: '损失' });
            }
            if (summary.val_accuracy !== undefined) {
                metrics.push({ key: 'val_accuracy', label: '验证准确率' });
            }
            if (summary.val_loss !== undefined) {
                metrics.push({ key: 'val_loss', label: '验证损失' });
            }
            
            // 如果没有标准指标，显示前两个数值型指标
            if (metrics.length === 0) {
                const numericKeys = Object.keys(summary).filter(key => 
                    typeof summary[key] === 'number' && !key.startsWith('_')
                );
                
                numericKeys.slice(0, 2).forEach(key => {
                    metrics.push({ key, label: this.formatLabel(key) });
                });
            }
            
            return metrics.slice(0, 2);
        }
    },
    methods: {
        handleClick() {
            if (this.comparisonMode) {
                this.$emit('toggle-selection', this.experiment.run_id);
            } else {
                this.$emit('click', this.experiment);
            }
        },
        refreshExperiment(runId) {
            this.$emit('refresh', runId);
        },
        getStatusText(status) {
            const statusMap = {
                'running': '运行中',
                'finished': '已完成',
                'failed': '失败',
                'unknown': '未知'
            };
            return statusMap[status] || status;
        },
        
        formatMetricValue(value) {
            if (typeof value === 'number') {
                if (value < 0.01) {
                    return value.toExponential(2);
                } else if (value < 1) {
                    return value.toFixed(4);
                } else if (value < 100) {
                    return value.toFixed(2);
                } else {
                    return value.toFixed(0);
                }
            }
            return value || 'N/A';
        },
        
        formatDuration(seconds) {
            if (!seconds) return 'N/A';
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            if (hours > 0) {
                return `${hours}小时${minutes}分钟`;
            } else {
                return `${minutes}分钟`;
            }
        },
        formatMetricValue(value) {
            if (typeof value === 'number') {
                if (value < 0.01) {
                    return value.toExponential(2);
                } else if (value < 1) {
                    return value.toFixed(4);
                } else if (value < 100) {
                    return value.toFixed(2);
                } else {
                    return value.toFixed(0);
                }
            }
            return value || 'N/A';
        },
        formatLabel(key) {
            return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        },
        formatDate(dateString) {
            if (!dateString) return 'N/A';
            const date = new Date(dateString);
            return date.toLocaleDateString('zh-CN') + ' ' + date.toLocaleTimeString('zh-CN');
        },
        formatDuration(seconds) {
            if (!seconds) return 'N/A';
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            if (hours > 0) {
                return `${hours}小时${minutes}分钟`;
            } else {
                return `${minutes}分钟`;
            }
        }
    }
};

const ExperimentListItem = {
    props: ['experiment', 'comparisonMode', 'selected'],
    template: `
        <div :class="['experiment-card card-hover fade-in', { 'selected': selected, 'comparison-mode': comparisonMode }]" 
             @click="handleClick">
            <div class="flex items-center space-x-4">
                <input v-if="comparisonMode" 
                       type="checkbox" 
                       :checked="selected"
                       @click.stop
                       @change="$emit('toggle-selection', experiment.run_id)"
                       class="w-4 h-4 text-blue-600">
                
                <div class="flex-1">
                    <div class="flex items-center space-x-3">
                        <h3 class="text-lg font-semibold text-gray-900">{{ experiment.name }}</h3>
                        <span :class="['status-indicator', 'status-' + experiment.status]"></span>
                        <span class="text-sm text-gray-600">{{ getStatusText(experiment.status) }}</span>
                    </div>
                    <div class="flex items-center space-x-4 mt-1 text-sm text-gray-600">
                        <span>项目: {{ experiment.project }}</span>
                        <span>开始时间: {{ formatDate(experiment.start_time) }}</span>
                        <span>{{ formatDuration(experiment.duration) }}</span>
                    </div>
                </div>
                
                <div class="flex items-center space-x-4">
                    <div v-for="metric in displayMetrics" :key="metric.key" class="text-center">
                        <div class="text-lg font-bold text-gray-900">
                            {{ formatMetricValue(experiment.summary?.[metric.key]) }}
                        </div>
                        <div class="text-xs text-gray-600">{{ metric.label }}</div>
                    </div>
                    
                    <div v-if="experiment.tags && experiment.tags.length > 0" class="flex space-x-1">
                        <span v-for="tag in experiment.tags.slice(0, 3)" :key="tag" class="filter-tag text-xs">
                            {{ tag }}
                        </span>
                    </div>
                    
                    <button @click.stop="$emit('click', experiment)" class="btn-primary text-sm px-3 py-1">
                        <i class="fas fa-chart-line mr-1"></i>查看详情
                    </button>
                </div>
            </div>
        </div>
    `,
    computed: {
        displayMetrics() {
            const metrics = [];
            const summary = this.experiment.summary || {};
            
            // 智能选择要显示的指标
            const priorityKeys = ['accuracy', 'loss', 'val_accuracy', 'val_loss'];
            
            for (const key of priorityKeys) {
                if (summary[key] !== undefined) {
                    metrics.push({ 
                        key, 
                        label: this.formatLabel(key) 
                    });
                    if (metrics.length >= 2) break;
                }
            }
            
            // 如果没有标准指标，显示前两个数值型指标
            if (metrics.length === 0) {
                const numericKeys = Object.keys(summary).filter(key => 
                    typeof summary[key] === 'number' && !key.startsWith('_')
                );
                
                numericKeys.slice(0, 2).forEach(key => {
                    metrics.push({ key, label: this.formatLabel(key) });
                });
            }
            
            return metrics;
        }
    },
    methods: {
        handleClick() {
            if (this.comparisonMode) {
                this.$emit('toggle-selection', this.experiment.run_id);
            } else {
                this.$emit('click', this.experiment);
            }
        },
        getStatusText(status) {
            const statusMap = {
                'running': '运行中',
                'finished': '已完成',
                'failed': '失败',
                'unknown': '未知'
            };
            return statusMap[status] || status;
        },
        
        formatMetricValue(value) {
            if (typeof value === 'number') {
                if (value < 0.01) {
                    return value.toExponential(2);
                } else if (value < 1) {
                    return value.toFixed(4);
                } else if (value < 100) {
                    return value.toFixed(2);
                } else {
                    return value.toFixed(0);
                }
            }
            return value || 'N/A';
        },
        
        formatDuration(seconds) {
            if (!seconds) return 'N/A';
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            if (hours > 0) {
                return `${hours}小时${minutes}分钟`;
            } else {
                return `${minutes}分钟`;
            }
        },
        formatMetricValue(value) {
            if (typeof value === 'number') {
                if (value < 0.01) {
                    return value.toExponential(2);
                } else if (value < 1) {
                    return value.toFixed(4);
                } else if (value < 100) {
                    return value.toFixed(2);
                } else {
                    return value.toFixed(0);
                }
            }
            return value || 'N/A';
        },
        formatLabel(key) {
            return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        },
        formatDate(dateString) {
            if (!dateString) return 'N/A';
            const date = new Date(dateString);
            return date.toLocaleDateString('zh-CN');
        },
        formatDuration(seconds) {
            if (!seconds) return 'N/A';
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            if (hours > 0) {
                return `${hours}h ${minutes}m`;
            } else {
                return `${minutes}m`;
            }
        }
    }
};

const ComparisonExperimentCard = {
    props: ['experiment'],
    template: `
        <div class="bg-white p-4 rounded-lg border border-gray-200 relative">
            <button @click="$emit('remove', experiment.run_id)" 
                    class="absolute top-2 right-2 text-gray-400 hover:text-gray-600">
                <i class="fas fa-times"></i>
            </button>
            
            <h4 class="font-semibold text-gray-900 mb-2">{{ experiment.name }}</h4>
            <div class="space-y-2 text-sm">
                <div class="flex justify-between">
                    <span class="text-gray-600">项目:</span>
                    <span>{{ experiment.project }}</span>
                </div>
                <div class="flex justify-between">
                    <span class="text-gray-600">状态:</span>
                    <span>
                        <span :class="['status-indicator', 'status-' + experiment.status]"></span>
                        {{ getStatusText(experiment.status) }}
                    </span>
                </div>
                <div class="flex justify-between">
                    <span class="text-gray-600">准确率:</span>
                    <span>{{ formatMetricValue(experiment.summary?.accuracy) }}</span>
                </div>
                <div class="flex justify-between">
                    <span class="text-gray-600">损失:</span>
                    <span>{{ formatMetricValue(experiment.summary?.loss) }}</span>
                </div>
                <div class="flex justify-between">
                    <span class="text-gray-600">时长:</span>
                    <span>{{ formatDuration(experiment.duration) }}</span>
                </div>
            </div>
        </div>
    `,
    methods: {
        getStatusText(status) {
            const statusMap = {
                'running': '运行中',
                'finished': '已完成',
                'failed': '失败',
                'unknown': '未知'
            };
            return statusMap[status] || status;
        },
        
        formatMetricValue(value) {
            if (typeof value === 'number') {
                if (value < 0.01) {
                    return value.toExponential(2);
                } else if (value < 1) {
                    return value.toFixed(4);
                } else if (value < 100) {
                    return value.toFixed(2);
                } else {
                    return value.toFixed(0);
                }
            }
            return value || 'N/A';
        },
        
        formatDuration(seconds) {
            if (!seconds) return 'N/A';
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            if (hours > 0) {
                return `${hours}小时${minutes}分钟`;
            } else {
                return `${minutes}分钟`;
            }
        },
        formatMetricValue(value) {
            if (typeof value === 'number') {
                if (value < 0.01) {
                    return value.toExponential(2);
                } else if (value < 1) {
                    return value.toFixed(4);
                } else if (value < 100) {
                    return value.toFixed(2);
                } else {
                    return value.toFixed(0);
                }
            }
            return value || 'N/A';
        },
        formatDuration(seconds) {
            if (!seconds) return 'N/A';
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            if (hours > 0) {
                return `${hours}h ${minutes}m`;
            } else {
                return `${minutes}m`;
            }
        }
    }
};

const ExperimentDetailTabs = {
    props: ['experiment'],
    data() {
        return {
            activeTab: 'overview'
        };
    },
    template: `
        <div class="p-6">
            <!-- 标签页导航 -->
            <div class="border-b border-gray-200 mb-6">
                <nav class="flex space-x-8">
                    <button @click="activeTab = 'overview'" 
                            :class="['py-2 px-1 border-b-2 font-medium text-sm', 
                                     activeTab === 'overview' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700']">
                        概览
                    </button>
                    <button @click="activeTab = 'metrics'" 
                            :class="['py-2 px-1 border-b-2 font-medium text-sm', 
                                     activeTab === 'metrics' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700']">
                        指标
                    </button>
                    <button @click="activeTab = 'config'" 
                            :class="['py-2 px-1 border-b-2 font-medium text-sm', 
                                     activeTab === 'config' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700']">
                        配置
                    </button>
                    <button @click="activeTab = 'media'" 
                            :class="['py-2 px-1 border-b-2 font-medium text-sm', 
                                     activeTab === 'media' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700']">
                        媒体文件
                    </button>
                    <button @click="activeTab = 'artifacts'" 
                            :class="['py-2 px-1 border-b-2 font-medium text-sm', 
                                     activeTab === 'artifacts' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700']">
                        Artifacts
                    </button>
                </nav>
            </div>
            
            <!-- 标签页内容 -->
            <div>
                <!-- 概览标签页 -->
                <div v-if="activeTab === 'overview'" class="space-y-6">
                    <experiment-overview :experiment="experiment"></experiment-overview>
                </div>
                
                <!-- 指标标签页 -->
                <div v-if="activeTab === 'metrics'" class="space-y-6">
                    <experiment-metrics :experiment="experiment"></experiment-metrics>
                </div>
                
                <!-- 配置标签页 -->
                <div v-if="activeTab === 'config'" class="space-y-6">
                    <experiment-config :experiment="experiment"></experiment-config>
                </div>
                
                <!-- 媒体文件标签页 -->
                <div v-if="activeTab === 'media'" class="space-y-6">
                    <experiment-media :experiment="experiment"></experiment-media>
                </div>
                
                <!-- Artifacts标签页 -->
                <div v-if="activeTab === 'artifacts'" class="space-y-6">
                    <experiment-artifacts :experiment="experiment"></experiment-artifacts>
                </div>
            </div>
        </div>
    `
};

const ExperimentOverview = {
    props: ['experiment'],
    template: `
        <div class="space-y-6">
            <!-- 基本信息 -->
            <div class="bg-gray-50 p-6 rounded-lg">
                <h3 class="text-lg font-semibold mb-4">基本信息</h3>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <span class="text-sm text-gray-600">实验ID:</span>
                        <span class="font-mono text-sm ml-2">{{ experiment.run_id }}</span>
                    </div>
                    <div>
                        <span class="text-sm text-gray-600">项目:</span>
                        <span class="ml-2">{{ experiment.project }}</span>
                    </div>
                    <div>
                        <span class="text-sm text-gray-600">状态:</span>
                        <span class="ml-2">
                            <span :class="['status-indicator', 'status-' + experiment.status]"></span>
                            {{ getStatusText(experiment.status) }}
                        </span>
                    </div>
                    <div>
                        <span class="text-sm text-gray-600">开始时间:</span>
                        <span class="ml-2">{{ formatDate(experiment.start_time) }}</span>
                    </div>
                    <div>
                        <span class="text-sm text-gray-600">结束时间:</span>
                        <span class="ml-2">{{ experiment.end_time ? formatDate(experiment.end_time) : '运行中' }}</span>
                    </div>
                    <div>
                        <span class="text-sm text-gray-600">持续时间:</span>
                        <span class="ml-2">{{ formatDuration(experiment.duration) }}</span>
                    </div>
                </div>
            </div>
            
            <!-- 实验摘要 -->
            <div v-if="experiment.summary && Object.keys(experiment.summary).length > 0">
                <h3 class="text-lg font-semibold mb-4">实验摘要</h3>
                <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                    <div v-for="(value, key) in experiment.summary" :key="key" class="bg-gray-50 p-4 rounded-lg">
                        <div class="text-sm text-gray-600">{{ formatLabel(key) }}</div>
                        <div class="text-lg font-semibold">{{ formatMetricValue(value) }}</div>
                    </div>
                </div>
            </div>
            
            <!-- 标签和注释 -->
            <div class="space-y-4">
                <div v-if="experiment.tags && experiment.tags.length > 0">
                    <h3 class="text-lg font-semibold mb-2">标签</h3>
                    <div class="flex flex-wrap gap-2">
                        <span v-for="tag in experiment.tags" :key="tag" class="filter-tag">
                            {{ tag }}
                        </span>
                    </div>
                </div>
                
                <div v-if="experiment.notes">
                    <h3 class="text-lg font-semibold mb-2">注释</h3>
                    <div class="bg-gray-50 p-4 rounded-lg">
                        <p class="text-gray-700">{{ experiment.notes }}</p>
                    </div>
                </div>
            </div>
        </div>
    `,
    methods: {
        getStatusText(status) {
            const statusMap = {
                'running': '运行中',
                'finished': '已完成',
                'failed': '失败',
                'unknown': '未知'
            };
            return statusMap[status] || status;
        },
        
        formatMetricValue(value) {
            if (typeof value === 'number') {
                if (value < 0.01) {
                    return value.toExponential(2);
                } else if (value < 1) {
                    return value.toFixed(4);
                } else if (value < 100) {
                    return value.toFixed(2);
                } else {
                    return value.toFixed(0);
                }
            }
            return value || 'N/A';
        },
        
        formatDuration(seconds) {
            if (!seconds) return 'N/A';
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            if (hours > 0) {
                return `${hours}小时${minutes}分钟`;
            } else {
                return `${minutes}分钟`;
            }
        },
        formatLabel(key) {
            return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        },
        formatMetricValue(value) {
            if (typeof value === 'number') {
                if (value < 0.01) {
                    return value.toExponential(2);
                } else if (value < 1) {
                    return value.toFixed(4);
                } else if (value < 100) {
                    return value.toFixed(2);
                } else {
                    return value.toFixed(0);
                }
            }
            return value || 'N/A';
        },
        formatDate(dateString) {
            if (!dateString) return 'N/A';
            const date = new Date(dateString);
            return date.toLocaleDateString('zh-CN') + ' ' + date.toLocaleTimeString('zh-CN');
        },
        formatDuration(seconds) {
            if (!seconds) return 'N/A';
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            if (hours > 0) {
                return `${hours}小时${minutes}分钟`;
            } else {
                return `${minutes}分钟`;
            }
        }
    }
};

const ExperimentMetrics = {
    props: ['experiment'],
    data() {
        return {
            selectedMetrics: [],
            chartType: 'line'
        };
    },
    template: `
        <div class="space-y-6">
            <!-- 指标选择器 -->
            <div class="bg-gray-50 p-4 rounded-lg">
                <h3 class="text-lg font-semibold mb-3">指标选择</h3>
                <div class="flex flex-wrap gap-2 mb-4">
                    <label v-for="metric in availableMetrics" :key="metric" class="flex items-center">
                        <input type="checkbox" v-model="selectedMetrics" :value="metric" class="mr-2">
                        <span class="text-sm">{{ formatLabel(metric) }}</span>
                    </label>
                </div>
                
                <div class="flex items-center space-x-4">
                    <span class="text-sm font-medium">图表类型:</span>
                    <label class="flex items-center">
                        <input type="radio" v-model="chartType" value="line" class="mr-2">
                        <span class="text-sm">折线图</span>
                    </label>
                    <label class="flex items-center">
                        <input type="radio" v-model="chartType" value="scatter" class="mr-2">
                        <span class="text-sm">散点图</span>
                    </label>
                </div>
            </div>
            
            <!-- 图表容器 -->
            <div v-if="selectedMetrics.length > 0" class="chart-container">
                <div ref="chartContainer" style="height: 400px;"></div>
            </div>
            
            <!-- 数据表格 -->
            <div v-if="experiment.history && experiment.history.length > 0" class="table-container">
                <h3 class="text-lg font-semibold mb-3">历史数据</h3>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>步骤</th>
                            <th>时间</th>
                            <th v-for="metric in selectedMetrics" :key="metric">
                                {{ formatLabel(metric) }}
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="entry in paginatedHistory" :key="entry.step">
                            <td>{{ entry.step }}</td>
                            <td>{{ formatDate(entry.timestamp) }}</td>
                            <td v-for="metric in selectedMetrics" :key="metric">
                                {{ formatMetricValue(entry.data?.[metric]) }}
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    `,
    computed: {
        availableMetrics() {
            if (!this.experiment.history || this.experiment.history.length === 0) {
                return [];
            }
            
            const metrics = new Set();
            this.experiment.history.forEach(entry => {
                if (entry.data) {
                    Object.keys(entry.data).forEach(key => {
                        if (typeof entry.data[key] === 'number') {
                            metrics.add(key);
                        }
                    });
                }
            });
            
            return Array.from(metrics);
        },
        paginatedHistory() {
            // 显示最近的100条记录
            return this.experiment.history.slice(-100);
        }
    },
    watch: {
        selectedMetrics() {
            this.$nextTick(() => {
                this.renderChart();
            });
        },
        chartType() {
            this.$nextTick(() => {
                this.renderChart();
            });
        }
    },
    mounted() {
        // 默认选择前两个指标
        this.selectedMetrics = this.availableMetrics.slice(0, 2);
    },
    methods: {
        renderChart() {
            if (!this.$refs.chartContainer || this.selectedMetrics.length === 0) {
                return;
            }
            
            const chart = echarts.init(this.$refs.chartContainer);
            
            const series = this.selectedMetrics.map(metric => {
                const data = this.experiment.history
                    .filter(entry => entry.data && entry.data[metric] !== undefined)
                    .map(entry => [entry.step, entry.data[metric]]);
                
                return {
                    name: this.formatLabel(metric),
                    type: this.chartType,
                    data: data,
                    smooth: this.chartType === 'line',
                    symbol: this.chartType === 'scatter' ? 'circle' : 'none',
                    symbolSize: 4
                };
            });
            
            const option = {
                title: {
                    text: '实验指标趋势',
                    left: 'center'
                },
                tooltip: {
                    trigger: 'axis',
                    axisPointer: {
                        type: 'cross'
                    }
                },
                legend: {
                    data: this.selectedMetrics.map(this.formatLabel),
                    bottom: 10
                },
                grid: {
                    left: '3%',
                    right: '4%',
                    bottom: '15%',
                    containLabel: true
                },
                xAxis: {
                    type: 'value',
                    name: 'Step'
                },
                yAxis: {
                    type: 'value'
                },
                series: series
            };
            
            chart.setOption(option);
            
            // 响应式
            window.addEventListener('resize', () => {
                chart.resize();
            });
        },
        formatLabel(key) {
            return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        },
        formatMetricValue(value) {
            if (typeof value === 'number') {
                if (value < 0.01) {
                    return value.toExponential(2);
                } else if (value < 1) {
                    return value.toFixed(4);
                } else if (value < 100) {
                    return value.toFixed(2);
                } else {
                    return value.toFixed(0);
                }
            }
            return value || 'N/A';
        },
        formatDate(dateString) {
            if (!dateString) return 'N/A';
            const date = new Date(dateString);
            return date.toLocaleDateString('zh-CN') + ' ' + date.toLocaleTimeString('zh-CN');
        }
    }
};

const ExperimentConfig = {
    props: ['experiment'],
    template: `
        <div class="space-y-6">
            <!-- 实验配置 -->
            <div v-if="experiment.config && Object.keys(experiment.config).length > 0">
                <h3 class="text-lg font-semibold mb-3">实验配置</h3>
                <div class="bg-gray-50 p-4 rounded-lg">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>参数名</th>
                                <th>参数值</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="(value, key) in experiment.config" :key="key">
                                <td class="font-medium">{{ formatLabel(key) }}</td>
                                <td>{{ formatConfigValue(value) }}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- 标签管理 -->
            <div>
                <h3 class="text-lg font-semibold mb-3">标签管理</h3>
                <div class="bg-gray-50 p-4 rounded-lg">
                    <div class="flex flex-wrap gap-2 mb-3">
                        <span v-for="tag in experiment.tags" :key="tag" class="filter-tag">
                            {{ tag }}
                            <button @click="removeTag(tag)" class="ml-2 text-gray-500 hover:text-red-500">
                                <i class="fas fa-times text-xs"></i>
                            </button>
                        </span>
                    </div>
                    <div class="flex space-x-2">
                        <input v-model="newTag" type="text" placeholder="输入新标签" class="flex-1 search-input">
                        <button @click="addTag" class="btn-primary">添加标签</button>
                    </div>
                </div>
            </div>
            
            <!-- 注释管理 -->
            <div>
                <h3 class="text-lg font-semibold mb-3">注释管理</h3>
                <div class="bg-gray-50 p-4 rounded-lg">
                    <textarea v-model="experiment.notes" 
                              rows="4" 
                              placeholder="输入实验注释..."
                              class="w-full search-input mb-3"></textarea>
                    <button @click="updateNotes" class="btn-primary">更新注释</button>
                </div>
            </div>
        </div>
    `,
    data() {
        return {
            newTag: ''
        };
    },
    methods: {
        formatLabel(key) {
            return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        },
        formatConfigValue(value) {
            if (typeof value === 'object') {
                return JSON.stringify(value, null, 2);
            }
            return String(value);
        },
        async addTag() {
            if (!this.newTag.trim()) return;
            
            const tags = [...(this.experiment.tags || []), this.newTag.trim()];
            
            try {
                const response = await fetch(`/api/experiments/${this.experiment.run_id}/tags`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(tags)
                });
                
                if (response.ok) {
                    this.experiment.tags = tags;
                    this.newTag = '';
                    this.$root.showAlert('标签添加成功', 'success');
                } else {
                    this.$root.showAlert('标签添加失败', 'error');
                }
            } catch (error) {
                this.$root.showAlert('标签添加失败: ' + error.message, 'error');
            }
        },
        async removeTag(tagToRemove) {
            const tags = (this.experiment.tags || []).filter(tag => tag !== tagToRemove);
            
            try {
                const response = await fetch(`/api/experiments/${this.experiment.run_id}/tags`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(tags)
                });
                
                if (response.ok) {
                    this.experiment.tags = tags;
                    this.$root.showAlert('标签删除成功', 'success');
                } else {
                    this.$root.showAlert('标签删除失败', 'error');
                }
            } catch (error) {
                this.$root.showAlert('标签删除失败: ' + error.message, 'error');
            }
        },
        async updateNotes() {
            try {
                const response = await fetch(`/api/experiments/${this.experiment.run_id}/notes`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(this.experiment.notes)
                });
                
                if (response.ok) {
                    this.$root.showAlert('注释更新成功', 'success');
                } else {
                    this.$root.showAlert('注释更新失败', 'error');
                }
            } catch (error) {
                this.$root.showAlert('注释更新失败: ' + error.message, 'error');
            }
        }
    }
};

const ExperimentMedia = {
    props: ['experiment'],
    template: `
        <div class="space-y-6">
            <div class="flex justify-between items-center">
                <h3 class="text-lg font-semibold">媒体文件 ({{ experiment.media_files?.length || 0 }})</h3>
                <div class="flex space-x-2">
                    <button @click="viewMode = 'grid'" :class="['btn-secondary text-sm', viewMode === 'grid' ? 'bg-blue-600 text-white' : '']">
                        <i class="fas fa-th"></i>
                    </button>
                    <button @click="viewMode = 'list'" :class="['btn-secondary text-sm', viewMode === 'list' ? 'bg-blue-600 text-white' : '']">
                        <i class="fas fa-list"></i>
                    </button>
                </div>
            </div>
            
            <!-- 网格视图 -->
            <div v-if="viewMode === 'grid'" class="media-grid">
                <div v-for="file in experiment.media_files" :key="file.name" 
                     class="media-item bg-gray-100 rounded-lg overflow-hidden"
                     @click="openMedia(file)">
                    <img v-if="file.type === 'image'" :src="file.url" :alt="file.name" 
                         class="w-full h-full object-cover">
                    <div v-else-if="file.type === 'audio'" class="w-full h-full flex items-center justify-center bg-blue-100">
                        <i class="fas fa-music text-4xl text-blue-500"></i>
                    </div>
                    <div v-else-if="file.type === 'video'" class="w-full h-full flex items-center justify-center bg-green-100">
                        <i class="fas fa-video text-4xl text-green-500"></i>
                    </div>
                    <div v-else class="w-full h-full flex items-center justify-center bg-gray-200">
                        <i class="fas fa-file text-4xl text-gray-500"></i>
                    </div>
                    
                    <div class="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white p-2">
                        <div class="text-sm font-medium truncate">{{ file.name }}</div>
                        <div class="text-xs opacity-75">{{ formatFileSize(file.size) }}</div>
                    </div>
                </div>
            </div>
            
            <!-- 列表视图 -->
            <div v-if="viewMode === 'list'" class="table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>文件名</th>
                            <th>类型</th>
                            <th>大小</th>
                            <th>修改时间</th>
                            <th>操作</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="file in experiment.media_files" :key="file.name">
                            <td>{{ file.name }}</td>
                            <td>
                                <span :class="['px-2 py-1 rounded text-xs', getFileTypeClass(file.type)]">
                                    {{ file.type }}
                                </span>
                            </td>
                            <td>{{ formatFileSize(file.size) }}</td>
                            <td>{{ formatDate(file.modified) }}</td>
                            <td>
                                <button @click="openMedia(file)" class="btn-secondary text-sm px-2 py-1">
                                    <i class="fas fa-eye mr-1"></i>查看
                                </button>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <!-- 空状态 -->
            <div v-if="!experiment.media_files || experiment.media_files.length === 0" class="text-center py-12">
                <i class="fas fa-file-image text-6xl text-gray-300 mb-4"></i>
                <h3 class="text-xl font-semibold text-gray-700 mb-2">暂无媒体文件</h3>
                <p class="text-gray-500">使用 wandb.Image, wandb.Audio 等API来记录媒体文件</p>
            </div>
        </div>
    `,
    data() {
        return {
            viewMode: 'grid'
        };
    },
    methods: {
        formatFileSize(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        },
        formatDate(timestamp) {
            if (!timestamp) return 'N/A';
            const date = new Date(timestamp * 1000);
            return date.toLocaleDateString('zh-CN') + ' ' + date.toLocaleTimeString('zh-CN');
        },
        getFileTypeClass(type) {
            const classMap = {
                'image': 'bg-blue-100 text-blue-800',
                'audio': 'bg-green-100 text-green-800',
                'video': 'bg-purple-100 text-purple-800',
                'json': 'bg-yellow-100 text-yellow-800'
            };
            return classMap[type] || 'bg-gray-100 text-gray-800';
        },
        openMedia(file) {
            if (file.type === 'image') {
                window.open(file.url, '_blank');
            } else if (file.type === 'audio') {
                // 创建音频播放器
                const audio = new Audio(file.url);
                audio.play();
            } else {
                // 下载文件
                const link = document.createElement('a');
                link.href = file.url;
                link.download = file.name;
                link.click();
            }
        }
    }
};

const ExperimentArtifacts = {
    props: ['experiment'],
    template: `
        <div class="space-y-6">
            <div class="flex justify-between items-center">
                <h3 class="text-lg font-semibold">Artifacts ({{ experiment.artifacts?.length || 0 }})</h3>
            </div>
            
            <div v-if="experiment.artifacts && experiment.artifacts.length > 0" class="space-y-4">
                <div v-for="artifact in experiment.artifacts" :key="artifact.name" 
                     class="sweep-card">
                    <div class="flex justify-between items-start mb-3">
                        <div>
                            <h4 class="text-lg font-semibold text-gray-900">{{ artifact.name }}</h4>
                            <p class="text-sm text-gray-600 mt-1">
                                类型: {{ artifact.metadata.type || 'unknown' }} | 
                                文件数: {{ artifact.files }} | 
                                大小: {{ formatFileSize(artifact.size) }}
                            </p>
                        </div>
                        <button @click="downloadArtifact(artifact)" class="btn-primary text-sm">
                            <i class="fas fa-download mr-1"></i>下载
                        </button>
                    </div>
                    
                    <div v-if="artifact.metadata.description" class="mb-3">
                        <p class="text-sm text-gray-700">{{ artifact.metadata.description }}</p>
                    </div>
                    
                    <div class="parameter-space">
                        <h5 class="text-sm font-medium text-gray-700 mb-2">文件列表</h5>
                        <div class="grid grid-cols-2 md:grid-cols-3 gap-2 text-sm">
                            <div v-for="(fileInfo, fileName) in artifact.metadata.files" :key="fileName" 
                                 class="bg-white p-2 rounded border">
                                <div class="font-medium truncate">{{ fileName }}</div>
                                <div class="text-xs text-gray-500">{{ formatFileSize(fileInfo.size) }}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 空状态 -->
            <div v-else class="text-center py-12">
                <i class="fas fa-archive text-6xl text-gray-300 mb-4"></i>
                <h3 class="text-xl font-semibold text-gray-700 mb-2">暂无Artifacts</h3>
                <p class="text-gray-500">使用 wandb.Artifact API来管理数据版本</p>
            </div>
        </div>
    `,
    methods: {
        formatFileSize(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        },
        downloadArtifact(artifact) {
            // 实现artifact下载功能
            this.$root.showAlert('Artifact下载功能开发中', 'warning');
        }
    }
};

// 主应用
const app = Vue.createApp({
    components: {
        'experiment-card': ExperimentCard,
        'experiment-list-item': ExperimentListItem,
        'comparison-experiment-card': ComparisonExperimentCard,
        'experiment-detail-tabs': ExperimentDetailTabs,
        'experiment-overview': ExperimentOverview,
        'experiment-metrics': ExperimentMetrics,
        'experiment-config': ExperimentConfig,
        'experiment-media': ExperimentMedia,
        'experiment-artifacts': ExperimentArtifacts
    },
    data() {
        return {
            experiments: [],
            projects: [],
            sweeps: [],
            selectedExperiment: null,
            selectedExperiments: [],
            comparisonMode: false,
            darkMode: false,
            isLoading: false,
            connectionStatus: 'disconnected',
            websocket: null,
            searchQuery: '',
            filters: {
                project: '',
                statuses: ['running', 'finished'],
                timeRange: ''
            },
            selectedTags: [],
            viewMode: 'grid',
            currentPage: 1,
            pageSize: 12,
            alerts: [],
            alertId: 0
        };
    },
    computed: {
        connectionStatusText() {
            const statusMap = {
                'connected': '实时同步中',
                'disconnected': '连接断开',
                'error': '连接错误'
            };
            return statusMap[this.connectionStatus] || '未知状态';
        },
        stats() {
            return {
                totalExperiments: this.experiments.length,
                runningExperiments: this.experiments.filter(exp => exp.status === 'running').length,
                finishedExperiments: this.experiments.filter(exp => exp.status === 'finished').length,
                totalProjects: this.projects.length
            };
        },
        activeExperimentsCount() {
            return this.experiments.filter(exp => exp.status === 'running').length;
        },
        completedExperimentsCount() {
            return this.experiments.filter(exp => exp.status === 'finished').length;
        },
        averageDuration() {
            const durations = this.experiments
                .filter(exp => exp.duration > 0)
                .map(exp => exp.duration / 60); // 转换为分钟
            
            if (durations.length === 0) return 0;
            
            const avg = durations.reduce((sum, duration) => sum + duration, 0) / durations.length;
            return Math.round(avg);
        },
        availableTags() {
            const tags = new Set();
            this.experiments.forEach(exp => {
                if (exp.tags) {
                    exp.tags.forEach(tag => tags.add(tag));
                }
            });
            return Array.from(tags).sort();
        },
        filteredExperiments() {
            let filtered = [...this.experiments];
            
            // 搜索过滤
            if (this.searchQuery) {
                const query = this.searchQuery.toLowerCase();
                filtered = filtered.filter(exp => {
                    return exp.name.toLowerCase().includes(query) ||
                           exp.project.toLowerCase().includes(query) ||
                           (exp.tags && exp.tags.some(tag => tag.toLowerCase().includes(query))) ||
                           (exp.notes && exp.notes.toLowerCase().includes(query)) ||
                           JSON.stringify(exp.config).toLowerCase().includes(query);
                });
            }
            
            // 项目过滤
            if (this.filters.project) {
                filtered = filtered.filter(exp => exp.project === this.filters.project);
            }
            
            // 状态过滤
            if (this.filters.statuses.length > 0) {
                filtered = filtered.filter(exp => this.filters.statuses.includes(exp.status));
            }
            
            // 标签过滤
            if (this.selectedTags.length > 0) {
                filtered = filtered.filter(exp => {
                    return this.selectedTags.some(tag => (exp.tags || []).includes(tag));
                });
            }
            
            // 时间过滤
            if (this.filters.timeRange) {
                const now = new Date();
                const timeFilters = {
                    '1h': 60 * 60 * 1000,
                    '24h': 24 * 60 * 60 * 1000,
                    '7d': 7 * 24 * 60 * 60 * 1000,
                    '30d': 30 * 24 * 60 * 60 * 1000
                };
                
                const threshold = timeFilters[this.filters.timeRange];
                if (threshold) {
                    filtered = filtered.filter(exp => {
                        const startTime = new Date(exp.start_time);
                        return now - startTime <= threshold;
                    });
                }
            }
            
            return filtered;
        },
        paginatedExperiments() {
            const start = (this.currentPage - 1) * this.pageSize;
            const end = start + this.pageSize;
            return this.filteredExperiments.slice(start, end);
        },
        totalPages() {
            return Math.ceil(this.filteredExperiments.length / this.pageSize);
        }
    },
    methods: {
        async loadExperiments() {
            this.isLoading = true;
            
            try {
                const response = await fetch('/api/experiments.json');
                const data = await response.json();
                
                this.experiments = data.experiments;
                
                // 加载项目信息
                await this.loadProjects();
                
                // 加载sweep信息
                await this.loadSweeps();
                
            } catch (error) {
                console.error('Error loading experiments:', error);
                this.showAlert('加载实验数据失败', 'error');
            } finally {
                this.isLoading = false;
            }
        },
        async loadProjects() {
            try {
                const response = await fetch('/api/projects.json');
                const data = await response.json();
                this.projects = data.projects;
            } catch (error) {
                console.error('Error loading projects:', error);
            }
        },
        async loadSweeps() {
            try {
                // 暂时禁用sweeps，因为没有静态数据文件
                // const response = await fetch('/api/sweeps.json');
                return [];
                const data = await response.json();
                this.sweeps = data.sweeps;
            } catch (error) {
                console.error('Error loading sweeps:', error);
            }
        },
        setupWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.connectionStatus = 'connected';
            };
            
            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.connectionStatus = 'disconnected';
                // 尝试重新连接
                setTimeout(() => this.setupWebSocket(), 5000);
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.connectionStatus = 'error';
            };
        },
        handleWebSocketMessage(data) {
            if (data.type === 'new_experiment') {
                this.addNewExperiment(data.data);
            }
        },
        addNewExperiment(experimentData) {
            const exists = this.experiments.find(exp => exp.run_id === experimentData.run_id);
            if (!exists) {
                this.experiments.unshift(experimentData);
                this.showAlert('发现新实验: ' + experimentData.name, 'success');
            }
        },
        handleExperimentClick(experiment) {
            if (this.comparisonMode) {
                this.toggleExperimentSelection(experiment.run_id);
            } else {
                // 打开独立详情页面
                window.open('/experiment-detail.html?id=' + experiment.run_id, '_blank');
            }
        },
        closeExperimentModal() {
            this.selectedExperiment = null;
        },
        toggleComparisonMode() {
            this.comparisonMode = !this.comparisonMode;
            if (!this.comparisonMode) {
                this.selectedExperiments = [];
            }
            console.log('Comparison mode:', this.comparisonMode);
        },
        toggleExperimentSelection(runId, selected) {
            if (selected) {
                if (!this.selectedExperiments.includes(runId)) {
                    this.selectedExperiments.push(runId);
                }
            } else {
                this.selectedExperiments = this.selectedExperiments.filter(id => id !== runId);
            }
        },
        getExperimentById(runId) {
            return this.experiments.find(exp => exp.run_id === runId);
        },
        startComparison() {
            if (this.selectedExperiments.length < 2) {
                this.showAlert('请至少选择2个实验进行对比', 'warning');
                return;
            }
            
            // 实现对比分析功能
            this.showAlert(`开始对比分析 ${this.selectedExperiments.length} 个实验`, 'info');
            
            // 这里可以打开对比分析页面或模态框
            // 实现多实验指标对比图表
        },
        clearComparison() {
            this.selectedExperiments = [];
        },
        async refreshExperiment(runId) {
            try {
                await fetch(`/api/experiments/${runId}/refresh`, { method: 'POST' });
                await this.loadExperiments();
                this.showAlert('实验刷新成功', 'success');
            } catch (error) {
                this.showAlert('实验刷新失败', 'error');
            }
        },
        async refreshData() {
            await this.loadExperiments();
            this.showAlert('数据刷新成功', 'success');
        },
        performSearch() {
            this.currentPage = 1;
        },
        clearFilters() {
            this.searchQuery = '';
            this.filters = {
                project: '',
                statuses: ['running', 'finished'],
                timeRange: ''
            };
            this.selectedTags = [];
            this.currentPage = 1;
        },
        toggleTagFilter(tag) {
            const index = this.selectedTags.indexOf(tag);
            if (index > -1) {
                this.selectedTags.splice(index, 1);
            } else {
                this.selectedTags.push(tag);
            }
        },
        toggleDarkMode() {
            this.darkMode = !this.darkMode;
            document.body.classList.toggle('dark-mode', this.darkMode);
        },
        previousPage() {
            if (this.currentPage > 1) {
                this.currentPage--;
            }
        },
        nextPage() {
            if (this.currentPage < this.totalPages) {
                this.currentPage++;
            }
        },
        showAlert(message, type = 'info') {
            const alertId = ++this.alertId;
            this.alerts.push({
                id: alertId,
                title: type === 'success' ? '成功' : type === 'error' ? '错误' : type === 'warning' ? '警告' : '信息',
                message: message,
                type: type
            });
            
            // 3秒后自动移除
            setTimeout(() => {
                this.removeAlert(alertId);
            }, 3000);
        },
        removeAlert(alertId) {
            const index = this.alerts.findIndex(alert => alert.id === alertId);
            if (index > -1) {
                this.alerts.splice(index, 1);
            }
        }
    },
    mounted() {
        this.loadExperiments();
        this.setupWebSocket();
        
        // 定期刷新数据
        setInterval(() => {
            if (this.connectionStatus === 'connected') {
                this.loadExperiments();
            }
        }, 30000);
    }
});

// 挂载应用
app.mount('#app');