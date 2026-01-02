<script setup lang="ts">
/**
 * 时间线面板组件
 * 显示漫画剧情时间线，支持简单模式和增强模式
 * 简单模式：事件分组、缩略图
 * 增强模式：剧情弧、角色、线索
 */

import { ref, computed, onMounted, watch } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'

// ============================================================
// 类型定义
// ============================================================

/** 时间线模式 */
type TimelineMode = 'simple' | 'enhanced'

/** 时间线分组 */
interface TimelineGroup {
  id: string
  page_range: { start: number; end: number }
  events: string[]
  summary?: string
  thumbnail_page?: number
  // 增强模式字段
  plot_arc?: string
  characters?: string[]
  clues?: string[]
  mood?: string
}

/** 时间线数据 */
interface TimelineData {
  mode?: string
  groups?: TimelineGroup[]
  events?: any[]
  stats?: {
    total_events: number
    total_pages: number
    total_arcs?: number
    total_characters?: number
    total_threads?: number
  }
  // 增强模式额外数据
  story_summary?: string
  main_characters?: Array<{
    name: string
    description: string
    first_appearance: number
    arc?: string
    key_moments?: any[]
  }>
  plot_arcs?: Array<{
    id?: string
    name: string
    description: string
    page_range?: { start: number; end: number }
    start_page?: number
    end_page?: number
    mood?: string
    event_ids?: string[]
  }>
  plot_threads?: Array<{
    id: string
    name: string
    type: string
    status: string
    description: string
    introduced_at?: number
    resolved_at?: number | null
  }>
  cached?: boolean
}

// ============================================================
// 状态
// ============================================================

const insightStore = useInsightStore()

/** 是否正在加载 */
const isLoading = ref(false)

/** 是否正在重新生成 */
const isRegenerating = ref(false)

/** 时间线数据 */
const timelineData = ref<TimelineData | null>(null)

/** 当前显示模式 */
const currentMode = ref<TimelineMode>('simple')

/** 展开的分组ID集合 */
const expandedGroups = ref<Set<string>>(new Set())

/** 错误消息 */
const errorMessage = ref('')

// ============================================================
// 计算属性
// ============================================================

/** 是否有时间线数据 */
const hasTimelineData = computed(() => {
  if (!timelineData.value) return false
  // 简单模式检查groups，增强模式检查story_arcs/plot_arcs
  const hasGroups = timelineData.value.groups && timelineData.value.groups.length > 0
  const hasArcs = timelineData.value.plot_arcs && timelineData.value.plot_arcs.length > 0
  return hasGroups || hasArcs
})

/** 总事件数 */
const totalEvents = computed(() => timelineData.value?.stats?.total_events || 0)

/** 总页数 */
const totalPages = computed(() => timelineData.value?.stats?.total_pages || 0)

/** 是否为增强模式数据 */
const isEnhancedData = computed(() => {
  return timelineData.value?.mode === 'enhanced' || 
         !!timelineData.value?.story_summary ||
         !!timelineData.value?.plot_arcs?.length
})

/** 主要角色列表 */
const mainCharacters = computed(() => timelineData.value?.main_characters || [])

/** 剧情弧列表 */
const plotArcs = computed(() => timelineData.value?.plot_arcs || [])

/** 故事摘要 */
const storySummary = computed(() => timelineData.value?.story_summary || '')

// ============================================================
// 方法
// ============================================================

/**
 * 加载时间线
 */
async function loadTimeline(): Promise<void> {
  if (!insightStore.currentBookId) return

  isLoading.value = true
  errorMessage.value = ''

  try {
    const response = await insightApi.getTimeline(insightStore.currentBookId) as any
    console.log('Timeline API response:', response)
    
    if (response.success) {
      // API返回格式根据mode不同而不同：
      // 简单模式: { success, cached, groups, events, stats, mode: "simple" }
      // 增强模式: { success, cached, story_arcs, characters, plot_threads, summary, stats, mode: "enhanced" }
      timelineData.value = {
        mode: response.mode || 'simple',
        // 简单模式使用groups，增强模式使用story_arcs
        groups: response.groups || [],
        stats: response.stats,
        // 增强模式数据
        story_summary: response.summary?.one_sentence || '',
        main_characters: response.characters || [],
        plot_arcs: response.story_arcs || [],
        plot_threads: response.plot_threads || [],
        events: response.events || [],
        cached: response.cached
      }
      console.log('Parsed timelineData:', timelineData.value)
    } else {
      errorMessage.value = response.error || '加载时间线失败'
    }
  } catch (error) {
    console.error('加载时间线失败:', error)
    errorMessage.value = error instanceof Error ? error.message : '加载失败'
  } finally {
    isLoading.value = false
  }
}

/**
 * 重新生成时间线
 */
async function regenerateTimeline(): Promise<void> {
  if (!insightStore.currentBookId) return

  isRegenerating.value = true
  errorMessage.value = ''

  try {
    const response = await insightApi.regenerateTimeline(insightStore.currentBookId)
    if (response.success) {
      timelineData.value = response as unknown as TimelineData
    } else {
      errorMessage.value = '重新生成失败'
    }
  } catch (error) {
    console.error('重新生成时间线失败:', error)
    errorMessage.value = error instanceof Error ? error.message : '重新生成失败'
  } finally {
    isRegenerating.value = false
  }
}

/**
 * 获取缩略图URL
 * @param pageNum - 页码
 */
function getThumbnailUrl(pageNum: number): string {
  if (!insightStore.currentBookId) return ''
  return insightApi.getThumbnailUrl(insightStore.currentBookId, pageNum)
}

/**
 * 显示页面详情
 * @param pageNum - 页码
 */
function showPageDetail(pageNum: number): void {
  insightStore.selectPage(pageNum)
}

/**
 * 切换分组展开状态
 * @param groupId - 分组ID
 */
function toggleGroup(groupId: string): void {
  if (expandedGroups.value.has(groupId)) {
    expandedGroups.value.delete(groupId)
  } else {
    expandedGroups.value.add(groupId)
  }
}

/**
 * 检查分组是否展开
 * @param groupId - 分组ID
 */
function isGroupExpanded(groupId: string): boolean {
  return expandedGroups.value.has(groupId)
}

/**
 * 切换显示模式
 * @param mode - 模式
 */
function switchMode(mode: TimelineMode): void {
  currentMode.value = mode
}

/**
 * 展开所有分组
 */
function expandAll(): void {
  if (timelineData.value?.groups) {
    timelineData.value.groups.forEach(g => expandedGroups.value.add(g.id))
  }
}

/**
 * 折叠所有分组
 */
function collapseAll(): void {
  expandedGroups.value.clear()
}

// ============================================================
// 生命周期
// ============================================================

onMounted(() => {
  if (insightStore.currentBookId) {
    loadTimeline()
  }
})

// 监听书籍ID变化，重新加载时间线
watch(() => insightStore.currentBookId, (newBookId) => {
  if (newBookId) {
    timelineData.value = null
    loadTimeline()
  }
})

// 监听数据刷新触发器（分析完成后自动刷新）
watch(() => insightStore.dataRefreshKey, (newKey) => {
  if (newKey > 0 && insightStore.currentBookId) {
    console.log('TimelinePanel: 收到刷新信号，重新加载数据')
    loadTimeline()
  }
})
</script>

<template>
  <div class="timeline-tab">
    <!-- 头部 -->
    <div class="timeline-header">
      <h3>📈 剧情时间线</h3>
      <button 
        class="btn btn-secondary btn-sm" 
        :disabled="isLoading || isRegenerating"
        :class="{ loading: isRegenerating }"
        @click="regenerateTimeline"
      >
        <span v-if="isRegenerating" class="btn-spinner"></span>
        {{ isRegenerating ? '生成中...' : '🔄 重新生成' }}
      </button>
    </div>
    
    <!-- 错误消息 -->
    <div v-if="errorMessage" class="error-message">
      ⚠️ {{ errorMessage }}
    </div>
    
    <div class="timeline-container">
      <!-- 加载中 -->
      <div v-if="isLoading" class="loading-state">
        <div class="loading-spinner"></div>
        <p>加载时间线...</p>
      </div>
      
      <!-- 无数据 -->
      <div v-else-if="!hasTimelineData" class="timeline-empty-state">
        <div class="empty-icon">📈</div>
        <h4>时间线尚未生成</h4>
        <p>完成漫画分析后会自动生成时间线，或点击下方按钮手动生成</p>
        <button 
          class="btn btn-primary btn-sm" 
          :disabled="isRegenerating"
          @click="regenerateTimeline"
        >
          {{ isRegenerating ? '生成中...' : '生成时间线' }}
        </button>
      </div>
      
      <!-- 时间线内容 -->
      <template v-else>
        <!-- 统计信息 -->
        <div class="timeline-stats">
          <span v-if="timelineData?.stats?.total_arcs" class="stat-badge">🎭 {{ timelineData.stats.total_arcs }} 个剧情弧</span>
          <span class="stat-badge">📊 {{ totalEvents }} 个事件</span>
          <span v-if="timelineData?.stats?.total_characters" class="stat-badge">👥 {{ timelineData.stats.total_characters }} 个角色</span>
          <span v-if="timelineData?.stats?.total_threads" class="stat-badge">🔗 {{ timelineData.stats.total_threads }} 条线索</span>
          <span class="stat-badge">📄 {{ totalPages }} 页</span>
        </div>
        
        <!-- 故事概要卡片 -->
        <div v-if="storySummary" class="timeline-summary-card">
          <h4>📖 故事概要</h4>
          <p class="one-sentence">{{ storySummary }}</p>
          <div v-if="timelineData?.plot_threads?.length" class="themes">
            <span>主题：</span>
            <span 
              v-for="thread in timelineData.plot_threads.slice(0, 5)" 
              :key="thread.id"
              class="theme-tag"
            >{{ thread.name }}</span>
          </div>
        </div>
        
        <!-- 主要角色 -->
        <div v-if="mainCharacters.length > 0" class="characters-section">
          <h4>👥 主要角色</h4>
          <div class="characters-grid">
            <div 
              v-for="char in mainCharacters" 
              :key="char.name"
              class="character-card"
              @click="showPageDetail(char.first_appearance)"
            >
              <div class="character-name">{{ char.name }}</div>
              <div class="character-desc">{{ char.description }}</div>
              <div class="character-page">首次出现：第 {{ char.first_appearance }} 页</div>
            </div>
          </div>
        </div>
        
        <!-- 剧情发展标题 -->
        <div v-if="isEnhancedData && plotArcs.length > 0" class="timeline-section">
          <h4>🎭 剧情发展</h4>
        </div>
        
        <!-- 增强模式：剧情弧时间线 -->
        <div v-if="isEnhancedData && plotArcs.length > 0" class="timeline-track">
          <div 
            v-for="(arc, index) in plotArcs" 
            :key="arc.id || arc.name"
            class="timeline-group"
            :class="{ expanded: isGroupExpanded(arc.id || `arc-${index}`) }"
          >
            <div class="timeline-node">
              <div class="timeline-node-dot" @click="toggleGroup(arc.id || `arc-${index}`)"></div>
              <div class="timeline-node-line"></div>
            </div>
            <div class="timeline-card">
              <!-- 卡片头部 -->
              <div class="timeline-card-header" @click="toggleGroup(arc.id || `arc-${index}`)">
                <img 
                  class="timeline-thumbnail" 
                  :src="getThumbnailUrl(arc.page_range?.start || 1)" 
                  :alt="`第${arc.page_range?.start || 1}页`"
                  loading="lazy"
                  @error="($event.target as HTMLImageElement).style.display = 'none'"
                  @click.stop="showPageDetail(arc.page_range?.start || 1)"
                >
                <div class="timeline-card-title">
                  <span class="timeline-page-range">
                    第 {{ arc.page_range?.start || arc.start_page || '?' }}-{{ arc.page_range?.end || arc.end_page || '?' }} 页
                  </span>
                  <span class="timeline-event-count">{{ arc.name }}</span>
                </div>
                <span class="expand-icon">{{ isGroupExpanded(arc.id || `arc-${index}`) ? '▼' : '▶' }}</span>
              </div>
              
              <!-- 剧情弧描述 -->
              <div v-if="arc.description" class="timeline-summary">{{ arc.description }}</div>
              
              <!-- 氛围 -->
              <div v-if="arc.mood" class="timeline-mood">
                <span class="label">🎨 氛围：</span>{{ arc.mood }}
              </div>
            </div>
          </div>
        </div>
        
        <!-- 简单模式：事件分组时间线 -->
        <div v-else-if="timelineData?.groups && timelineData.groups.length > 0" class="timeline-track">
          <div 
            v-for="group in timelineData.groups" 
            :key="group.id"
            class="timeline-group"
            :class="{ expanded: isGroupExpanded(group.id) }"
          >
            <div class="timeline-node">
              <div class="timeline-node-dot" @click="toggleGroup(group.id)"></div>
              <div class="timeline-node-line"></div>
            </div>
            <div class="timeline-card">
              <!-- 卡片头部 -->
              <div class="timeline-card-header" @click="toggleGroup(group.id)">
                <img 
                  class="timeline-thumbnail" 
                  :src="getThumbnailUrl(group.thumbnail_page || group.page_range.start)" 
                  :alt="`第${group.page_range.start}页`"
                  loading="lazy"
                  @error="($event.target as HTMLImageElement).style.display = 'none'"
                  @click.stop="showPageDetail(group.page_range.start)"
                >
                <div class="timeline-card-title">
                  <span class="timeline-page-range">
                    第 {{ group.page_range.start }}-{{ group.page_range.end }} 页
                  </span>
                  <span class="timeline-event-count">{{ group.events.length }} 个事件</span>
                </div>
                <span class="expand-icon">{{ isGroupExpanded(group.id) ? '▼' : '▶' }}</span>
              </div>
              
              <!-- 摘要 -->
              <div v-if="group.summary" class="timeline-summary">{{ group.summary }}</div>
              
              <!-- 事件列表（展开时显示） -->
              <ul v-if="isGroupExpanded(group.id) && group.events.length > 0" class="timeline-events-list">
                <li 
                  v-for="(event, index) in group.events" 
                  :key="index"
                  class="timeline-event-item"
                >
                  {{ event }}
                </li>
              </ul>
            </div>
          </div>
        </div>
        
        <!-- 伏笔与线索 -->
        <div v-if="timelineData?.plot_threads && timelineData.plot_threads.length > 0" class="timeline-section">
          <h4>🔗 伏笔与线索</h4>
          <div class="plot-threads-list">
            <div 
              v-for="thread in timelineData.plot_threads" 
              :key="thread.id"
              class="plot-thread-item"
              :class="{ resolved: thread.status === '已解决' }"
            >
              <div class="thread-header">
                <span class="thread-name">{{ thread.name || '未命名线索' }}</span>
                <span class="thread-status" :class="{ resolved: thread.status === '已解决' }">
                  {{ thread.status || '进行中' }}
                </span>
              </div>
              <p v-if="thread.description" class="thread-desc">{{ thread.description }}</p>
              <span v-if="thread.introduced_at" class="thread-intro">第 {{ thread.introduced_at }} 页引入</span>
            </div>
          </div>
        </div>
        
        <!-- 无时间线数据时的提示 -->
        <div v-if="!hasTimelineData" class="placeholder-text">
          暂无详细时间线数据，点击"重新生成"按钮生成
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
/* ==================== TimelinePanel 完整样式 ==================== */

/* ==================== CSS变量 ==================== */
.timeline-tab {
  --bg-primary: #f8fafc;
  --bg-secondary: #ffffff;
  --bg-tertiary: #f1f5f9;
  --bg-hover: rgba(99, 102, 241, 0.1);
  --text-primary: #1a202c;
  --text-secondary: #64748b;
  --text-muted: #94a3b8;
  --border-color: #e2e8f0;
  --primary-color: #6366f1;
  --primary: #6366f1;
  --primary-light: #818cf8;
  --primary-dark: #4f46e5;
  --success-color: #22c55e;
  --success: #22c55e;
  --warning-color: #f59e0b;
  --error-color: #ef4444;
  --danger: #ef4444;
}

/* ==================== 按钮样式 ==================== */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 10px 18px;
  font-size: 14px;
  font-weight: 500;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  text-decoration: none;
}

.btn-primary {
  background: var(--primary-color);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: var(--primary-dark);
}

.btn-primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-secondary {
  background: var(--bg-tertiary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
}

.btn-secondary:hover:not(:disabled) {
  background: var(--border-color);
}

.btn-secondary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-sm {
  padding: 8px 14px;
  font-size: 13px;
}

/* ==================== 组件特定样式 ==================== */

/* 头部 */
.timeline-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.timeline-header h3 {
  margin: 0;
  font-size: 18px;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

/* 模式切换 */
.mode-toggle {
  display: flex;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  overflow: hidden;
}

.mode-btn {
  padding: 4px 12px;
  font-size: 12px;
  border: none;
  background: var(--bg-secondary);
  cursor: pointer;
  transition: all 0.2s;
}

.mode-btn:first-child {
  border-right: 1px solid var(--border-color);
}

.mode-btn.active {
  background: var(--primary);
  color: white;
}

.mode-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* 错误消息 */
.error-message {
  font-size: 12px;
  color: var(--danger, #ef4444);
  background: rgba(239, 68, 68, 0.1);
  padding: 8px 12px;
  border-radius: 4px;
  margin-bottom: 12px;
}

/* 加载状态 */
.loading-state {
  text-align: center;
  padding: 40px;
  color: var(--text-secondary);
}

.loading-spinner {
  width: 32px;
  height: 32px;
  border: 3px solid var(--border-color);
  border-top-color: var(--primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin: 0 auto 12px;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* 空状态 */
.timeline-empty-state {
  text-align: center;
  padding: 40px 20px;
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.timeline-empty-state h4 {
  margin: 0 0 8px 0;
  font-size: 18px;
}

.timeline-empty-state p {
  color: var(--text-secondary);
  margin: 0 0 16px 0;
}

/* 工具栏 */
.timeline-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border-color);
}

.timeline-stats {
  display: flex;
  gap: 8px;
}

.stat-badge {
  font-size: 12px;
  padding: 4px 8px;
  background: var(--bg-secondary);
  border-radius: 4px;
}

.stat-badge.cached {
  background: rgba(34, 197, 94, 0.1);
  color: var(--success, #22c55e);
}

.timeline-actions {
  display: flex;
  gap: 8px;
}

.btn-text {
  font-size: 12px;
  color: var(--primary);
  background: none;
  border: none;
  cursor: pointer;
  padding: 4px 8px;
}

.btn-text:hover {
  text-decoration: underline;
}

/* 故事概要卡片 - 与原版一致的紫色渐变背景 */
.timeline-summary-card {
  background: linear-gradient(135deg, var(--primary-color, #6366f1) 0%, var(--primary-dark, #4f46e5) 100%);
  color: white;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 24px;
}

.timeline-summary-card h4 {
  margin: 0 0 12px 0;
  font-size: 16px;
  font-weight: 600;
}

.timeline-summary-card .one-sentence {
  font-size: 15px;
  line-height: 1.6;
  margin-bottom: 12px;
}

.timeline-summary-card .themes {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.timeline-summary-card .theme-tag {
  background: rgba(255, 255, 255, 0.2);
  padding: 4px 10px;
  border-radius: 20px;
  font-size: 12px;
}

/* 剧情发展标题 */
.timeline-section {
  margin-bottom: 20px;
}

.timeline-section h4 {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0 0 16px 0;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--primary-color, #6366f1);
  display: inline-block;
}

/* 角色部分 */
.characters-section {
  margin-bottom: 20px;
}

.characters-section h4 {
  font-size: 14px;
  margin: 0 0 12px 0;
}

.characters-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 12px;
}

.character-card {
  background: var(--bg-secondary);
  border-radius: 8px;
  padding: 12px;
  cursor: pointer;
  transition: all 0.2s;
}

.character-card:hover {
  background: var(--bg-hover);
  transform: translateY(-2px);
}

.character-name {
  font-weight: 600;
  font-size: 14px;
  margin-bottom: 4px;
}

.character-desc {
  font-size: 12px;
  color: var(--text-secondary);
  margin-bottom: 6px;
  line-height: 1.4;
}

.character-page {
  font-size: 11px;
  color: var(--primary);
}

/* 伏笔与线索 */
.plot-threads-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.plot-thread-item {
  background: var(--bg-secondary);
  border-radius: 10px;
  padding: 14px;
  border-left: 3px solid var(--warning-color, #f59e0b);
}

.plot-thread-item.resolved {
  border-left-color: var(--success-color, #22c55e);
  opacity: 0.8;
}

.thread-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.thread-name {
  font-weight: 600;
  font-size: 14px;
  color: var(--text-primary);
}

.thread-status {
  font-size: 11px;
  padding: 3px 10px;
  border-radius: 10px;
  background: var(--warning-color, #f59e0b);
  color: white;
}

.thread-status.resolved {
  background: var(--success-color, #22c55e);
}

.thread-desc {
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.5;
  margin: 0 0 8px 0;
}

.thread-intro {
  font-size: 12px;
  color: var(--text-muted);
}

/* 剧情弧部分 */
.plot-arcs-section {
  margin-bottom: 20px;
}

.plot-arcs-section h4 {
  font-size: 14px;
  margin: 0 0 12px 0;
}

.plot-arcs-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.plot-arc-item {
  background: var(--bg-secondary);
  border-radius: 6px;
  padding: 10px 12px;
}

.arc-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}

.arc-name {
  font-weight: 500;
  font-size: 13px;
}

.arc-pages {
  font-size: 11px;
  color: var(--text-secondary);
}

.arc-desc {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.4;
}

/* 时间线轨道 */
.timeline-track {
  position: relative;
  padding-left: 24px;
}

.timeline-group {
  position: relative;
  margin-bottom: 16px;
}

.timeline-node {
  position: absolute;
  left: -24px;
  top: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.timeline-node-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: var(--primary);
  cursor: pointer;
  transition: transform 0.2s;
}

.timeline-node-dot:hover {
  transform: scale(1.2);
}

.timeline-node-line {
  width: 2px;
  flex: 1;
  background: var(--border-color);
  min-height: 40px;
}

.timeline-group:last-child .timeline-node-line {
  display: none;
}

/* 时间线卡片 */
.timeline-card {
  background: var(--bg-secondary);
  border-radius: 8px;
  padding: 12px;
  margin-left: 8px;
}

.timeline-card-header {
  display: flex;
  align-items: center;
  gap: 12px;
  cursor: pointer;
}

.timeline-thumbnail {
  width: 48px;
  height: 64px;
  object-fit: cover;
  border-radius: 4px;
  cursor: pointer;
}

.timeline-thumbnail:hover {
  opacity: 0.8;
}

.timeline-card-title {
  flex: 1;
}

.timeline-page-range {
  display: block;
  font-weight: 500;
  font-size: 14px;
  margin-bottom: 2px;
}

.timeline-event-count {
  font-size: 12px;
  color: var(--text-secondary);
}

.expand-icon {
  font-size: 10px;
  color: var(--text-secondary);
  transition: transform 0.2s;
}

.timeline-group.expanded .expand-icon {
  transform: rotate(0deg);
}

/* 时间线内容 */
.timeline-summary {
  font-size: 13px;
  color: var(--text-secondary);
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid var(--border-color);
  line-height: 1.5;
}

.timeline-plot-arc,
.timeline-characters,
.timeline-mood {
  font-size: 12px;
  margin-top: 6px;
}

.timeline-plot-arc .label,
.timeline-characters .label,
.timeline-mood .label,
.timeline-clues .label {
  color: var(--text-secondary);
}

/* 事件列表 */
.timeline-events-list {
  margin: 10px 0 0 0;
  padding: 10px 0 0 16px;
  border-top: 1px dashed var(--border-color);
  list-style: disc;
}

.timeline-event-item {
  font-size: 13px;
  margin-bottom: 6px;
  line-height: 1.4;
}

/* 线索 */
.timeline-clues {
  font-size: 12px;
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed var(--border-color);
}

.timeline-clues ul {
  margin: 4px 0 0 16px;
  padding: 0;
}

.timeline-clues li {
  margin-bottom: 4px;
}

/* 按钮加载状态 */
.btn.loading {
  opacity: 0.7;
  cursor: wait;
}

.btn-spinner {
  display: inline-block;
  width: 12px;
  height: 12px;
  border: 2px solid currentColor;
  border-right-color: transparent;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-right: 6px;
}

/* ==================== 时间线完整样式 - 从 manga-insight.css 迁移 ==================== */

.timeline-container {
    padding: 20px;
    position: relative;
    max-height: calc(100vh - 200px);
    overflow-y: auto;
}

.timeline-stats {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-color);
}

.timeline-stats .stat-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 6px 12px;
    background: var(--bg-tertiary);
    border-radius: 16px;
    font-size: 13px;
    color: var(--text-secondary);
}

.timeline-track {
    position: relative;
    padding-left: 20px;
}

.timeline-group {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
    position: relative;
}

.timeline-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 20px;
    flex-shrink: 0;
}

.timeline-node-dot {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--primary-color);
    border: 3px solid var(--bg-primary);
    box-shadow: 0 0 0 2px var(--primary-color);
    z-index: 1;
}

.timeline-node-line {
    flex: 1;
    width: 2px;
    background: linear-gradient(180deg, var(--primary-color), var(--border-color));
    margin-top: 4px;
}

.timeline-group:last-child .timeline-node-line {
    display: none;
}

.timeline-card {
    flex: 1;
    background: var(--bg-secondary);
    border-radius: 12px;
    border: 1px solid var(--border-color);
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}

.timeline-card:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.timeline-card-header {
    display: flex;
    gap: 12px;
    padding: 12px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border-color);
}

.timeline-thumbnail {
    width: 60px;
    height: 80px;
    object-fit: cover;
    border-radius: 6px;
    cursor: pointer;
    transition: transform 0.2s;
    background: var(--bg-primary);
}

.timeline-thumbnail:hover {
    transform: scale(1.05);
}

.timeline-card-title {
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 4px;
}

.timeline-page-range {
    font-weight: 600;
    font-size: 15px;
    color: var(--text-primary);
}

.timeline-event-count {
    font-size: 12px;
    color: var(--text-secondary);
    background: var(--bg-primary);
    padding: 2px 8px;
    border-radius: 10px;
    display: inline-block;
    width: fit-content;
}

.timeline-summary {
    padding: 12px;
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.6;
    border-bottom: 1px solid var(--border-color);
}

.timeline-events-list {
    margin: 0;
    padding: 12px;
    padding-left: 28px;
    list-style: none;
}

.timeline-event-item {
    position: relative;
    padding: 6px 0;
    font-size: 13px;
    color: var(--text-primary);
    line-height: 1.5;
}

.timeline-event-item::before {
    content: '';
    position: absolute;
    left: -16px;
    top: 12px;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--primary-color);
}

.timeline-event-item:not(:last-child) {
    border-bottom: 1px dashed var(--border-color);
}

.timeline-container .placeholder-text {
    text-align: center;
    padding: 60px 20px;
    color: var(--text-secondary);
}

.timeline-empty-state {
    text-align: center;
    padding: 60px 20px;
}

.timeline-empty-state .empty-icon {
    font-size: 48px;
    margin-bottom: 16px;
}

.timeline-empty-state h4 {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 8px;
    color: var(--text-primary);
}

.timeline-empty-state p {
    font-size: 14px;
    color: var(--text-secondary);
    margin-bottom: 20px;
}

.timeline-empty-state .btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* 增强时间线样式 */
.enhanced-timeline {
    padding: 16px;
}

.timeline-stats.enhanced {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-color);
}

.timeline-summary-card {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
    color: white;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 24px;
}

.timeline-summary-card h4 {
    margin: 0 0 12px 0;
    font-size: 16px;
    font-weight: 600;
}

.timeline-summary-card .one-sentence {
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 12px;
}

.timeline-summary-card .main-conflict {
    font-size: 14px;
    opacity: 0.9;
    margin-bottom: 12px;
}

.timeline-summary-card .themes {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    font-size: 14px;
}

.timeline-summary-card .theme-tag {
    background: rgba(255, 255, 255, 0.2);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
}

.timeline-section {
    margin-bottom: 28px;
}

.timeline-section h4 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--primary-color);
    display: inline-block;
}

.timeline-section h4.collapsible {
    cursor: pointer;
    user-select: none;
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    justify-content: space-between;
    border-bottom: none;
    padding: 10px 0;
}

.timeline-section h4.collapsible:hover {
    color: var(--primary-color);
}

.collapse-icon {
    font-size: 12px;
    transition: transform 0.2s;
}

.story-arcs-track {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.story-arc-card {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 16px;
    border-left: 4px solid var(--primary-color);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s, box-shadow 0.2s;
}

.story-arc-card:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.story-arc-card.mood-紧张 { border-left-color: #ef4444; }
.story-arc-card.mood-温馨 { border-left-color: #f59e0b; }
.story-arc-card.mood-悲伤 { border-left-color: #6366f1; }
.story-arc-card.mood-欢乐 { border-left-color: #22c55e; }
.story-arc-card.mood-神秘 { border-left-color: #8b5cf6; }
.story-arc-card.mood-激动 { border-left-color: #ec4899; }

.arc-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.arc-name {
    font-weight: 600;
    font-size: 15px;
    color: var(--text-primary);
}

.arc-pages {
    font-size: 12px;
    color: var(--text-secondary);
    background: var(--bg-primary);
    padding: 4px 10px;
    border-radius: 12px;
}

.arc-description {
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.5;
    margin: 0 0 10px 0;
}

.arc-mood {
    display: inline-block;
    font-size: 12px;
    padding: 3px 10px;
    background: var(--bg-tertiary);
    border-radius: 12px;
    color: var(--text-secondary);
}

.arc-events {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px dashed var(--border-color);
    font-size: 13px;
}

.arc-events strong {
    color: var(--text-primary);
    font-size: 12px;
}

.arc-events ul {
    margin: 8px 0 0 0;
    padding-left: 20px;
    color: var(--text-secondary);
}

.arc-events li {
    padding: 3px 0;
    line-height: 1.4;
}

.arc-events li.more {
    color: var(--text-muted);
    font-style: italic;
}

.characters-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
}

.character-card {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s;
}

.character-card:hover {
    transform: translateY(-2px);
}

.character-name {
    font-weight: 600;
    font-size: 15px;
    color: var(--text-primary);
    margin-bottom: 8px;
}

.character-desc {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.5;
    margin: 0 0 8px 0;
}

.character-arc {
    font-size: 13px;
    color: var(--text-secondary);
    margin: 0 0 8px 0;
}

.first-appear {
    font-size: 12px;
    color: var(--text-muted);
    background: var(--bg-primary);
    padding: 3px 8px;
    border-radius: 10px;
}

.plot-threads-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.plot-thread-item {
    background: var(--bg-secondary);
    border-radius: 10px;
    padding: 14px;
    border-left: 3px solid var(--warning-color);
}

.plot-thread-item.resolved {
    border-left-color: var(--success-color);
    opacity: 0.8;
}

.thread-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.thread-name {
    font-weight: 600;
    font-size: 14px;
    color: var(--text-primary);
}

.thread-status {
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 10px;
    background: var(--warning-color);
    color: white;
}

.thread-status.resolved {
    background: var(--success-color);
}

.thread-desc {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.5;
    margin: 0 0 8px 0;
}

.thread-intro {
    font-size: 12px;
    color: var(--text-muted);
}

.events-list-section {
    max-height: 800px;
    overflow-y: auto;
    transition: max-height 0.3s ease;
}

.events-list-section.collapsed {
    max-height: 0;
    overflow: hidden;
}

.event-item {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    padding: 10px 12px;
    background: var(--bg-secondary);
    border-radius: 8px;
    margin-bottom: 8px;
}

.event-item.importance-high {
    border-left: 3px solid var(--error-color);
}

.event-item.importance-critical {
    border-left: 3px solid #dc2626;
    background: rgba(239, 68, 68, 0.05);
}

.event-pages {
    font-size: 11px;
    color: var(--text-muted);
    background: var(--bg-primary);
    padding: 3px 8px;
    border-radius: 8px;
    white-space: nowrap;
    flex-shrink: 0;
}

.event-text {
    font-size: 13px;
    color: var(--text-primary);
    line-height: 1.5;
    flex: 1;
}

.event-chars {
    font-size: 11px;
    color: var(--primary-color);
    background: rgba(99, 102, 241, 0.1);
    padding: 2px 8px;
    border-radius: 8px;
    white-space: nowrap;
}
</style>
