<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import PlotThreadsList from './timeline/PlotThreadsList.vue'
import TimelineCharacterGrid from './timeline/TimelineCharacterGrid.vue'
import TimelineHeader from './timeline/TimelineHeader.vue'
import TimelineStats from './timeline/TimelineStats.vue'
import TimelineSummaryCard from './timeline/TimelineSummaryCard.vue'
import TimelineTrack from './timeline/TimelineTrack.vue'
import { useTimelinePanel } from './timeline/useTimelinePanel'

const {
  errorMessage,
  expandedGroupIds,
  getThumbnailUrl,
  hasTimelineData,
  isEnhancedData,
  isLoading,
  isRegenerating,
  mainCharacters,
  plotArcs,
  plotThreads,
  regenerateTimeline,
  showPageDetail,
  storySummary,
  timelineData,
  toggleGroup,
  totalEvents,
  totalPages,
} = useTimelinePanel()
</script>

<template>
  <div class="timeline-tab">
    <TimelineHeader
      :is-loading="isLoading"
      :is-regenerating="isRegenerating"
      @regenerate="regenerateTimeline"
    />

    <div v-if="errorMessage" class="error-message">
      ⚠️ {{ errorMessage }}
    </div>

    <div class="timeline-container">
      <div v-if="isLoading" class="loading-state">
        <div class="loading-spinner"></div>
        <p>加载时间线...</p>
      </div>

      <div v-else-if="!hasTimelineData" class="timeline-empty-state">
        <div class="empty-icon">📈</div>
        <h4>时间线尚未生成</h4>
        <p>完成漫画分析后会自动生成时间线，或点击下方按钮手动生成</p>
        <UiButton
          variant="primary"
          size="sm"
          class="timeline-empty-state__action"
          :disabled="isRegenerating"
          @click="regenerateTimeline"
        >
          {{ isRegenerating ? '生成中...' : '生成时间线' }}
        </UiButton>
      </div>

      <template v-else>
        <TimelineStats
          :stats="timelineData?.stats"
          :total-events="totalEvents"
          :total-pages="totalPages"
        />

        <TimelineSummaryCard
          v-if="storySummary"
          :plot-threads="plotThreads"
          :story-summary="storySummary"
        />

        <TimelineCharacterGrid
          v-if="mainCharacters.length > 0"
          :characters="mainCharacters"
          @show-page="showPageDetail"
        />

        <div v-if="isEnhancedData && plotArcs.length > 0" class="timeline-section">
          <h4>🎭 剧情发展</h4>
        </div>

        <TimelineTrack
          :expanded-ids="expandedGroupIds"
          :groups="timelineData?.groups || []"
          :is-enhanced-data="isEnhancedData"
          :plot-arcs="plotArcs"
          :thumbnail-url-for="getThumbnailUrl"
          @show-page="showPageDetail"
          @toggle="toggleGroup"
        />

        <div v-if="plotThreads.length > 0" class="timeline-section">
          <h4>🔗 伏笔与线索</h4>
          <PlotThreadsList :threads="plotThreads" />
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.timeline-tab {
  --timeline-panel-card-shadow: rgba(0, 0, 0, .1);
  --timeline-panel-character-shadow: rgba(0, 0, 0, .05);
  --timeline-panel-summary-tag-surface: rgba(255, 255, 255, .2);
  --ui-button-padding: 10px 18px;
  --ui-button-font-size: 14px;
  --ui-button-primary-background: var(--insight-color-primary);
  --ui-button-primary-hover-background: var(--insight-primary-dark);
  --ui-button-secondary-background: var(--insight-bg-tertiary);
  --ui-button-secondary-color: var(--insight-text-primary);
  --ui-button-secondary-border: 1px solid var(--color-border-muted);
  --ui-button-secondary-hover-background: var(--color-border-muted);
  --ui-button-sm-padding: 8px 14px;
  --ui-button-sm-font-size: 13px;
  --ui-button-disabled-opacity: 0.6;
}

.timeline-container {
  position: relative;
  max-height: calc(100dvh - 200px);
  padding: 20px;
  overflow-y: auto;
}

.error-message {
  margin-bottom: 12px;
  padding: 8px 12px;
  border-radius: 4px;
  background: var(--color-focus-danger-soft);
  color: var(--color-status-error);
  font-size: 12px;
}

.loading-state {
  padding: 40px;
  color: var(--insight-text-secondary);
  text-align: center;
}

.loading-spinner {
  width: 32px;
  height: 32px;
  margin: 0 auto 12px;
  border: 3px solid var(--color-border-muted);
  border-top-color: var(--insight-color-primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.timeline-empty-state {
  padding: 60px 20px;
  text-align: center;
}

.empty-icon {
  margin-bottom: 16px;
  font-size: 48px;
}

.timeline-empty-state h4 {
  margin: 0 0 8px;
  color: var(--insight-text-primary);
  font-weight: 600;
  font-size: 18px;
}

.timeline-empty-state p {
  margin: 0 0 20px;
  color: var(--insight-text-secondary);
  font-size: 14px;
}

.timeline-empty-state__action {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.timeline-section {
  margin-bottom: 28px;
}

.timeline-section h4 {
  display: inline-block;
  margin: 0 0 16px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--insight-color-primary);
  color: var(--insight-text-primary);
  font-weight: 600;
  font-size: 16px;
}
</style>
