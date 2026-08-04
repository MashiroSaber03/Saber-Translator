<script setup lang="ts">
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'
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
  hasMoreTimeline,
  isEnhancedData,
  isLoading,
  isLoadingMore,
  isRegenerating,
  mainCharacters,
  loadMoreTimeline,
  pendingMessage,
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
  <div class="timeline-panel">
    <TimelineHeader
      :is-loading="isLoading"
      :is-regenerating="isRegenerating"
      @regenerate="regenerateTimeline"
    />

    <ProductStatusBanner
      v-if="errorMessage"
      aria-live="assertive"
      class="timeline-panel__status-banner"
      tone="danger"
    >
      {{ errorMessage }}
    </ProductStatusBanner>

    <div class="timeline-panel__body">
      <div v-if="isLoading" class="timeline-panel__loading-state">
        <UiSpinner
          class="timeline-panel__loading-indicator"
          label="加载时间线"
          :decorative="false"
          :size="32"
        />
        <p>加载时间线...</p>
      </div>

      <ProductStatusBanner
        v-else-if="pendingMessage"
        aria-live="polite"
        class="timeline-panel__status-banner"
        icon-name="refresh"
        title="时间线生成中"
        tone="neutral"
      >
        {{ pendingMessage }}
      </ProductStatusBanner>

      <ProductEmptyState
        v-else-if="!hasTimelineData"
        class="timeline-panel__empty"
        icon-name="bar-chart"
        title="时间线尚未生成"
        description="完成漫画分析后会自动生成时间线，或点击下方按钮手动生成"
      >
        <template #actions>
          <UiButton
            variant="primary"
            size="sm"
            :disabled="isRegenerating"
            @click="regenerateTimeline"
          >
            {{ isRegenerating ? '生成中...' : '生成时间线' }}
          </UiButton>
        </template>
      </ProductEmptyState>

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

        <div v-if="isEnhancedData && plotArcs.length > 0" class="timeline-panel__section">
          <h4 class="timeline-panel__section-title">
            <UiIcon name="book-marked" size="16" />
            <span>剧情发展</span>
          </h4>
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

        <div v-if="plotThreads.length > 0" class="timeline-panel__section">
          <h4 class="timeline-panel__section-title">
            <UiIcon name="link" size="16" />
            <span>伏笔与线索</span>
          </h4>
          <PlotThreadsList :threads="plotThreads" />
        </div>

        <UiButton
          v-if="hasMoreTimeline"
          class="timeline-panel__load-more"
          variant="secondary"
          size="sm"
          :disabled="isLoadingMore"
          @click="loadMoreTimeline"
        >
          {{ isLoadingMore ? '加载中...' : '加载更多时间线内容' }}
        </UiButton>
      </template>
    </div>
  </div>
</template>

<style scoped>
.timeline-panel {
  --timeline-panel-card-shadow: var(--shadow-medium);
  --timeline-panel-character-shadow: var(--shadow-soft);
}

.timeline-panel__body {
  position: relative;
  padding: 20px;
}

.timeline-panel__status-banner {
  margin-bottom: 12px;
}

.timeline-panel__loading-state {
  display: grid;
  justify-items: center;
  gap: 12px;
  padding: 40px;
  color: var(--insight-text-secondary);
  text-align: center;
}

.timeline-panel__loading-indicator {
  color: var(--insight-action-primary);
}

.timeline-panel__empty {
  --product-empty-state-min-height: 280px;

  padding-block: 48px;
}

.timeline-panel__section {
  margin-bottom: 28px;
}

.timeline-panel__section-title {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin: 0 0 16px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--insight-action-primary);
  color: var(--insight-text-primary);
  font-weight: 600;
  font-size: 16px;
}

.timeline-panel__load-more {
  width: 100%;
}
</style>
